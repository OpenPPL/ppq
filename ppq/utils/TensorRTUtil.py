#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Iterable, Callable

import pycuda.driver as cuda
import tensorrt as trt
import torch
from tqdm import tqdm

from ppq.api import ENABLE_CUDA_KERNEL
from ppq import convert_any_to_numpy, convert_any_to_torch_tensor, TorchExecutor, torch_snr_error
from ppq.IR import BaseGraph
from ppq.quantization.analyse.util import MeasurePrinter

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 8 * (2 ** 30)  # 8 GB

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0
        self.builder.max_batch_size = self.batch_size

    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=25000,
                      calib_batch_size=8, calib_preprocessor=None):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        :param calib_preprocessor: The ImageBatcher preprocessor algorithm to use.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def setDynamicRange(network, json_file: str):
    """Sets ranges for network layers."""
    with open(json_file) as file:
        quant_param_json = json.load(file)
    act_quant = quant_param_json["act_quant_info"]

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            value = act_quant[input_tensor.name]
            tensor_max = abs(value)
            tensor_min = -abs(value)
            input_tensor.dynamic_range = (tensor_min, tensor_max)

    for i in range(network.num_layers):
        layer = network.get_layer(i)

        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)

            if act_quant.__contains__(tensor.name):
                value = act_quant[tensor.name]
                tensor_max = abs(value)
                tensor_min = -abs(value)
                tensor.dynamic_range = (tensor_min, tensor_max)


def build_engine(
    onnx_file: str, engine_file: str,
    fp16: bool = True, int8: bool = False, 
    int8_scale_file: str = None,
    explicit_batch: bool = True, 
    workspace: int = 4294967296, # 4GB
    ):
    TRT_LOGGER = trt.Logger()
    """
    Build a TensorRT Engine with given onnx model.

    Flag int8, fp16 specifies the precision of layer:
        For building FP32 engine: set int8 = False, fp16 = False, int8_scale_file = None
        For building FP16 engine: set int8 = False, fp16 = True, int8_scale_file = None
        For building INT8 engine: set True = False, fp16 = True, int8_scale_file = 'json file name'

    """

    if int8 is True:
        if int8_scale_file is None:
            raise ValueError('Build Quantized TensorRT Engine Requires a JSON file which specifies variable scales, '
                             'however int8_scale_file is None now.')

    builder = trt.Builder(TRT_LOGGER)
    if explicit_batch:
        network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    else: network = builder.create_network()
    
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.max_workspace_size = workspace

    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f'ONNX file {onnx_file} not found')

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    if fp16: config.set_flag(trt.BuilderFlag.FP16)
    if int8_scale_file is not None and int8:
        config.set_flag(trt.BuilderFlag.INT8)
        setDynamicRange(network, int8_scale_file)

    engine = builder.build_engine(network, config)

    with open(engine_file, "wb") as f:
        f.write(engine.serialize())


class MyProfiler(trt.IProfiler):
    def __init__(self, steps: int):
        trt.IProfiler.__init__(self)
        self.total_runtime = 0.0
        self.recorder = defaultdict(lambda: 0.0)
        self.steps = steps

    def report_layer_time(self, layer_name: str, ms: float):
        self.total_runtime += ms * 1000 / self.steps
        self.recorder[layer_name] += ms * 1000 / self.steps
    
    def report(self):
        MeasurePrinter(data=self.recorder, measure='RUNNING TIME(us)', order=None).print()
        print(f'\nTotal Inference Time: {self.total_runtime:.4f}(us)')
        print('You should notice that inference time != Lantancy, cause layer can be executed concurrently.')


def Benchmark(engine_file: str, steps: int = 10000) -> float:
    """ Benckmark TensorRT Engine """
    logger = trt.Logger(trt.Logger.INFO)
    import pycuda.autoinit
    
    with open(engine_file, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(context.engine)
        for input in inputs:
            input.host = convert_any_to_numpy(torch.rand(input.host.shape).float())

        tick = time.time()
        for _ in tqdm(range(steps), desc='TensorRT is running...'):
            do_inference_v2(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream)
        tok  = time.time()
        print(f'Time span: {tok - tick  : .4f} sec')
    return tick - tok


def Profiling(engine_file: str, steps: int = 1000):
    """ Profiling TensorRT Engine """
    logger = trt.Logger(trt.Logger.ERROR)
    import pycuda.autoinit
    
    with open(engine_file, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        context.profiler = MyProfiler(steps)
        inputs, outputs, bindings, stream = allocate_buffers(context.engine)

        for input in inputs:
            input.host = convert_any_to_numpy(torch.rand(input.host.shape).float())

        for _ in tqdm(range(steps), desc='TensorRT is running...'):
            do_inference_v2(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream)
        context.profiler.report()


def TestAlignment(engine_file: str, graph: BaseGraph, samples: Iterable, collate_fn: Callable = None) -> dict:
    """ Test Alignment with TensorRT Engine and PPQ Graph. """
    logger = trt.Logger(trt.Logger.ERROR)

    feed_dicts = []
    for sample in samples:
        if collate_fn is not None: sample = collate_fn(sample)
        feed_dict = {}
        if isinstance(sample, torch.Tensor):
            assert len(graph.inputs) == 1, 'Graph Needs More than 1 input tensor, however only 1 was given.'
            for name in graph.inputs:
                feed_dict[name] = sample
        elif isinstance(sample, list):
            for name, value in zip(graph.inputs, sample):
                feed_dict[name] = value
        elif isinstance(sample, dict):
            feed_dict = sample
        else:
            raise TypeError('Given Sample is Invalid.')
        feed_dicts.append(feed_dict)
    
    TensorRT_Results, PPQ_Results = [], []
    with ENABLE_CUDA_KERNEL():
        executor = TorchExecutor(graph)
        for feed_dict in tqdm(feed_dicts, desc='PPQ Infer...'):
            PPQ_Results.append([value.cpu() for value in executor.forward(feed_dict)])
    
    import pycuda.autoinit
    with open(engine_file, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(context.engine)

        for feed_dict in tqdm(feed_dicts, desc='TensorRT Infer...'):
            for name, input in zip(graph.inputs, inputs):
                input.host = convert_any_to_numpy(feed_dict[name])

            results = do_inference_v2(
                context, bindings=bindings, inputs=inputs, 
                outputs=outputs, stream=stream)
            
            TensorRT_Results.append([convert_any_to_torch_tensor(value) for value in results])

    collector = {}
    for ref, pred in zip(TensorRT_Results, PPQ_Results):
        for name in graph.outputs: collector[name] = 0.0
        for name, ref_, pred_ in zip(graph.outputs, ref, pred):
            collector[name] += torch_snr_error(pred_.reshape([1, -1]), ref_.reshape([1, -1])).item()

    for name, value in collector.items():
        collector[name] = value / len(samples)
    return collector