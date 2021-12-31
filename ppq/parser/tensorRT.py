import os
from typing import Callable, Iterable, List

import numpy as np

try:
    import pycuda
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError as e:
    raise ImportError('Install tensorRT and related dependencies before using tensorRT export.')

import torch
from ppq.core import convert_any_to_numpy, ppq_warning
from ppq.IR import BaseGraph
from tqdm import tqdm

from .onnx_exporter import OnnxExporter

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class EngineCalib(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader: Iterable, batchsize: int, collate_fn: Callable = None) -> None:
        super().__init__()
        self._dataloader = dataloader
        self._collate_fn = collate_fn
        self._batchsize  = batchsize
        self._cur_iter   = dataloader.__iter__()
        self._progress_bar = tqdm(desc='TensorRT Calibration Progress')

    def get_batch(self, names: List[str]) -> List[pycuda._driver.DeviceAllocation]:
        self._progress_bar.update(1)
        # print('TensorRT is Calibrating your model, Please wait...')
        def send_numpy_to_cuda(numpy_value: np.ndarray) -> pycuda._driver.DeviceAllocation:
            size     = numpy_value.nbytes
            cuda_ptr = cuda.mem_alloc(size)
            cuda.memcpy_htod(cuda_ptr, np.ascontiguousarray(numpy_value))
            return cuda_ptr

        try:
            # get next calibration data.
            batch = next(self._cur_iter)
        except StopIteration as e:
            # at the end of dataloader, we just return None to notice tensorRT stop calibration.
            return None

        if self._collate_fn is not None: batch = self._collate_fn(batch)
        assert type(batch) in {torch.Tensor, list, dict}, (
            f'Calibration Dataloader should only contains torch.Tensor, list of tensors, dict of tensors. '
            f'{type(batch)} was given however.')
        if isinstance(batch, list): raise TypeError(
            'Passing calibration data with list is accepted with ppq, '
            'however it is not support by tensorRT. Make your calibration data as a named dict instead.')

        # if input is a single tensor, just send tensor data to cuda.
        if isinstance(batch, torch.Tensor):
            # pdb.set_trace()
            return [send_numpy_to_cuda(convert_any_to_numpy(batch))]
        # else if input is a list of tensors, send all of them to cuda.
        elif isinstance(batch, dict):
            cuda_ptrs = []
            for name in names:
                if name not in batch: raise KeyError(
                    f"TensorRT calibrator are requiring input data of variable {name}, "
                    "however we can not find it with your batch data.")

                data = batch[name]
                assert isinstance(data, torch.Tensor), (
                    f"TensorRT calibration data should only be torch.Tensor. However {type(data)} was given.")
                cuda_ptrs.append(send_numpy_to_cuda(convert_any_to_numpy(data)))
            return cuda_ptrs
        else: raise Exception('Oops, you shuold never reach here.')

    def get_batch_size(self):
        return self._batchsize

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        return None

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        return None


class TensorRTExporter(OnnxExporter):

    def __init__(self) -> None:
        super().__init__()

        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 8 * (2 ** 30)  # 8 GB

    def export(
        self, file_path: str, onnx_file_path: str, graph: BaseGraph, 
        calib_dataloader: Iterable, batchsize: int, collate_fn: Callable = None):

        if os.path.exists(file_path) or os.path.exists(onnx_file_path):
            ppq_warning(
                f'Path "{file_path}" | "{onnx_file_path}" already been taken by other file. '
                'Those file will overwrite by PPQ.')

        # invoke super class method to dump onnx model.
        super().export(file_path=onnx_file_path, config_path=None, graph=graph)

        # parse onnx to tensorRT network
        network = self.parse_onnx_to_trt(onnx_path=onnx_file_path)
        
        # create tensorRT engine (as a memory flow)
        engine_mem_flow = self.create_engine(
            network=network, dataloader=calib_dataloader, 
            batchsize=batchsize, collate_fn=collate_fn)

        # dump tensorRT engine.
        with open(file_path, 'wb') as file: file.write(engine_mem_flow)

    def parse_onnx_to_trt(self, onnx_path: str) -> trt.INetworkDefinition:
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """

        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        network = self.builder.create_network(network_flags)
        parser  = trt.OnnxParser(network, TRT_LOGGER)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            parsing_flag = parser.parse(f.read())
            if not parsing_flag:
                err_msg = ''.join([str(parser.get_error(error)) + '\n' for error in range(parser.num_errors)])
                raise RuntimeError('Parsing Onnx failed, with following error(s): \n' + err_msg)
        return network

    def create_engine(self, network: trt.INetworkDefinition, dataloader: Iterable, 
        batchsize: int, collate_fn: Callable = None) -> trt.IHostMemory:
        """
            Build the TensorRT engine and serialize it to disk.
        """
        with trt.Builder(TRT_LOGGER) as builder, trt.Runtime(TRT_LOGGER) as runtime:
            # We set the builder batch size to be the same as the calibrator's, as we use the same batches
            # during inference. Note that this is not required in general, and inference batch size is
            # independent of calibration batch size.

            config = builder.create_builder_config()
            config.max_workspace_size = 8 * (1 << 30)
            config.set_flag(trt.BuilderFlag.INT8)

            config.int8_calibrator = EngineCalib(
                dataloader=dataloader, batchsize=batchsize, collate_fn=collate_fn)
            # Build engine and do int8 calibration.
            buffer = builder.build_serialized_network(network, config)
        return buffer
