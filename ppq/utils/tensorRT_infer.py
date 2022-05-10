# This file contains a helper class for tensorRT engine inference.
#   usage:
#       engine = TensorRTEngine('trt.engine')
#       print(engine.forward(torch.zeros(1, 3, 224, 224))[0].shape)

from typing import List, Union

import numpy
import torch
from ppq.core import convert_any_to_numpy, convert_any_to_torch_tensor

try:
    import pycuda
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError as e:
    raise ImportError('Install tensorRT and related dependencies before using tensorRT export.')
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class HostDeviceMem(object):
    #Simple helper data class that's a little nicer to use than a 2-tuple.
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)
    def __repr__(self):
        return self.__str__()


class TensorRTEngine():
    """TensorRT Engine is a helper class for tensorRT engine inference. it
    wraps tensorRT Engine as a normal torch module.

    ATTENTION: you can not use this class together with other pytorch function.
        it will cause some unexpected CUDNN errors.
    """
    def __init__(self, engine_path: str) -> None:
        self._runtime = trt.Runtime(TRT_LOGGER)
        self._engine  = self._runtime.deserialize_cuda_engine(self.load_from_file(engine_path))
        self._cuda_stream = cuda.Stream()

    def load_from_file(self, engine_path: str) -> bytes:
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f'Can not locate TensorRT engine file at {engine_path}')
        with open(engine_path, 'rb') as file:
            return file.read()

    def alloc_buffer(self):
        inputs, outputs, bindings = [], [], []

        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding)) * self._engine.max_batch_size
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)

            # host_mem = [0. 0. 0. ... 0. 0. 0.],
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    def inference(self, inputs: List[HostDeviceMem], bindings: List[int],
        outputs: List[HostDeviceMem], context) -> List[numpy.ndarray]:
        """Inputs and outputs are expected to be lists of HostDeviceMem
        objects."""
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self._cuda_stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=self._cuda_stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self._cuda_stream) for out in outputs]
        # Synchronize the stream
        self._cuda_stream.synchronize()
        # Return only the host outputs.
        raws = [out.host for out in outputs]
        return [numpy.asarray(value) for value in raws]

    def prase_feed_value(self, inputs: Union[torch.Tensor, list, dict]) -> List[numpy.ndarray]:
        input_names = []
        for binding in self._engine:
            if self._engine.binding_is_input(binding):
                input_names.append(str(binding))

        feed_list = []
        if isinstance(inputs, torch.Tensor):
            if len(input_names) != 1: raise ValueError(
                f'TensorRT Engine needs {len(input_names)} input values, '
                'while only one torch Tensor was given.')
            feed_list.append(convert_any_to_numpy(inputs))
        elif isinstance(inputs, list):
            for value in inputs:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f'PPQ feeds tensorRT Engine with only torch.Tensor, '
                    f'however {type(value)} was gievn.')
                feed_list.append(convert_any_to_numpy(value))
        elif isinstance(inputs, dict):
            feed_list = [None for _ in input_names]
            for name, value in inputs.items():
                if name not in input_names:
                    raise KeyError(f'Can not feed value of variable: {name}, '
                    'can not find it in tensorRT engine.')
                feed_list[input_names.index(name)] = convert_any_to_numpy(value)
        else:
            raise TypeError('Can not parse feeding value.')

        return feed_list

    def forward(self, inputs: Union[torch.Tensor, list, dict]) -> List[torch.Tensor]:
        with self._engine.create_execution_context() as context:
            inputs_mem, outputs_mem, bindings = self.alloc_buffer()

            inputs = self.prase_feed_value(inputs)
            # feed value towards host memory
            for input_mem, input_value in zip(inputs_mem, inputs):
                input_mem.host = numpy.ascontiguousarray(input_value)

            # do inference
            result = self.inference(inputs=inputs_mem, bindings=bindings, outputs=outputs_mem, context=context)
            return [convert_any_to_torch_tensor(value) for value in result]

if __name__ == '__main__':
    engine = TensorRTEngine('trt.engine')
    print(engine.forward(torch.zeros(1, 3, 224, 224))[0].shape)
