import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import math
from ppq import *

import sys
sys.path.append("../detection")

# from memory_profiler import profile


class HostDeviceMem(object):
    #Simple helper data class that's a little nicer to use than a 2-tuple.
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

# @profile
def alloc_buf_N(engine,data,binding_shapes):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []

    stream = cuda.Stream()

    data_type = []

    for binding in engine:

        # import pdb
        # pdb.set_trace()

        if engine.binding_is_input(binding):
            size = data.shape[0]*data.shape[1]*data.shape[2]*data.shape[3]
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            data_type.append(dtype)

            # 在host上分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            
            # 在显存（Device）中分配空间
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        
        else:
            binding_shape = binding_shapes[binding]

            size = trt.volume(binding_shape) * engine.max_batch_size

            # 在host上分配内存
            host_mem = cuda.pagelocked_empty(size, data_type[0])

            # 在显存（Device）中分配空间
            device_mem = cuda.mem_alloc(host_mem.nbytes)            
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
            # print("outputs alloc_buf_N", outputs)
    return inputs, outputs, bindings, stream



def do_inference_v2(engine, inputs, bindings, outputs, stream, data):
    """
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """

    # Transfer input data to the GPU.
    # 拷贝内存从host到device
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
        # cuda.memcpy_htod(inp.device, inp.host)
    # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # context.active_optimization_profile = 0 # add

    with engine.create_execution_context() as context:
        context.set_binding_shape(0, data.shape)

        # Run inference.
        # context.execute_v2(bindings=bindings)
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
            # cuda.memcpy_dtoh(out.host, out.device)
        # [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        # 将系统缓冲区中的内容写回磁盘，以确保数据同步。
    return [out.host for out in outputs] 


trt_logger = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    print("\033[1;32musing %s\033[0m" % engine_path)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class DynamicTrtInferenceModel:
    def __init__(self,model_path) -> None:
        self.engine = load_engine(model_path)
        
    
    # @ profile
    def __call__(self,input_tensor):
        input_tensor = convert_any_to_numpy(input_tensor)

        # 确定输出的shape,和输入相关
        binding_shapes = {}
        h,w = input_tensor.shape[2:]
        h,w = h//8,w//8
        for i in range(1,6):
            binding_shapes[f"output{i}"] = (1,720,h,w)
            binding_shapes[f"output{i+5}"] = (1,36,h,w)
            h = math.ceil(h/2)
            w = math.ceil(w/2)

        inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(self.engine,input_tensor,binding_shapes)
        inputs_alloc_buf[0].host = np.ascontiguousarray(input_tensor)
        outputs = do_inference_v2(self.engine, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf,stream_alloc_buf, input_tensor)
        for i in range(len(outputs)):
            outputs[i] = outputs[i].reshape(binding_shapes[self.engine[i+1]])


        trt_outputs = [None for _ in range(len(outputs))]
        trt_outputs[:5] = outputs[5:]
        trt_outputs[5:] = outputs[:5]
        return trt_outputs

