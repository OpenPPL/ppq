import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import cv2
import pdb

class HostDeviceMem(object):
    #Simple helper data class that's a little nicer to use than a 2-tuple.
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

def alloc_buf_N(engine, input_data):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []

# import pdb
# pdb.set_trace()

    stream = cuda.Stream()

    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = input_data.size * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        # 在内存（Host）中分配空间
        host_mem = cuda.pagelocked_empty(size, dtype)
        # host_mem = [0. 0. 0. ... 0. 0. 0.],

        # 在显存（Device）中分配空间
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            # print("inputs alloc_buf_N", inputs)
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            # print("outputs alloc_buf_N", outputs)
        return inputs, outputs, bindings, stream


def do_inference_v2(context, inputs, bindings, outputs, stream):
    """
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """

    # Transfer input data to the GPU.
    # 拷贝内存从host到device
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
        # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

        context.set_binding_shape(0, (1, 3, 223, 224))

        # Run inference.
        context.execute_v2(bindings=bindings)

    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
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


engine_path ='engines/2309_model_normal_dynamic.engine'
bin_file ='1_3_250_300.bin'


if __name__ == '__main__':
    src_data = np.fromfile(bin_file,dtype=np.float32)
    inputs = np.array(src_data).astype(np.float32)

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # data = np.reshape(inputs,(3,224,224))
    # data = np.expand_dims(data, 0) # (1, 3, 224, 224)

    data = np.reshape(inputs,(1,3,250,300))

    context.set_binding_shape(0, data.shape)

    d_input = cuda.mem_alloc(data.nbytes) # data.nbytes:1x3x224x224x4

    output_shape = context.get_binding_shape(1)
    buffer = np.empty(output_shape, dtype=np.float32)

    d_output = cuda.mem_alloc(buffer.nbytes)

    cuda.memcpy_htod(d_input, data)
    bindings = [d_input, d_output]

    # 进行推理,并将结果从gpu拷贝到cpu
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(buffer, d_output)
    output = buffer.reshape(output_shape)

    print(output.shape)
    print(output)


