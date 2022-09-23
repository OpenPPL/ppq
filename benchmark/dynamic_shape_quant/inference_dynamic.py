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

        # context = self.engine.create_execution_context()

        inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(self.engine,input_tensor,binding_shapes)
        inputs_alloc_buf[0].host = np.ascontiguousarray(input_tensor)
        outputs = do_inference_v2(self.engine, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf,stream_alloc_buf, input_tensor)
        for i in range(len(outputs)):
            outputs[i] = outputs[i].reshape(binding_shapes[self.engine[i+1]])


        trt_outputs = [None for _ in range(len(outputs))]
        trt_outputs[:5] = outputs[5:]
        trt_outputs[5:] = outputs[:5]
        return trt_outputs


# with open("/home/geng/tinyml/ppq/benchmark/dynamic_input_model/images_list.txt","r") as f:
#    img_path_list =  f.readlines()
# img_path_list  = [x.rstrip() for x in img_path_list]
# img_path = img_path_list[0]



if __name__ == '__main__':


    img_path = '/home/geng/fiftyone/coco-2017/validation/data/000000397133.jpg'

    from PIL import Image
    engine_path ='/home/geng/tinyml/ppq/benchmark/dynamic_shape_quant/FP32_model/Retinanet-wo-dynamic-FP32.engine'

    from process import preprocess
    img = Image.open(img_path)
    src_data = preprocess(img,(1,3,480,640))
    # import ctypes
    # ctypes.CDLL("/home/geng/tinyml/ppq/benchmark/detection/lib/libmmdeploy_tensorrt_ops.so")

    from dataset import build_dataset
    from torch.utils.data import DataLoader
    ann_file = "/home/geng/fiftyone/coco-2017/validation/labels.json"
    data_root = '/home/geng/fiftyone/coco-2017/validation/data/'  # 数据的根路径。
    input_size = (480,640)
    from mmcv.parallel import collate
    dataset = build_dataset(ann_file=ann_file,data_root=data_root,input_size=input_size)
    dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate)

    input = next(iter(dataloader))["img"][0].numpy()
    # data = inputs = np.array(src_data).astype(np.float32)[None,:,:,:]
    print(f"input shape:{input.shape}")
    # data = np.reshape(inputs,(1,3,400,400))
    print("input",input)
    print("src_data",src_data)
    engine = load_engine(engine_path)


    binding_shapes = {}
    h,w = input.shape[2:]
    h,w = h//8,w//8
    for i in range(1,6):
        binding_shapes[f"output{i}"] = (1,720,h,w)
        binding_shapes[f"output{i+5}"] = (1,36,h,w)
        h = math.ceil(h/2)
        w = math.ceil(w/2)
        # print(h,w)

    context = engine.create_execution_context()

    inputs_alloc_buf, outputs_alloc_buf, bindings_alloc_buf, stream_alloc_buf = alloc_buf_N(engine,input,binding_shapes)


    # np.ascontiguousarray将一个内存不连续存储的数组转换为内存连续存储的数组
    inputs_alloc_buf[0].host = np.ascontiguousarray(input)

    # import pdb
    # pdb.set_trace()

    outputs = do_inference_v2(context, inputs_alloc_buf, bindings_alloc_buf, outputs_alloc_buf,stream_alloc_buf, input)
    # trt_feature = np.asarray(trt_feature)


    # trt_feature = trt_feature.reshape((1,16,300,300))
    # import pdb
    # pdb.set_trace()
    for i in range(len(outputs)):
        outputs[i] = outputs[i].reshape(binding_shapes[engine[i+1]])

    trt_outputs = [None for _ in range(len(outputs))]
    trt_outputs[:5] = outputs[5:]
    trt_outputs[5:] = outputs[:5]

    for o in trt_outputs:
        print("trt_output",o.shape)  

    import onnxruntime
    import torch
    from torch.nn.functional import mse_loss
    onnxruntime_model_path = "/home/geng/tinyml/ppq/benchmark/dynamic_shape_quant/FP32_model/Retinanet-wo-FP32.onnx"
    sess = onnxruntime.InferenceSession(path_or_bytes=onnxruntime_model_path, providers=['CUDAExecutionProvider'])
    input_placeholder_name = sess.get_inputs()[0].name
    outputnames = [x.name for x in sess.get_outputs()]

    ort_outputs = sess.run(input_feed={input_placeholder_name: input},output_names=outputnames)
    for o in ort_outputs:
        print("ort_output",o.shape)
    for ort_output,trt_output in zip(ort_outputs,trt_outputs):
        print(ort_output.shape,"mse_loss",mse_loss(torch.from_numpy(ort_output),torch.from_numpy(trt_output)))

    print(trt_outputs[1],ort_outputs[1])
    print(trt_outputs[5],ort_outputs[5])    
# [[0.77146703 0.1168441  0.11168882]]
# (1, 3)
