# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理，并进行速度测试
# 目前 GPU 上 tensorRT 是跑的最快的部署框架 ...
# ---------------------------------------------------------------

import time
import numpy as np
import tensorrt as trt
import torch
import torchvision
import torchvision.models
from tqdm import tqdm

import trt_infer
from ppq import *
from ppq.api import *

# Nvidia Nsight Performance Profile
QUANT_PLATFROM   = TargetPlatform.TRT_INT8
BATCHSIZE        = 1
SAMPLES          = [torch.zeros(size=[BATCHSIZE, 3, 640, 640]) for _ in range(32)]
DEVICE           = 'cuda'
MODEL_PATH       = 'model_int8.engine'
CFG_VALID_RESULT = False

def infer_trt(model_path: str, samples: List[np.ndarray]) -> List[np.ndarray]:
    """ Run a tensorrt model with given samples
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []
    if CFG_VALID_RESULT:
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            for sample in tqdm(samples, desc='TensorRT is running...'):
                inputs[0].host = convert_any_to_numpy(sample)
                [output] = trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)[0]
                results.append(convert_any_to_torch_tensor(output).reshape([-1, 1000]))
    else:
        with engine.create_execution_context() as context:
            context.profiler = trt.Profiler()
            inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
            inputs[0].host = convert_any_to_numpy(samples[0])
            for sample in tqdm(samples, desc='TensorRT is running...'):
                trt_infer.do_inference(
                    context, bindings=bindings, inputs=inputs, 
                    outputs=outputs, stream=stream, batch_size=1)
    return results

infer_trt(model_path=MODEL_PATH, samples=SAMPLES)