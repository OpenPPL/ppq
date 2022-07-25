# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理，并进行速度测试
# 目前 GPU 上 tensorRT 是跑的最快的部署框架 ...
# ---------------------------------------------------------------

import numpy as np
import tensorrt as trt
import trt_infer
from tqdm import tqdm

# fp32 - 355
# int8(trt) - 590
# int8(ppq) - 530

# int8 / fp32 ~ 70%
# trt > ppq > fp32

# Nvidia Nsight Performance Profile
ENGINE_PATH = 'Output/yolov6s(b32_int8).engine'
BATCH_SIZE  = 32
INPUT_SHAPE = [BATCH_SIZE, 3, 640, 640]
BENCHMARK_SAMPLES = 512

print(f'Benchmark with {ENGINE_PATH}')
logger = trt.Logger(trt.Logger.ERROR)
with open(ENGINE_PATH, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
    inputs[0].host = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)

    for _ in tqdm(range(BENCHMARK_SAMPLES), desc=f'Benchmark ...'):
        trt_infer.do_inference(
            context, bindings=bindings, inputs=inputs, 
            outputs=outputs, stream=stream, batch_size=BATCH_SIZE)

