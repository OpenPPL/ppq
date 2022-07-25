# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理，并进行速度测试
# 目前 GPU 上 tensorRT 是跑的最快的部署框架 ...
# ---------------------------------------------------------------

import numpy as np
import tensorrt as trt

import trt_infer
# Conv_41 + PWN(PWN(Sigmoid_42), Mul_43): 0.028672ms
# Conv_41: 0.045056ms
# PWN(PWN(Sigmoid_42), Mul_43): 0.03584ms

# Nvidia Nsight Performance Profile
ENGINE_PATH = 'Output/yolov5s.v5(ppq).engine'
BATCH_SIZE  = 1
INPUT_SHAPE = [BATCH_SIZE, 3, 640, 640]

logger = trt.Logger(trt.Logger.ERROR)
with open(ENGINE_PATH, 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

with engine.create_execution_context() as context:
    context.profiler = trt.Profiler()
    inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
    inputs[0].host = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)

    trt_infer.do_inference(
        context, bindings=bindings, inputs=inputs, 
        outputs=outputs, stream=stream, batch_size=BATCH_SIZE)

