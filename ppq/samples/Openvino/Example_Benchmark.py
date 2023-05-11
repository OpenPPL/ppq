# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 openvino 对 PPQ 导出的模型进行推理
# 你需要注意，openvino 也可以运行各种各样的量化方案，你甚至可以用 tensorRT 的 policy
# 但总的来说，openvino 需要非对称量化的 activation 和对称量化的 weights
# 现在的写法针对单输入网络哦，多输入的你得自己改改
# ---------------------------------------------------------------

# For this onnx inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
import numpy as np
import openvino
import torch
from tqdm import tqdm
import time

from ppq import *
from ppq.api import *

QUANT_PLATFROM    = TargetPlatform.OPENVINO_INT8
BATCHSIZE         = 1
DEVICE            = 'cuda'
INPUTSHAPE        = [BATCHSIZE, 3, 640, 640]
SAMPLES           = [torch.rand(size=INPUTSHAPE) for _ in range(256)]
BENCHMARK_SAMPLES = 512
MODEL_PATH        = 'Models/yolox_s.onnx'
VALIDATION        = False

with ENABLE_CUDA_KERNEL():
    quantized = quantize_onnx_model(
        onnx_import_file=MODEL_PATH, calib_dataloader=SAMPLES, collate_fn=lambda x: x.to(DEVICE),
        calib_steps=32, input_shape=INPUTSHAPE,
        setting=QuantizationSettingFactory.default_setting(),
        platform=QUANT_PLATFROM)

    graphwise_error_analyse(graph=quantized, running_device='cuda', 
                            dataloader=SAMPLES, collate_fn=lambda x: x.cuda(), steps=32)

    export_ppq_graph(
        graph=quantized, platform=TargetPlatform.ONNX,
        graph_save_to='FP32.onnx')

    export_ppq_graph(
        graph=quantized, platform=TargetPlatform.OPENVINO_INT8,
        graph_save_to='INT8.onnx')

from ppq.utils.OpenvinoUtil import Benchmark
Benchmark(ir_or_onnx_file='FP32.onnx', samples=500, jobs=4)
Benchmark(ir_or_onnx_file='INT8.onnx', samples=500, jobs=4)