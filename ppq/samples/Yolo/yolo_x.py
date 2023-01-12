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