# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 Onnxruntime 对 PPQ 导出的模型进行推理
# Onnxruntime 提供一系列 providers 实现不同硬件上的神经网络推理

# CPUExecutionProvider, CUDAExecutionProvider 是 Onnxruntime 官方提供的
# TensortExecutionProvider 是 Nvidia 提供的
# 不同 Provider 对模型格式有不一样的要求，PPQ 导出的是 CPUExecutionProvider 格式的模型

# Onnxruntime 没写 INT8 算子的 CUDA 实现，因此当你的模型使用 Onnxruntime 进行部署时，如果使用
# CUDAExecutionProvider, 你无需考虑量化加速
# ---------------------------------------------------------------

import torchvision
import torch
import ppq
import ppq.api as API

calibration_dataloader = [torch.rand(size=[1, 3, 224, 224]).cuda()]
model = torchvision.models.shufflenet_v2_x1_0().cuda()

with API.ENABLE_CUDA_KERNEL():
    quantized = API.quantize_torch_model(
        model=model, calib_dataloader=calibration_dataloader, 
        calib_steps=8, input_shape=[1, 3, 224, 224], platform=ppq.TargetPlatform.ONNXRUNTIME)

API.export_ppq_graph(
    quantized, platform=ppq.TargetPlatform.ONNXRUNTIME, 
    graph_save_to='Quantized.onnx')

API.export_ppq_graph(
    quantized, platform=ppq.TargetPlatform.ONNX, 
    graph_save_to='FP32.onnx')

from ppq.utils.OnnxruntimeUtil import Benchmark, Profile

Benchmark('FP32.onnx', providers=['CPUExecutionProvider'])
Benchmark('Quantized.onnx', providers=['CPUExecutionProvider'])

Profile('FP32.onnx', providers=['CPUExecutionProvider'])
Profile('Quantized.onnx', providers=['CPUExecutionProvider'])