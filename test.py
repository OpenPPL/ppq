import torchvision
import torch
import ppq
import ppq.api as API

calibration_dataloader = [torch.rand(size=[1, 3, 224, 224]).cuda()]
model = torchvision.models.AlexNet().cuda()

with API.ENABLE_CUDA_KERNEL():
    quantized = API.quantize_torch_model(
        model=model, calib_dataloader=calibration_dataloader, 
        calib_steps=8, input_shape=[1, 3, 224, 224], platform=ppq.TargetPlatform.METAX_INT8_T)

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