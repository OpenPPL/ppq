# 这里我们展示两种不同的方法去生成 TensorRT Engine

# Plan B: PPQ 导出 engine

import os
import torchvision.transforms as transforms
from PIL import Image
from ppq import *
from ppq.api import *

ONNX_PATH        = 'models/yolov5s.v5.onnx'       # 你的模型位置
ENGINE_PATH      = 'Output/yolo5s.v5(ppq_2).onnx' # 生成的 Engine 位置
CALIBRATION_PATH = 'imgs'                         # 校准数据集
BATCHSIZE        = 1
EXECUTING_DEVICE = 'cuda'
# create dataloader
imgs = []
trans = transforms.Compose([
    transforms.Resize([640, 640]),  # [h,w]
    transforms.ToTensor(),
])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img)

from ppq.quantization.quantizer import TensorRTQuantizer
class MyTensorRTQuantizer(TensorRTQuantizer):
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        config = super().init_quantize_config(operation)
        if operation.type == 'Mul':
            config.input_quantization_config[0].state = QuantizationStates.FP32
            config.input_quantization_config[1].state = QuantizationStates.FP32
        return config


register_network_quantizer(MyTensorRTQuantizer, TargetPlatform.TRT_INT8)
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)
s          = QuantizationSettingFactory.default_setting()

s.dispatching_table.append('Mul_49', platform=TargetPlatform.TRT_INT8)
s.dispatching_table.append('Mul_69', platform=TargetPlatform.TRT_INT8)

with ENABLE_CUDA_KERNEL():
    qir = quantize_onnx_model(
        setting=s,
        platform=TargetPlatform.TRT_INT8,
        onnx_import_file=ONNX_PATH, 
        calib_dataloader=dataloader, 
        calib_steps=32, device=EXECUTING_DEVICE,
        input_shape=[BATCHSIZE, 3, 640, 640], 
        collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    snr_report = graphwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    export_ppq_graph(
        qir, platform=TargetPlatform.TRT_INT8, 
        graph_save_to=ENGINE_PATH)