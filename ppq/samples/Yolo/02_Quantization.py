# 这里我们展示两种不同的方法去生成 TensorRT Engine

# Plan B: PPQ 导出 engine

import os
import torchvision.transforms as transforms
from PIL import Image
from ppq import *
from ppq.api import *

ONNX_PATH        = 'models/yolov5s6.onnx'      # 你的模型位置
OUTPUT_PATH      = 'Output'                    # 生成的量化模型的位置
CALIBRATION_PATH = 'imgs'                      # 校准数据集
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
    imgs.append(img) # img is 0 - 1

from torch.utils.data import DataLoader
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)

with ENABLE_CUDA_KERNEL():
    qir = quantize_onnx_model(
        platform=TargetPlatform.TRT_INT8,
        onnx_import_file=ONNX_PATH, 
        calib_dataloader=dataloader, 
        calib_steps=32, device=EXECUTING_DEVICE,
        input_shape=[BATCHSIZE, 3, 640, 640], 
        collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    snr_report = graphwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    snr_report = layerwise_error_analyse(
        graph=qir, running_device=EXECUTING_DEVICE, 
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    export_ppq_graph(
        qir, platform=TargetPlatform.TRT_INT8, 
        graph_save_to=OUTPUT_PATH.join('/INT8.onnx'), 
        config_save_to=OUTPUT_PATH.join('/INT8.json'))
    
    from ppq.utils.TensorRTUtil import build_engine, Benchmark, Profiling
    build_engine(
        onnx_file=OUTPUT_PATH.join('/INT8.onnx'), 
        engine_file=OUTPUT_PATH.join('/INT8.engine'), int8=True, 
        int8_scale_file=OUTPUT_PATH.join('/INT8.json'))

    Benchmark(OUTPUT_PATH.join('/INT8.engine'))
    Profiling(OUTPUT_PATH.join('/INT8.engine'))