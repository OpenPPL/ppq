# This Example shows you how to analyse your quantized network.
# Check quantization error for each layer.

from typing import Iterable

import torch
import torchvision
from ppq import (QuantizationSettingFactory, TargetPlatform,
                 graphwise_error_analyse)
from ppq.api import quantize_torch_model
from torch.utils.data import DataLoader

from ppq.quantization.analyise.layerwise import layerwise_error_analyse, parameter_analyse

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8 # identify a target platform for your network.

def load_calibration_dataset() -> Iterable:
    # Any Iterable python object is acceptable for being a dataset in ppq.
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

# Load a pretrained mobilenet v2 model
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# create a setting for quantizing your network with PPL CUDA.
quant_setting = QuantizationSettingFactory.pplcuda_setting()
quant_setting.equalization = True # use layerwise equalization algorithm.
quant_setting.dispatcher   = 'conservative' # dispatch this network in conservertive way.

# Load training data for creating a calibration dataloader.
# Notice you can not set shuffle = True for analysing your network.
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset, 
    batch_size=BATCHSIZE, shuffle=False)

# quantize your model.
quantized = quantize_torch_model(
    model=model, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE, 
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

# invoke graph_similarity_analyse function to anaylse your network
reports = graphwise_error_analyse(
    graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
    dataloader=calibration_dataloader)

# WITH PPQ 0.6 or newer, you can invoke layerwise_error_analyse to get a more detailed report.
reports = layerwise_error_analyse(
    graph=quantized, running_device=DEVICE, collate_fn=collate_fn,
    dataloader=calibration_dataloader, interested_outputs='NETWORK OUTPUT VARIABLE NAME')

# WITH PPQ 0.6 or newer, you can invoke parameter_analyse to get a more detailed report.
parameter_analyse(graph=quantized)
