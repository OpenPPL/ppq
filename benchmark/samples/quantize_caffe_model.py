from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_caffe_model

BATCHSIZE = 32
INPUT_SHAPE = [3, 96, 96]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.
PROTO_PATH = 'Models/model.prototxt' # For successfully loading caffe model, .prototxt file is required.
MODEL_PATH = 'Models/model.caffemodel' # For successfully loading caffe model, .caffemodel file is required.

def load_calibration_dataset() -> Iterable:
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
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=BATCHSIZE, shuffle=True)

# quantize your model.
quantized = quantize_caffe_model(
    caffe_model_file=MODEL_PATH, caffe_proto_file=PROTO_PATH,
    calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(graph=quantized, platform=TargetPlatform.CAFFE,
                 graph_save_to='Output/quantized(caffe)',
                 config_save_to='Output/quantized(caffe).json')
