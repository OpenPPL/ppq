from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model, QuantizationSettingFactory, load_graph, dump_torch_to_onnx

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.

def load_calibration_dataset() -> Iterable:
    # for network training, you should better prepare your calibration dataset
    # make sure that your dataset has 8 ~ 512 batches of data
    # here out dataset are created with 32 batches.
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32 * BATCHSIZE)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

# Load a pretrained mobilenet v2 model
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# create a setting for quantizing your network with PPL CUDA.
quant_setting = QuantizationSettingFactory.pplcuda_setting()
quant_setting.equalization = True # use layerwise equalization algorithm.
quant_setting.dispatcher   = 'conservative' # dispatch this network in conservertive way.

quant_setting.advanced_optimization = True # train your network for better quantization
quant_setting.advanced_optimization_setting.lr = 1e-3 # change this for better performance

# Load training data for creating a calibration dataloader.
# Set shuffle=False when training your network.
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

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(graph=quantized, platform=PLATFORM, 
                 graph_save_to='Output/quantized(onnx).onnx', 
                 config_save_to='Output/quantized(onnx).json')
