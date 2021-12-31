from typing import Iterable

import torch
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_onnx_model

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.

# ---------------------------------------------------------------------------------------
# 
# This is a empty platform for you to implement custimized logic.
# rewrite code inside ppq/quantization/quantizer/MyQuantizer.py to create your quantizer
#
# ---------------------------------------------------------------------------------------
PLATFORM = TargetPlatform.EXTENSION
ONNX_PATH = 'Models/cls_model/mobilenet_v2.onnx'

def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

quant_setting = QuantizationSettingFactory.default_setting()
# ---------------------------------------------------------------------------------------
# 
# extension setting and extension pass are empty
# you can rewrite their code to optimize your network as will.
# setting:           ppq/api/setting.py
# optimization pass: ppq/quantization/optim/extension.py
# parameter passing: ppq/quantization/quantizer/base.py
#
# ---------------------------------------------------------------------------------------
quant_setting.extension = True

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset, 
    batch_size=BATCHSIZE, shuffle=True)

# quantize your model.
# ---------------------------------------------------------------------------------------
# 
# operation execution logic are inherit from default.
# you can write your custimize logic with ppq/executor/op/torch/extension.py
#
# ---------------------------------------------------------------------------------------
quantized = quantize_onnx_model(
    onnx_import_file=ONNX_PATH, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE, 
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
# ---------------------------------------------------------------------------------------
# 
# export logic is empty
# you can rewrite code inside ppq/parser/extension.py to export ppq graph to disk.
#
# ---------------------------------------------------------------------------------------
export_ppq_graph(graph=quantized, platform=PLATFORM, 
                 graph_save_to='Output/quantized(onnx).onnx', 
                 config_save_to='Output/quantized(onnx).json')
