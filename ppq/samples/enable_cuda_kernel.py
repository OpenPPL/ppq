# Since ppq 0.6.4, PPQ_CONFIG.USING_CUDA_KERNEL = False is the defualt execution option in ppq.
# However you should notice that if you are able to compile ppq kernel functions, the execution speed wiil boost at least 3x.
# This example will show you how to enable kernel function within ppq.
# if you want to use kernel function everywhere, just rewrite ppq.core.config.PPQ_CONFIG.USING_CUDA_KERNEL = True

from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model, ENABLE_CUDA_KERNEL

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.

def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)

# Load a pretrained mobilenet v2 model
model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model = model.to(DEVICE)

# use this to wrap up your code
# all functions inside ENABLE_CUDA_KERNEL will switch to ppq kernel functions.
with ENABLE_CUDA_KERNEL():

    # create a setting for quantizing your network with PPL CUDA.
    quant_setting = QuantizationSettingFactory.pplcuda_setting()
    quant_setting.equalization     = True # use layerwise equalization algorithm.
    quant_setting.dispatcher       = 'conservative' # dispatch this network in conservertive way.
    quant_setting.lsq_optimization = True  # finetune your network.

    # Load training data for creating a calibration dataloader.
    calibration_dataset = load_calibration_dataset()
    calibration_dataloader = DataLoader(
        dataset=calibration_dataset,
        batch_size=BATCHSIZE, shuffle=True)

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