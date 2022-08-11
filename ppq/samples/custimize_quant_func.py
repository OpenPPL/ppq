# This Example shows you how to replace quantization function with your own quant delegator.

from typing import Iterable

import torch
import torchvision
from ppq import *
from ppq.api import quantize_torch_model
from torch.utils.data import DataLoader
from tqdm import tqdm

class MyQuantDelegator(TorchQuantizeDelegator):
    """Use This class to realize your quantization logic.

    Inherit class TorchQuantizeDelegate, implement interface __call__, then
    register your delegator with executor.register_quantize_delegate
    """
    def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            raise ValueError('Sorry, this delegator handles only Symmetrical Quantizations.')
        print('You are invoking cusitmized quant function now.')
        return torch.round(tensor / config.scale) * config.scale

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8 # identify a target platform for your network.

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
quantized = quantize_torch_model(
    model=model, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# build an executor:
executor = TorchExecutor(graph=quantized, device=DEVICE)

# register quant delegator
for var in quantized.variables.values():
    if isinstance(var, QuantableVariable):
        if var.source_op_config is not None:
            executor.register_quantize_delegate(var.source_op_config, MyQuantDelegator)

# run with your network, results are torch.Tensors
for data in tqdm(calibration_dataloader, desc='Running with executor.'):
    results = executor.forward(inputs=data.to(DEVICE))
    break

# remove delegators
for var in quantized.variables.values():
    if isinstance(var, QuantableVariable):
        if var.source_op_config is not None:
            executor.remove_quantize_delegate(var.source_op_config)
