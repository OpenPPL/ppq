# This Example shows you how to analyse your quantized network.
# Check quantization error for each layer.

from typing import Iterable

import torch
import torchvision
from ppq import QuantableOperation, QuantizationSettingFactory, TargetPlatform
from ppq.api import quantize_torch_model
from ppq.core.quant import QuantizationStates
from torch.utils.data import DataLoader

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

# dequantize with operation.dequantize() function:
for operation in quantized.operations.values():

    if isinstance(operation, QuantableOperation):
        # all parameters of this operation will be dequantized, baked value will be replaced.
        # input and output of this operation will not be quantized since now.
        operation.dequantize()

# restore quantization state:
for operation in quantized.operations.values():

    if isinstance(operation, QuantableOperation):
        # all parameters of this operation will restore its quantization result.
        # input and output of this operation will be quantized since now.
        operation.restore_quantize_state()

# manually dequantize an operation:
for operation in quantized.operations.values():
    if isinstance(operation, QuantableOperation):
        for cfg, var in operation.config_with_variable:

            if var.is_parameter and cfg.state == QuantizationStates.BAKED:
                print(f'Variable {var.name} is pre-baked, simply overriding its state takes no effects.')
            else:
                # once state is changed to FP32
                # executor will skip this quantization during executing.
                cfg.state = QuantizationStates.FP32
