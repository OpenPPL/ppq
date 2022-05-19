import torch
import torch.utils.data
import torchvision
from absl import logging

# 装一下下面这个库
from pytorch_quantization import nn as quant_nn

logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook

from pytorch_quantization import quant_modules

# 调用这个 quant_modules.initialize()
# 然后你正常训练就行了 ...
quant_modules.initialize()

model = torchvision.models.resnet50()
model.cuda()

# Quantization Aware Training is based on Straight Through Estimator (STE) derivative approximation. 
# It is some time known as “quantization aware training”. 
# We don’t use the name because it doesn’t reflect the underneath assumption. 
# If anything, it makes training being “unaware” of quantization because of the STE approximation.

# After calibration is done, Quantization Aware Training is simply select a training schedule and continue training the calibrated model. 
# Usually, it doesn’t need to fine tune very long. We usually use around 10% of the original training schedule, 
# starting at 1% of the initial training learning rate, 
# and a cosine annealing learning rate schedule that follows the decreasing half of a cosine period, 
# down to 1% of the initial fine tuning learning rate (0.01% of the initial training learning rate).

# Quantization Aware Training (Essentially a discrete numerical optimization problem) is not a solved problem mathematically.
# Based on our experience, here are some recommendations:

# For STE approximation to work well, it is better to use small learning rate. 
# Large learning rate is more likely to enlarge the variance introduced by STE approximation and destroy the trained network.

# Do not change quantization representation (scale) during training, at least not too frequently. 
# Changing scale every step, it is effectively like changing data format (e8m7, e5m10, e3m4, et.al) every step, 
# which will easily affect convergence.

# https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/examples/finetune_quant_resnet50.ipynb

def export_onnx(model, onnx_filename, batch_onnx):
    model.eval()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    opset_version = 13

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, 224, 224, device='cuda') #TODO: switch input dims by model
    torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, opset_version=opset_version, enable_onnx_checker=False, do_constant_folding=True)
    return True
