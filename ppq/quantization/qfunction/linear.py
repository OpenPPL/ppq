from typing import Any

import torch
from ppq.core import (USING_CUDA_KERNEL, ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates,
                      TensorQuantizationConfig)
from ppq.quantization.qfunction import BaseQuantFunction
from ppq.utils.round import ppq_tensor_round
from torch.autograd import Function

if USING_CUDA_KERNEL: from ppq.core import CUDA


class LinearQuantFunction(BaseQuantFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input_tensor: Any, quantization_config: TensorQuantizationConfig, **kwargs) -> Any:
        return super().__call__(input_tensor, quantization_config, **kwargs)


def torch_tensorwise_quantize(tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """
    Torch Tensorwise quantize is designed to quantize a torch Tensor with a given configuration.
        All quantization within PPQ will invoke this function to quantize its value.
        Any modification of this function will greatly affects system behaviour.
    
    This is a torch implemention of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True, 
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    
    Args:
        tensor (torch.Tensor): tensor to be quantized.
        config (TensorQuantizationConfig): quantization configuration.

    Raises:
        ValueError: [description]

    Returns:
        torch.Tensor: quantized tensor.
    """
    if not config.policy.has_property(QuantizationProperty.PER_TENSOR):
        raise ValueError('You are invoking torch tensor-wise quantize function, '
                            'however the input quantization config is not a per-tensor config. '
                            'Check your configuration twice, it is not allowed to quantize a '
                            'channel-wise value through this function.')
    
    tensor = ppq_tensor_round((tensor / config.scale), config.rounding) + config.offset
    tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    tensor = (tensor - config.offset) * config.scale
    return tensor


def torch_channelwise_quantize(tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """
    Torch Channelwise quantize is designed to quantize a torch Tensor with a given configuration.
        All quantization within PPQ will invoke this function to quantize its value.
        Any modification of this function will greatly affects system behaviour.
    
    This is a torch implemention of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True, 
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    
    Args:
        tensor (torch.Tensor): tensor to be quantized.
        config (TensorQuantizationConfig): quantization configuration.

    Raises:
        ValueError: [description]

    Returns:
        torch.Tensor: quantized tensor.
    """
    if (not config.policy.has_property(QuantizationProperty.PER_CHANNEL) or 
        not isinstance(config, ChannelwiseTensorQuantizationConfig)):
        raise ValueError('You are invoking torch channel-wise quantize function, '
                            'however the input quantization config is not a per-channel config. '
                            'Check your configuration twice, it is not allowed to quantize a '
                            'tensor-wise value through this function.')
    
    # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
    shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
    scale, offset = config.scale.view(shape), config.offset.view(shape)
    
    tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
    tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    tensor = (tensor - offset) * scale
    return tensor


if not USING_CUDA_KERNEL:
    class LinearQuantImpl(Function):
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, config: TensorQuantizationConfig) -> Any:
            if not QuantizationStates.is_activated(config.state): return tensor
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                return torch_channelwise_quantize(tensor=tensor, config=config)

            elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
                return torch_tensorwise_quantize(tensor=tensor, config=config)

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None


else: # if USING_CUDA_KERNEL:
    class LinearQuantImpl(Function):
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, config: TensorQuantizationConfig) -> Any:
            if not QuantizationStates.is_activated(config.state): return tensor

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                assert isinstance(config, ChannelwiseTensorQuantizationConfig)
                return CUDA.ChannelwiseLinearQuantize(
                    tensor=tensor,
                    scales=config.scale,
                    offsets=config.offset,
                    channel_axis=config.channel_axis,
                    minimum=config.quant_min,
                    maximum=config.quant_max,
                    rounding=config.rounding.value
                )

            elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
                assert isinstance(config, TensorQuantizationConfig)
                return CUDA.TensorwiseLinearQuantize(
                    tensor=tensor,
                    scale=config.scale,
                    offset=config.offset,
                    minimum=config.quant_min,
                    maximum=config.quant_max,
                    rounding=config.rounding.value
                )
            else:
                raise ValueError(
                    'You are using Linear quantization function, make sure your quantization config'
                    ' either have property "PER_CHANNEL" or property "PER_TENSOR"')

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None


TorchLinearQuantFunction = LinearQuantImpl().apply
