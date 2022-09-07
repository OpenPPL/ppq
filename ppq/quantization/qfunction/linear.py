from typing import Any

import torch
from ppq.core import (PPQ_CONFIG, ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TensorQuantizationConfig)
from ppq.quantization.qfunction import BaseQuantFunction
from ppq.utils.round import ppq_tensor_round
from torch.autograd import Function


class LinearQuantFunction(BaseQuantFunction):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input_tensor: Any, quantization_config: TensorQuantizationConfig, **kwargs) -> Any:
        return super().__call__(input_tensor, quantization_config, **kwargs)


class TensorwiseLinearQuantImpl(Function):
    """Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor,
                offsets: torch.Tensor, quant_min: int, quant_max: int,
                rounding: RoundingPolicy) -> torch.Tensor:

        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            scales = scales.to(tensor.device)
            offsets = offsets.to(tensor.device)
            # quantization function, pytorch implmentation
            tensor = ppq_tensor_round((tensor / scales), rounding) + offsets
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offsets) * scales
            return tensor
        
        else:
            from ppq.core import CUDA

            # quantization function, pure cuda implmentation
            quantized = CUDA.LinearQuantize_T(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value
            )
            return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None


class ChannelwiseLinearQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor,
                offsets: torch.Tensor, channel_axis: int,
                quant_min: int, quant_max: int,
                rounding: RoundingPolicy) -> torch.Tensor:
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            scales = scales.to(tensor.device)
            offsets = offsets.to(tensor.device)
            # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
            shape = [1 if axis != channel_axis else -1 for axis in range(tensor.ndim)]
            scale, offset = scales.view(shape), offsets.view(shape)

            tensor = ppq_tensor_round((tensor / scale), rounding) + offset
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offset) * scale
            return tensor
        else:
            from ppq.core import CUDA
            quantized = CUDA.LinearQuantize_C(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                channel_axis=channel_axis,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value)
            return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None, None


def PPQLinearQuantFunction(
    tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """
    PPQ 核心量化函数

    Args:
        tensor (torch.Tensor): _description_
        config (TensorQuantizationConfig): _description_

    Raises:
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if not QuantizationStates.is_activated(config.state): return tensor
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        assert isinstance(config, ChannelwiseTensorQuantizationConfig), (
            'Critical Quantization Error! Except a ChannelwiseTensorQuantizationConfig.')
        return ChannelwiseLinearQuantImpl.apply(
            tensor, config.scale, config.offset, config.channel_axis,
            config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseLinearQuantImpl.apply(
            tensor, config.scale, config.offset,
            config.quant_min, config.quant_max, config.rounding)


def PPQLinearQuant_toInt(tensor: torch.Tensor, config: TensorQuantizationConfig, ) -> torch.Tensor:
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        assert isinstance(config, ChannelwiseTensorQuantizationConfig), (
            'Critical Quantization Error! Except a ChannelwiseTensorQuantizationConfig.')
        shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
        scale, offset = config.scale.view(shape), config.offset.view(shape)
        tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        tensor = ppq_tensor_round((tensor / config.scale), config.rounding) + config.offset
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    return tensor.type(dtype=torch.int32)


class PPQLinearQuantize(torch.nn.Module):
    def __init__(self, config: TensorQuantizationConfig):
        self.config = config
        super().__init__()

    def forward(self, x: torch.Tensor):
        return PPQLinearQuantFunction(x, self.config)
