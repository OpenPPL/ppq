from typing import Any

import torch
from ppq.core import (USING_CUDA_KERNEL, ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
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


if not USING_CUDA_KERNEL:
    class TensorwiseLinearQuantImpl(Function):
        """
        Torch Tensorwise quantize is designed to quantize a torch Tensor with a given configuration.
            All quantization within PPQ will invoke this function to quantize its value.
            Any modification of this function will greatly affects system behaviour.
        
        This is a torch implemention of quantization itself.
        Notice that if ppq.config.USING_CUDA_KERNAL = True, 
            then all quantization will bypass this function by using ffi.CUDA instead.

        Notice this function will always clone your tensor value first.
        This function never quantize your tensor value inplace.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy, dropout: float = 0) -> torch.Tensor:
    
            tensor = ppq_tensor_round((tensor / scales), rounding) + offsets
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offsets) * scales
            return tensor

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None, None, None, None, None, None, None
    
    class ChannelwiseLinearQuantImpl(Function):
        """
        Torch Channelwise quantize is designed to quantize a torch Tensor with a given configuration.
            All quantization within PPQ will invoke this function to quantize its value.
            Any modification of this function will greatly affects system behaviour.
        
        This is a torch implemention of quantization itself.
        Notice that if ppq.config.USING_CUDA_KERNAL = True, 
            then all quantization will bypass this function by using ffi.CUDA instead.

        Notice this function will always clone your tensor value first.
        This function never quantize your tensor value inplace.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, channel_axis: int, 
                    quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy, dropout: float = 0) -> torch.Tensor:
            # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
            shape = [1 if axis != channel_axis else -1 for axis in range(tensor.ndim)]
            scale, offset = scales.view(shape), offsets.view(shape)
            
            tensor = ppq_tensor_round((tensor / scale), rounding) + offset
            tensor = torch.clamp(tensor, quant_min, quant_max)
            tensor = (tensor - offset) * scale
            return tensor

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            return dy, None, None, None, None, None, None, None, None


else: # if USING_CUDA_KERNEL:
    class TensorwiseLinearQuantImpl(Function):
        """
        Torch Tensorwise quantize is designed to quantize a torch Tensor with a given configuration.
            All quantization within PPQ will invoke this function to quantize its value.
            Any modification of this function will greatly affects system behaviour.
        
        This is a torch implemention of quantization itself.
        Notice that if ppq.config.USING_CUDA_KERNAL = True, 
            then all quantization will bypass this function by using ffi.CUDA instead.

        Notice this function will always clone your tensor value first.
        This function never quantize your tensor value inplace.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy, dropout: float) -> torch.Tensor:
            quantized = CUDA.LinearQuantize_T(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value,
                dropout=dropout
            )
            ctx.save_for_backward(tensor, quantized, scales, offsets)
            ctx._quant_params = [quant_min, quant_max]
            return quantized

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            tensor, quantized, scales, offsets = ctx.saved_tensors
            quant_min, quant_max = ctx._quant_params
            dx, ds, do = CUDA.LinearQuantize_T_B(tensor, quantized, scales, offsets, dy, quant_min, quant_max)
            return dx, ds, do, None, None, None, None, None
    
    class ChannelwiseLinearQuantImpl(Function):
        """
        Torch Channelwise quantize is designed to quantize a torch Tensor with a given configuration.
            All quantization within PPQ will invoke this function to quantize its value.
            Any modification of this function will greatly affects system behaviour.
        
        This is a torch implemention of quantization itself.
        Notice that if ppq.config.USING_CUDA_KERNAL = True, 
            then all quantization will bypass this function by using ffi.CUDA instead.

        Notice this function will always clone your tensor value first.
        This function never quantize your tensor value inplace.
        """
        @ staticmethod
        def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, 
                    offsets: torch.Tensor, channel_axis: int, 
                    quant_min: int, quant_max: int, 
                    rounding: RoundingPolicy, dropout: float) -> torch.Tensor:
            quantized = CUDA.LinearQuantize_C(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                channel_axis=channel_axis,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value,
                dropout=dropout
            )
            ctx.save_for_backward(tensor, quantized, scales, offsets)
            ctx._quant_params = [quant_min, quant_max, channel_axis]
            return quantized

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            tensor, quantized, scales, offsets = ctx.saved_tensors
            quant_min, quant_max, channel_axis = ctx._quant_params
            dx, ds, do = CUDA.LinearQuantize_C_B(tensor, quantized, scales, offsets, dy, quant_min, quant_max, channel_axis)
            return dx, ds, do, None, None, None, None, None


def PPQLinearQuantFunction(
    tensor: torch.Tensor, config: TensorQuantizationConfig, 
    dropout: float = 0.0) -> torch.Tensor:
    if not QuantizationStates.is_activated(config.state): return tensor
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        assert isinstance(config, ChannelwiseTensorQuantizationConfig), (
            'Critical Quantization Error! Except a ChannelwiseTensorQuantizationConfig.')
        return ChannelwiseLinearQuantImpl.apply(
            tensor, config.scale, config.offset, config.channel_axis,
            config.quant_min, config.quant_max, config.rounding,
            dropout)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseLinearQuantImpl.apply(
            tensor, config.scale, config.offset,
            config.quant_min, config.quant_max, config.rounding,
            dropout)


class PPQLinearQuantize(torch.nn.Module):
    def __init__(self, config: TensorQuantizationConfig):
        self.config = config
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        return PPQLinearQuantFunction(x, self.config)