import torch
from ppq.core import (PPQ_CONFIG, QuantizationProperty, QuantizationStates,
                      RoundingPolicy, TensorQuantizationConfig)
from torch.autograd import Function


class TensorwiseFloatingQuantImpl(Function):
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
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor,
                mantissa_bits: int, exponet_bits: int,
                quant_min: float, quant_max: float,
                rounding: RoundingPolicy) -> torch.Tensor:

        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            # quantization function, pytorch implmentation
            raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
        
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


class ChannelwiseFloatingQuantImpl(Function):
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

        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
            raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
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


def PPQFloatingQuantFunction(
    tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    if not PPQ_CONFIG.USING_CUDA_KERNEL:
        raise PermissionError('PPQ Floating Quant Function require PPQ_CONFIG.USING_CUDA_KERNEL = True')
    if not tensor.is_cuda():
        raise PermissionError('PPQ Floating Quant Function requires tensor device to be cuda, '
                              'CPU floating quantization is not implemented yet.')

    """PPQ 核心量化函数，没啥好说的了吧，这个玩意既做 quant 也做 dequant"""
    if not QuantizationStates.is_activated(config.state): return tensor
    if not config.policy.has_property(QuantizationProperty.FLOATING):
        raise ValueError('Critical Quantization Error! Unexpected policy detected. '
                         'PPQFloatingQuantFunction except a Floating Quantization Config.')
    if config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Unexpected Dynamic Flag in Quantization Policy.')

    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseFloatingQuantImpl.apply(
            tensor, config.scale, config.offset, config.channel_axis,
            config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseFloatingQuantImpl.apply(
            tensor, config.scale, config.offset,
            config.quant_min, config.quant_max, config.rounding)
