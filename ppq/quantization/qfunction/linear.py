import torch
from ppq.core import (PPQ_CONFIG, QuantizationProperty, QuantizationStates,
                      RoundingPolicy, TensorQuantizationConfig)
from ppq.utils.round import ppq_tensor_round
from torch.autograd import Function


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
        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)

        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
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
        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)

        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
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


class TensorwiseDynamicLinearQuantImpl(Function):
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
    def forward(ctx, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
        from ppq.quantization.observer.range import minmax_to_scale_offset
        # solve scale and offset at first.
        scales, offsets = minmax_to_scale_offset(
            tensor.min().item(), tensor.max().item(), config=config)
        print(scales, offsets)
        # quantization function, pytorch implmentation
        tensor = ppq_tensor_round((tensor / scales), config.rounding) + offsets
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        tensor = (tensor - offsets) * scales
        return tensor

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None


class ChannelwiseDynamicLinearQuantImpl(Function):
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
    def forward(ctx, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
        from ppq.quantization.observer.range import minmax_to_scale_offset
        
        channelwise_view = tensor.transpose(dim0=0, dim1=config.channel_axis).unsqueeze(-1)
        channelwise_view = torch.flatten(channelwise_view, start_dim=1)
        
        scales, offsets = [], []
        for _min, _max in zip(
            channelwise_view.min(dim=1)[0].tolist(),
            channelwise_view.max(dim=1)[0].tolist()
        ):
            s, o = minmax_to_scale_offset(_min, _max, config)
            scales.append(s)
            offsets.append(o)

        scales = torch.tensor(scales, dtype=torch.float32, device=tensor.device)
        offsets = torch.tensor(offsets, dtype=torch.float32, device=tensor.device)

        # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
        shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
        scales, offsets = scales.view(shape), offsets.view(shape)

        tensor = ppq_tensor_round((tensor / scales), config.rounding) + offsets
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
        tensor = (tensor - offsets) * scales
        return tensor

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None

def PPQDyamicLinearQuantFunction(tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """
    Dynamic Linear Quantization Function(PPQ 动态量化函数).
    
    When calling this method, we firstly solve a scale & offset setting by min-max observer.
    
    Then we applys ordinary Linear Quantization Function with solved setting.
    
    If there is a pre-defined scale & offset within given config, they will be dropped without warning.
    
    动态量化函数将在执行量化之前统计出 tensor 的 min - max, 而后计算出 scale & offset 并完成量化
    
    此时 TQC 中的 scale 与 offset 将被忽略
    """
    if not QuantizationStates.is_activated(config.state): return tensor
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if not config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Quantization Policy Do Not Have Dynamic Flag!')

    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseDynamicLinearQuantImpl.apply(tensor, config)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseDynamicLinearQuantImpl.apply(tensor, config)

def PPQLinearQuantFunction(
    tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """PPQ 核心量化函数，没啥好说的了吧，这个玩意既做 quant 也做 dequant"""
    if not QuantizationStates.is_activated(config.state): return tensor
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Unexpected Dynamic Flag in Quantization Policy. Use PPQDyamicQuantFunction Instead.')

    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseLinearQuantImpl.apply(
            tensor, config.scale, config.offset, config.channel_axis,
            config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseLinearQuantImpl.apply(
            tensor, config.scale, config.offset,
            config.quant_min, config.quant_max, config.rounding)

def PPQLinearQuant_toInt(tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """PPQ 核心量化函数，没啥好说的了吧，这个玩意只做 quant 不做 dequant"""
    if not config.policy.has_property(QuantizationProperty.LINEAR):
        raise ValueError('Critical Quantization Error! Non-linear config detected.')
    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
        scale, offset = config.scale.view(shape), config.offset.view(shape)
        tensor = ppq_tensor_round((tensor / scale), config.rounding) + offset
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        tensor = ppq_tensor_round((tensor / config.scale), config.rounding) + config.offset
        tensor = torch.clamp(tensor, config.quant_min, config.quant_max)

    if config.num_of_bits == 8:
        if config.policy.has_property(QuantizationProperty.SYMMETRICAL):
            return tensor.type(dtype=torch.int8)
        if config.policy.has_property(QuantizationProperty.ASYMMETRICAL):
            return tensor.type(dtype=torch.uint8)
    elif config.num_of_bits > 8:
        return tensor.type(dtype=torch.int32)
    else: raise Exception('Do not konw how to convert value into int. num of bits is unexpected.')
