import torch
from ppq.core import QuantizationProperty, TensorQuantizationConfig

from .base import BaseQuantFunction
from .floating import PPQFloatingQuantFunction
from .linear import (PPQDyamicLinearQuantFunction, PPQLinearQuant_toInt,
                     PPQLinearQuantFunction)


def PPQuantFunction(tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """
    ## PPQ 核心量化函数

    根据 config 中描述的策略，量化给定的 tensor.
    
    请注意 config.state 必须处在激活状态，该函数起作用。如果 config.state 处于
        INITIAL, FP32, PASSIVE_INIT 等未激活状态，该函数不进行任何处理，直接返回 tensor.

    ### 线性量化(QuantizationProperty.LINEAR):

        INT8 = Clip(Round((FP32 / scale)))

    ### 浮点量化(QuantizationProperty.FLOATING):

        FP8 = Clip(FP32_TO_FP8((FP32 / scale)))

    ### 动态线性量化(QuantizationProperty.DYNMAIC)

        scale = max(FP32) / 255

        INT8  = Clip(Round((FP32 / scale)))

    """
    if tensor is None: raise ValueError('Tensor is empty.')
    if config.policy.has_property(QuantizationProperty.LINEAR):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return PPQLinearQuantFunction(tensor, config)
        else: return PPQDyamicLinearQuantFunction(tensor, config)

    if config.policy.has_property(QuantizationProperty.FLOATING):
        return PPQFloatingQuantFunction(tensor, config)

    raise ValueError('Unexpected Quantization Property Found in PPQuantFunction. '
                     'Do not konw how to quantize your config yet.')


def PPQuantFunction_toInt(tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    """
    ## PPQ 核心量化函数

    根据 config 中描述的策略，这个函数将会执行线性量化，动态量化
    
    但是结果直接出来是整数
    """

    if config.policy.has_property(QuantizationProperty.LINEAR):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return PPQLinearQuant_toInt(tensor, config)

    raise ValueError('Unexpected Quantization Property Found in PPQuantFunction_toInt. '
                     'Do not konw how to quantize your config yet.')


__all__ = ['PPQuantFunction', 'PPQuantFunction_toInt', 'PPQDyamicLinearQuantFunction',
           'PPQLinearQuantFunction', 'PPQFloatingQuantFunction', 'BaseQuantFunction', 
           'PPQLinearQuant_toInt']
