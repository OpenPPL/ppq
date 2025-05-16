import torch

from mppq.ir.base.quantize import BaseQuantFunction
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)
from mppq.utils.qfunction.floating import floating_quant, floating_quant_tofloat
from mppq.utils.qfunction.linear import (
    dynamic_linear_quant,
    linear_fake_quant,
    linear_quant_toint,
)


def ppq_fake_quant(
    tensor: torch.Tensor, config: TensorQuantizationConfig
) -> torch.Tensor:
    """
    ## PPQ 核心量化函数

    根据 config 中描述的策略，量化给定的 tensor.

    请注意 config.state 必须处在激活状态，该函数起作用。如果 config.state 处于
        INITIAL, FP32, PASSIVE_INIT 等未激活状态，该函数不进行任何处理，直接返回 tensor.

    ### 线性量化(QuantizationProperty.LINEAR):

        INT8 = Clip(Round((FP32 / scale)))

    ### 浮点量化(QuantizationProperty.FLOATING):

        FP8 = Clip(FP32_TO_FP8((FP32 / scale)))

    ### 动态线性量化(QuantizationProperty.DYNAMIC)

        scale = max(FP32) / 255

        INT8  = Clip(Round((FP32 / scale)))

    """
    if tensor is None:
        raise ValueError("Tensor is empty.")
    if not QuantizationStates.is_activated(config.state):
        return tensor
    if config.policy.has_property(QuantizationProperty.LINEAR):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return linear_fake_quant(tensor, config)
        else:
            return dynamic_linear_quant(tensor, config)

    if config.policy.has_property(QuantizationProperty.FLOATING):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return floating_quant(tensor, config)

    raise ValueError(
        "Unexpected Quantization Property Found in ppq_fake_quant. "
        "Do not know how to quantize your config yet."
    )


def ppq_quant_toint(
    tensor: torch.Tensor, config: TensorQuantizationConfig
) -> torch.Tensor:
    """
    ## PPQ 核心量化函数

    根据 config 中描述的策略，这个函数将会执行线性量化，动态量化

    但是结果直接出来是整数
    """
    if config.policy.has_property(QuantizationProperty.LINEAR):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return linear_quant_toint(tensor, config)

    raise ValueError(
        "Unexpected Quantization Property Found in ppq_quant_toint. "
        "Do not know how to quantize your config yet."
    )
    
def ppq_quant_tofloat(
    tensor: torch.Tensor, config: TensorQuantizationConfig
) -> torch.Tensor:
    """
    ## PPQ 核心量化函数

    根据 config 中描述的策略，这个函数将会执行线性量化，动态量化

    但是结果直接出来是float8类型
    """
    if config.policy.has_property(QuantizationProperty.FLOATING):
        if not config.policy.has_property(QuantizationProperty.DYNAMIC):
            return floating_quant_tofloat(tensor, config)

    raise ValueError(
        "Unexpected Quantization Property Found in ppq_quant_tofp8. "
        "Do not know how to quantize your config yet."
    )


__all__ = [
    "ppq_fake_quant",
    "ppq_quant_toint",
    "ppq_quant_tofloat",
    "BaseQuantFunction",
]
