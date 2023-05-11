from typing import Union

import torch
from ppq.api.setting import QuantizationSetting
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim import (NxpInputRoundingRefinePass,
                                    NXPResizeModeChangePass,
                                    QuantizationOptimizationPipeline)

from .base import BaseQuantizer


class NXP_Quantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph,
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - int(pow(2, self._num_of_bits - 1))
        self._quant_max = int(pow(2, self._num_of_bits - 1) - 1)

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        pipeline = super().build_quant_pipeline(setting)
        pipeline.append_optimization_to_pipeline(NXPResizeModeChangePass(), at_front=True)
        pipeline.append_optimization_to_pipeline(NxpInputRoundingRefinePass(), at_front=True)
        return pipeline

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')

        if operation.type in {'Conv', 'Gemm'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.POWER_OF_2
                )
                conv_weight_config.rounding = RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO
                conv_weight_config.channel_axis = 0
                conv_weight_config.observer_algorithm = 'minmax'
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm'}:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.POWER_OF_2
                )
                gemm_weight_config.rounding = RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO
                gemm_weight_config.channel_axis = 0
                gemm_weight_config.observer_algorithm = 'minmax'

            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.POWER_OF_2
                )
                bias_config.num_of_bits = 30
                bias_config.quant_max = int(pow(2, 30))
                bias_config.quant_min = - int(pow(2, 30))
                bias_config.state = QuantizationStates.PASSIVE_INIT
                bias_config.channel_axis = 0
                bias_config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False

        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.NXP_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'Max', 'Sub', 'Div',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Slice'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR +
            QuantizationProperty.POWER_OF_2
        )

    @ property
    def rounding_policy(self):
        return RoundingPolicy.ROUND_HALF_UP

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}
