from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation
from .base import BaseQuantizer


class AscendQuantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = 0
        self._quant_max = int(pow(2, self._num_of_bits) - 1)


    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        base_quant_config = self.create_default_quant_config(
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )

        if operation.type in {'Conv', 'Gemm', 'ConvTranspose'}:

            assert operation.num_of_input > 0, 'Seems you got a Computing layer with no parameters.'

            # if operation.type in {'Conv', 'ConvTranspose'}:
            if operation.type == "Conv":
                if operation.inputs[1].is_parameter:
                    conv_weight_config = base_quant_config.input_quantization_config[1]
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    # conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                    conv_weight_config.channel_axis = 1
                    conv_weight_config.observer_algorithm = 'minmax'
        

            elif operation.type == 'Gemm':
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = base_quant_config.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_TENSOR
                    )
                    
                    # gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'

            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.ASC_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {'Conv', 'ConvTranspose', 'Gemm', 'AveragePool'}

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}

