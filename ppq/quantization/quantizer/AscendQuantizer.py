from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation
from .base import BaseQuantizer

from ppq.executor.op.torch.base import GET_ATTRIBUTE_FROM_OPERATION

def ASSERT_CONV_AND_DECONV(op: Operation):
    if op.type == "Conv":
        filter = op.inputs[1].value.shape
        assert len(filter) == 4, (
            f'Ascend Quantization needs the dimension of filter must be 4, while your filter is {len(filter)}')


    if op.type == "ConvTranspose":
        filter = op.inputs[1].value.shape
        dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', compulsive=True, default=1)
        groups    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', compulsive=True, default=1)

        assert len(filter) == 4, (
            f'Ascend Quantization needs the dimension of filter must be 4, while your filter is {len(filter)}')
        assert dilation == [1, 1], (
            f'Ascend Quantization needs the dilation must be [1, 1], while your dilation is {dilation}')
        assert groups == 1, (
            f'Ascend Quantization needs the groups must be 1, while your groups is {groups}')



def ASSERT_GEMM(op: Operation):
    axis  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', compulsive=True, default=1)
    assert axis == 1, (
            f'Ascend Quantization needs the axis must be 1, while your axis is {axis}')

    if 'transpose' in op.attributes:
        transpose = op.attributes['transpose']
        assert transpose == False, (
            f'Ascend Quantization needs the transpose must be False, while your transpose is {transpose}')



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
            quant_max= self._quant_max, quant_min= self._quant_min, observer_algorithm='percentile', 
            policy=self.quantize_policy, rounding=self.rounding_policy,
        )

        ASSERT_CONV_AND_DECONV(operation)

        if operation.type in {'Conv', 'Gemm', 'ConvTranspose'}:

            assert operation.num_of_input > 0, 'Seems you got a Computing layer with no parameters.'

            if operation.type in {'Conv', 'ConvTranspose'}:

                if operation.inputs[1].is_parameter:
                    conv_weight_config = base_quant_config.input_quantization_config[1]
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                    conv_weight_config.observer_algorithm = 'minmax'
                    conv_weight_config.quant_max = 127
                    conv_weight_config.quant_min = -128

            elif operation.type == 'Gemm':
                
                ASSERT_GEMM(operation)

                if operation.inputs[1].is_parameter:
                    gemm_weight_config = base_quant_config.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_TENSOR
                    )
                    
                    # gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'
                    gemm_weight_config.quant_max = 127
                    gemm_weight_config.quant_min = -128

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

