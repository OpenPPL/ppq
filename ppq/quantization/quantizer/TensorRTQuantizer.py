from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, GraphCommandProcessor, Operation

from .base import BaseQuantizer


class TensorRTQuantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcessor]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - 128
        self._quant_max = + 127

    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        OQC = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
            # base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                if operation.inputs[1].is_parameter:
                    conv_weight_config = OQC.input_quantization_config[1]
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                    conv_weight_config.observer_algorithm = 'minmax'

            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = OQC.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'

            if operation.num_of_input > 2:
                bias_config = OQC.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type == 'LayerNormalization':
            # Layernorm - gamma and beta need to be FP32
            for TQC in OQC.input_quantization_config[1: ]:
                TQC.state = QuantizationStates.FP32

        if operation.type == 'Attention':
            # Attention - Only input and weight need to be quantized.
            for TQC in OQC.input_quantization_config[2: ]:
                TQC.state = QuantizationStates.FP32

        return OQC

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.TRT_INT8

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Softmax',
            'Mul', 'Add', 'Max', 'Sub', 'Div', 'Reshape',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Interp',
            'ReduceMean', 'Transpose', 'Slice', 'Flatten',
            'HardSwish', 'HardSigmoid', 'MatMul',
            'Attention', 'LayerNormalization', 'Gelu',
            'PPQBiasFusedMatMul'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Swish', 'Clip', 'SoftPlus', 'Sigmoid', 'Gelu'}


class TensorRTQuantizer_InputOnly(BaseQuantizer):
    """
    This is another version of TensorRT Int8 Quantizer.
        Only Quantize Conv, Gemm, AveragePooling.
        Only Quantize Input Of those layer.
    """
    def __init__(
        self, graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - 128
        self._quant_max = + 127

    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
            base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
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
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul', 'PPQBiasFusedMatMul'}:
                if operation.inputs[1].is_parameter:
                    gemm_weight_config = base_quant_config.input_quantization_config[1]
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    gemm_weight_config.channel_axis = 0
                    gemm_weight_config.observer_algorithm = 'minmax'
            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.TRT_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Gemm', 'ConvTranspose', 'MatMul',
            'AveragePool', 'GlobalAveragePool', 
            'PPQBiasFusedMatMul'}

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        # 这个量化器只对输入定点，不需要考虑激活函数融合
        return {'Relu', 'Clip'}
