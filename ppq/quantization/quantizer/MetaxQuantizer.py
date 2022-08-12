from typing import Union

import torch
from ppq.api.setting import QuantizationSetting
from ppq.core import (PASSIVE_OPERATIONS, ChannelwiseTensorQuantizationConfig,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.executor.base import BaseGraphExecutor
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim.base import QuantizationOptimizationPipeline
from ppq.quantization.optim.morph import MetaxGemmSplitPass

from .base import BaseQuantizer


class MetaxTensorwiseQuantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = 0
        self._quant_max = 255

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        base_quant_config = self.create_default_quant_config(
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )

        if operation.type in {'Conv'}:
            # if operation has bias
            if operation.num_of_input == 3:
                bias_config = base_quant_config.input_quantization_config[-1]
                # bias should be quantized with 32 bits
                # in python3, int indicates long long in C++
                # so that it has enough precision to represent a number like 2^32
                # however, it may cause a scale underflow
                # here we give bias a 30 bits precision, which is pettery enough in all cases
                bias_config.num_of_bits = 30
                bias_config.quant_max = int(pow(2, 30 - 1))
                bias_config.quant_min = - int(pow(2, 30 - 1))
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR)
                bias_config.state = QuantizationStates.FP32
            for config in base_quant_config.input_quantization_config[1: ]:
                config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.METAX_INT8_T

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Relu', 'PRelu', 'Clip', 'Gemm',
            'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'LeakyRelu', 'Split', 'Concat',
            'Transpose', 'Slice', 'Reshape', 'Flatten',
            'MatMul'}

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR)

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    def build_prequant_pipeline(
        self, setting: QuantizationSetting, executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        return super().build_prequant_pipeline(setting, executor).append_optimization_to_pipeline(
            MetaxGemmSplitPass(), at_front=True)

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}


class MetaxChannelwiseQuantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, BaseGraph]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = -127
        self._quant_max = 127

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'MatMul'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.quant_max = 127
                conv_weight_config.quant_min = -127
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                base_quant_config.input_quantization_config[1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = conv_weight_config,
                        offset = None, scale  = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'MatMul'}:
                matmul_weight_config = base_quant_config.input_quantization_config[1]
                matmul_weight_config.quant_max = 127
                matmul_weight_config.quant_min = -127
                matmul_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                base_quant_config.input_quantization_config[1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = matmul_weight_config,
                        offset = None, scale  = None, channel_axis = 1
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # if operation has bias
            if operation.type == 'Conv' and operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                bias_config.num_of_bits = 30
                bias_config.quant_max = int(pow(2, bias_config.num_of_bits - 1))
                bias_config.quant_min = - int(pow(2, bias_config.num_of_bits - 1))
                bias_config.state = QuantizationStates.PASSIVE_INIT
                base_quant_config.input_quantization_config[-1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = bias_config, offset = None,
                        scale = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[-1].observer_algorithm = 'Minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.METAX_INT8_C

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Relu', 'PRelu', 'Clip', 'Gemm',
            'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool',
            'Mul', 'Add', 'LeakyRelu', 'Split', 'Concat',
            'Transpose', 'Slice', 'Reshape', 'Flatten',
            'MatMul'}

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

    def build_prequant_pipeline(
        self, setting: QuantizationSetting, executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        return super().build_prequant_pipeline(setting, executor).append_optimization_to_pipeline(
            MetaxGemmSplitPass(), at_front=True)

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}
