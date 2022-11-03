from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation

from .base import BaseQuantizer


class GraphCoreQuantizer(BaseQuantizer):
    def __init__(
        self, graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits    = 8
        self._exponent_bits  = 4
        self._quant_min      = - 448.0
        self._quant_max      = + 448.0

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=self._exponent_bits,
            quant_max=self._quant_max, quant_min=self._quant_min, observer_algorithm='floating')

        # 一些特殊的算子需要更复杂的量化逻辑，我们在下面的代码中对量化信息进行调整
        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:

            # 将 Conv 的参数设置为对称 per-channel 量化
            if operation.type in {'Conv', 'ConvTranspose'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.PER_CHANNEL + 
                    QuantizationProperty.FLOATING +
                    QuantizationProperty.POWER_OF_2
                )
                # 量化的 channel_axis 对于 conv 而言是 0，对于反卷积而言是 1
                # 参数量化使用 min max observe
                conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                conv_weight_config.observer_algorithm = 'floating'

            # 将 Gemm 的参数设置为对称 per-channel 量化
            elif operation.type in {'Gemm', 'MatMul'}:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.FLOATING +
                    QuantizationProperty.POWER_OF_2
                )
                gemm_weight_config.channel_axis = 0
                gemm_weight_config.observer_algorithm = 'floating'

            # 如果有 bias, 那 bias 就不量化了，因为我也不知道应该怎么量化他
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # 有一些算子是被动量化的，他们的输入输出将共享量化 scale
            base_quant_config.is_active_quant_op = False

        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_CUDA_INT8

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
            'Sigmoid', 'ReduceMean', 'ConvTranspose',
            'MatMul'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.PER_TENSOR +
            QuantizationProperty.FLOATING +
            QuantizationProperty.POWER_OF_2
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Sigmoid', 'LeakyRelu', 'Swish', 'Mish', 'PRelu'}



class TensorRTQuantizer_FP8(BaseQuantizer):
    def __init__(
        self, graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits    = 8
        self._exponent_bits = 4
        self._quant_min      = - 448.0
        self._quant_max      = + 448.0

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=self._exponent_bits,
            quant_max=self._quant_max, quant_min=self._quant_min, observer_algorithm='floating')

        # 一些特殊的算子需要更复杂的量化逻辑，我们在下面的代码中对量化信息进行调整
        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:

            # 将 Conv 的参数设置为对称 per-channel 量化
            if operation.type in {'Conv', 'ConvTranspose'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.PER_CHANNEL + 
                    QuantizationProperty.FLOATING +
                    QuantizationProperty.POWER_OF_2
                )
                # 量化的 channel_axis 对于 conv 而言是 0，对于反卷积而言是 1
                # 参数量化使用 min max observe
                conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)
                conv_weight_config.observer_algorithm = 'floating'

            # 将 Gemm 的参数设置为对称 per-channel 量化
            elif operation.type in {'Gemm', 'MatMul'}:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.PER_CHANNEL +
                    QuantizationProperty.FLOATING +
                    QuantizationProperty.POWER_OF_2
                )
                gemm_weight_config.channel_axis = 0
                gemm_weight_config.observer_algorithm = 'floating'

            # 如果有 bias, 那 bias 就不量化了，因为我也不知道应该怎么量化他
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32
                
            # 所有算子只量化输入
            for output_config in base_quant_config.output_quantization_config:
                output_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # 有一些算子是被动量化的，他们的输入输出将共享量化 scale
            base_quant_config.is_active_quant_op = False

        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_CUDA_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Gemm',
            'AveragePool',
            'GlobalAveragePool',
            'ConvTranspose',
            'MatMul'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.PER_TENSOR +
            QuantizationProperty.FLOATING +
            QuantizationProperty.POWER_OF_2
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Sigmoid', 'LeakyRelu', 'Swish', 'Mish', 'PRelu'}
