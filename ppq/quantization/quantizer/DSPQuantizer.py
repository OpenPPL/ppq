from typing import Union

import torch
from ppq.api.setting import *
from ppq.core import (PASSIVE_OPERATIONS, ChannelwiseTensorQuantizationConfig,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim import (PPLDSPTIReCalibrationPass,
                                    QuantizationOptimizationPipeline)

from .base import BaseQuantizer


class PPL_DSP_Quantizer(BaseQuantizer):
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
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            # if operation has bias
            if operation.num_of_input == 3:
                bias_config = base_quant_config.input_quantization_config[-1]
                # bias should be quantized with 32 bits
                # in python3, int indicates long long in C++
                # so that it has enough precision to represent a number like 2^32
                # however, it may cause a scale underflow
                # here we give bias a 30 bits precision, which is pettery enough in all cases
                bias_config.num_of_bits = 30
                bias_config.quant_max = int(pow(2, 30 - 1) - 1)
                bias_config.quant_min = - int(pow(2, 30 - 1))
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR)
                bias_config.state = QuantizationStates.PASSIVE_INIT
            for config in base_quant_config.input_quantization_config[1: ]:
                config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_DSP_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
            'Clip', 'Pad', 'Resize', 'MaxPool', 'AveragePool',
            'GlobalMaxPool', 'GlobalAveragePool', 'Softmax',
            'Mul', 'Add', 'Max', 'Sub', 'Div', 'Reshape',
            'LeakyRelu', 'Concat', 'Sigmoid', 'Slice', 'Interp',
            'ReduceMean', 'Flatten'}

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}


class PPL_DSP_TI_Quantizer(PPL_DSP_Quantizer):
    def __init__(self, graph: BaseGraph) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph)
        self._num_of_bits = 8
        self._quant_min   = -int(pow(2, self._num_of_bits - 1))
        self._quant_max   = int(pow(2, self._num_of_bits - 1)) - 1

    def build_quant_pipeline(
        self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        pipeline = super().build_quant_pipeline(setting)
        pipeline.append_optimization_to_pipeline(PPLDSPTIReCalibrationPass())
        return pipeline

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type == 'Conv':
                conv_weight_config = base_quant_config.input_quantization_config[1]
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

            elif operation.type == 'ConvTranspose':
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                base_quant_config.input_quantization_config[1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = conv_weight_config,
                        offset = None, scale  = None, channel_axis = 1
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type == 'Gemm':
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                base_quant_config.input_quantization_config[1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = gemm_weight_config,
                        offset = None, scale  = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # if operation has bias
            # let bias be fp32 because dsp ti don't need quantization parameter of bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_DSP_TI_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}
