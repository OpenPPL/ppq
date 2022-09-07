from typing import Union

import torch
from ppq.api.setting import *
from ppq.core import (OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim import QuantizationOptimizationPipeline

from .base import BaseQuantizer


class FPGAQuantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph,
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - 128
        self._quant_max = + 127

    def build_quant_pipeline(self, setting: QuantizationSetting) -> QuantizationOptimizationPipeline:
        pipeline = super().build_quant_pipeline(setting)
        return pipeline

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR +
                    QuantizationProperty.POWER_OF_2)

                # Xilinx FPGA bias 并不是 32 位的！
                bias_config.num_of_bits = 30
                bias_config.quant_max = + int(pow(2, 29))
                bias_config.quant_min = - int(pow(2, 29))
                bias_config.state = QuantizationStates.PASSIVE_INIT
                base_quant_config.input_quantization_config[-1].observer_algorithm = 'Minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False

        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.FPGA_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'ConvTranspose', 'Gemm', 'Relu', 'PRelu',
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
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        """
        我不知道这个对不对, Xilinx FPGA 图融合做的并不少，
        如果你的网络中有特殊的激活函数，我建议你手动调整这个融合选项

        Returns:
            set: _description_
        """
        return {'Relu', 'Clip'}
