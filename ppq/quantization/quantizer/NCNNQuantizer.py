from typing import Union

import torch
from ppq.api.setting import QuantizationSetting
from ppq.core import (ChannelwiseTensorQuantizationConfig,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, Operation
from ppq.quantization.optim import (NCNNFormatGemmPass,
                                    QuantizationOptimizationPipeline)

from .base import BaseQuantizer


class NCNNQuantizer(BaseQuantizer):
    def __init__(
        self, graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - 127
        self._quant_max = + 127

    def build_prequant_pipeline(
        self, setting: QuantizationSetting, executor: BaseGraphExecutor) -> QuantizationOptimizationPipeline:
        pipeline = super().build_prequant_pipeline(setting, executor)
        pipeline.append_optimization_to_pipeline(NCNNFormatGemmPass(), at_front=True)
        return pipeline

    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'Gemm'}:
            assert operation.num_of_input > 0, 'Seems you got a Computing layer with no parameters.'

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

                group        = operation.attributes.get('group', 1)
                dilations    = operation.attributes.get('dilations', [1, 1])
                strides      = operation.attributes.get('strides', [1, 1])
                kernel_shape = operation.attributes.get('kernel_shape')
                if group == 1 and all([i == 1 for i in dilations]) and all([j == 1 for j in strides])\
                    and all([k == 3 for k in kernel_shape]):
                    base_quant_config.input_quantization_config[1].num_of_bits = 6
                    base_quant_config.input_quantization_config[1].quant_max   = +31
                    base_quant_config.input_quantization_config[1].quant_min   = -31

            elif operation.type == 'Gemm':
                assert operation.attributes.get('transB', 0) and operation.attributes.get('alpha', 1.0) == 1.0 \
                    and operation.attributes.get('beta', 1.0) == 1.0
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
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

            base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.NCNN_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Gemm'
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
        return RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO

    @ property
    def activation_fusion_types(self) -> set:
        """
        ncnn 只需要输入定点 不需要考虑激活函数融合

        Returns:
            set: _description_
        """
        return {}
