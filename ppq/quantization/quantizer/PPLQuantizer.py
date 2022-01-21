from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, ChannelwiseTensorQuantizationConfig,
                      OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)
from ppq.IR import BaseGraph, GraphCommandProcesser
from ppq.IR.base.graph import Operation, Variable
from ppq.core.quant import TensorQuantizationConfig

from .base import BaseQuantizer


class PPLCUDAQuantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcesser]
    ) -> Union[torch.Tensor, list, dict]:

        self._num_of_bits = 8
        self._quant_min = - int(pow(2, self._num_of_bits - 1))
        self._quant_max = int(pow(2, self._num_of_bits - 1) - 1)

        super().__init__(graph=graph)

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
            if operation.type in {'Conv', 'ConvTranspose'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                base_quant_config.input_quantization_config[1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = conv_weight_config,
                        offsets = None, scales  = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm'}:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                base_quant_config.input_quantization_config[1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = gemm_weight_config,
                        offsets = None, scales  = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'
            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                bias_config.num_of_bits = 32
                bias_config.quant_max = int(pow(2, bias_config.num_of_bits - 1)) - 1
                bias_config.quant_min = - int(pow(2, bias_config.num_of_bits - 1)) + 1
                bias_config.state = QuantizationStates.PASSIVE_INIT
                base_quant_config.input_quantization_config[-1] = \
                    ChannelwiseTensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = bias_config, offsets = None,
                        scales = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[-1].observer_algorithm = 'Minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
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
            'Transpose', 'Slice', 'Reshape', 'Flatten'
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


class PPLCUDAMixPrecisionQuantizer(PPLCUDAQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcesser]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        config = super().init_quantize_config(operation=operation)
        if operation.platform == TargetPlatform.PPL_CUDA_INT4:
            for cfg, var in zip(config.input_quantization_config, operation.inputs):
                assert isinstance(cfg, TensorQuantizationConfig)
                assert isinstance(var, Variable)
                if cfg.state == QuantizationStates.INITIAL:
                    cfg.num_of_bits, cfg.quant_max, cfg.quant_min = 4, 7, -8
        return config


class PPLCUDA_INT4_Quantizer(PPLCUDAQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcesser]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)

    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcesser]
    ) -> Union[torch.Tensor, list, dict]:

        super().__init__(graph=graph)
        self._num_of_bits = 4
        self._quant_min = - int(pow(2, self._num_of_bits - 1))
        self._quant_max = int(pow(2, self._num_of_bits - 1) - 1)

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_CUDA_INT8