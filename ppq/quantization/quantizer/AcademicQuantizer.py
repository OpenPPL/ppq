from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform,
                      TensorQuantizationConfig)
from ppq.IR import BaseGraph, Operation, Variable

from .base import BaseQuantizer


class ACADEMICQuantizer(BaseQuantizer):
    """ACADEMICQuantizer applies a loose quantization scheme where only input
    variables of computing ops are quantized(symmetrical per-tensor for weight
    and asymmetrical per-tensor for activation).

    This setting doesn't align with any kind of backend for now and it's
    designed only for purpose of paper reproducing and algorithm verification.
    """
    def __init__(self, graph: BaseGraph, verbose: bool = True) -> None:
        super().__init__(graph, verbose)
        self._num_of_bits = 8
        self._quant_min = 0
        self._quant_max = 255

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:

        # create a basic quantization configuration.
        config = self.create_default_quant_config(
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile', policy=self.quantize_policy,
            rounding=self.rounding_policy,
        )

        # actually usually only support quantization of inputs of computing
        # ops in academic settings
        if operation.type in {'Conv', 'Gemm', 'ConvTranspose'}:

            W_config = config.input_quantization_config[1]
            output_config = config.output_quantization_config[0]

            W_config.quant_max = int(pow(2, self._num_of_bits - 1) - 1)
            W_config.quant_min = - int(pow(2, self._num_of_bits - 1))
            W_config.policy = QuantizationPolicy(
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.LINEAR +
                QuantizationProperty.PER_TENSOR
            )
            output_config.state = QuantizationStates.FP32


            if operation.num_of_input == 3:
                bias_config = config.input_quantization_config[-1]
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

            for tensor_config in config.input_quantization_config[1: ]:
                tensor_config.observer_algorithm = 'minmax'

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            config.is_active_quant_op = False
        return config


    @ property
    def target_platform(self) -> TargetPlatform:

        return TargetPlatform.ACADEMIC_INT8


    @ property
    def default_platform(self) -> TargetPlatform:

        return TargetPlatform.FP32


    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'Gemm', 'ConvTranspose'}


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
        return {}


class ACADEMIC_INT4_Quantizer(ACADEMICQuantizer):
    def __init__(self, graph: BaseGraph, verbose: bool = True) -> None:
        super().__init__(graph, verbose)
        self._num_of_bits = 4
        self._quant_min = 0
        self._quant_max = 15

    @ property
    def target_platform(self) -> TargetPlatform:

        return TargetPlatform.ACADEMIC_INT4


class ACADEMIC_Mix_Quantizer(ACADEMICQuantizer):
    def __init__(self, graph: BaseGraph, verbose: bool = True) -> None:
        super().__init__(graph, verbose)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        config = super().init_quantize_config(operation=operation)
        if operation.platform == TargetPlatform.ACADEMIC_INT4:
            for idx, (cfg, var) in enumerate(zip(config.input_quantization_config, operation.inputs)):
                assert isinstance(cfg, TensorQuantizationConfig)
                assert isinstance(var, Variable)
                if cfg.state == QuantizationStates.INITIAL:
                    cfg.num_of_bits = 4
                    if idx == 0:
                        cfg.quant_max, cfg.quant_min = 15, 0
                    else:
                        cfg.quant_max, cfg.quant_min = 7, -8
        return config
