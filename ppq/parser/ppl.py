import json

from ppq.core import (DataType, QuantizationProperty, QuantizationStates,
                      TargetPlatform, TensorQuantizationConfig)
from ppq.IR import BaseGraph
from ppq.IR.quantize import QuantableOperation

from .onnx_exporter import OnnxExporter
from .util import convert_value


def convert_type(platform: TargetPlatform) -> str:
    if platform == TargetPlatform.PPL_CUDA_INT8: return 'INT8'
    if platform == TargetPlatform.SHAPE_OR_INDEX: return None
    if platform == TargetPlatform.FP32: return None
    raise TypeError(f'Unsupported platform type. ({str(platform)})')


class PPLBackendExporter(OnnxExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        var_quant_info_recorder, op_platform_recorder = {}, {}
        for operation in graph.operations.values():
            if not isinstance(operation, QuantableOperation): continue
            for config, var in operation.config_with_variable:
                if not config.can_export(): continue

                # PATCH 2021.11.25
                # REMOVE BIAS FROM CONFIGURATION
                if config.num_of_bits > 8: continue

                if config.state in {
                    QuantizationStates.SOI,
                    QuantizationStates.DEACTIVATED,
                    QuantizationStates.DEQUANTIZED,
                    QuantizationStates.FP32
                }: continue
                # Simply override recorder is acceptable here,
                # we do not support mix precision quantization for CUDA backend now.
                # All configurations for this variable should keep identical towards each other.

                if config.state == QuantizationStates.SLAVE and var.name in var_quant_info_recorder: continue
                var_quant_info_recorder[var.name] = config

        # ready to render config to json.
        for var in var_quant_info_recorder:
            config = var_quant_info_recorder[var]
            assert isinstance(config, TensorQuantizationConfig)
            tensorwise = config.policy.has_property(QuantizationProperty.PER_TENSOR)
            var_quant_info_recorder[var] = {
                'bit_width'  : config.num_of_bits,
                'per_channel': config.policy.has_property(QuantizationProperty.PER_CHANNEL),
                'quant_flag' : True,
                'sym'        : config.policy.has_property(QuantizationProperty.SYMMETRICAL),
                'scale'      : convert_value(config.scale, tensorwise, DataType.FP32),
                'zero_point' : convert_value(config.offset, tensorwise, DataType.INT32),
                'tensor_min' : convert_value(config.scale * (config.quant_min - config.offset), tensorwise, DataType.FP32),
                'tensor_max' : convert_value(config.scale * (config.quant_max - config.offset), tensorwise, DataType.FP32),
                'q_min'      : config.quant_min,
                'q_max'      : config.quant_max,
                'hash'       : config.__hash__(),
                'dominator'  : config.dominated_by.__hash__()
            }

        for op in graph.operations.values():
            if convert_type(op.platform) is not None:
                op_platform_recorder[op.name] = {
                    'data_type': convert_type(op.platform)
                }

        exports = {
            'quant_info': var_quant_info_recorder,
            'op_info': op_platform_recorder}

        with open(file=config_path, mode='w') as file:
            json.dump(exports, file, indent=4)
