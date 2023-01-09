import json
from typing import List

from ppq.core import (DataType, QuantizationStates,
                      QuantizationVisibility, NetworkFramework)
from ppq.IR import BaseGraph, GraphExporter
from ppq.IR.quantize import QuantableOperation

from .onnx_exporter import OnnxExporter
from .caffe_exporter import CaffeExporter
from .util import convert_value


class QNNDSPExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        activation_info, param_info = {}, {}
        topo_order =  graph.topological_sort()
        for operation in topo_order:
            if not isinstance(operation, QuantableOperation): continue
            for config, var in operation.config_with_variable:
                if not QuantizationStates.can_export(config.state):
                    raise PermissionError(
                        'Can not export quant config cause not all quantization configurations '
                        'have been correctly initialized(or some of them has been deactivated). '
                        f'Operation {operation.name} has an invalid quantization state({config.state}) '
                        f'at variable {var.name}.')

                if config.visibility == QuantizationVisibility.INTERNAL: continue
                if config.state in {
                    QuantizationStates.FP32,
                    QuantizationStates.SOI
                }: continue

                if config.state == QuantizationStates.PASSIVE and var.name in activation_info: continue
                info =  [{
                            'bitwidth': config.num_of_bits,
                            'max'     : convert_value(config.scale * (config.quant_max - config.offset), True, DataType.FP32),
                            'min'     : convert_value(config.scale * (config.quant_min - config.offset), True, DataType.FP32),
                            'offset'  : convert_value(config.offset, True, DataType.INT32),
                            'scale'   : convert_value(config.scale, True, DataType.FP32)
                        }]
                if var.is_parameter:
                    param_info[var.name] = info
                else:
                    activation_info[var.name] = info

        exports = {
            'activation_encodings': activation_info,
            'param_encodings': param_info
        }

        with open(file=config_path, mode='w') as file:
            json.dump(exports, file, indent=4)


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        if config_path is not None:
            self.export_quantization_config(config_path, graph)
        if graph._built_from == NetworkFramework.CAFFE:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)
        elif graph._built_from == NetworkFramework.ONNX:
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
