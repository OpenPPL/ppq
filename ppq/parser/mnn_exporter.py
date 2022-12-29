
import os
from typing import List
import json

from ppq.core import (DataType, NetworkFramework, QuantizationProperty,
                      QuantizationStates, ppq_warning, QuantizationPolicy)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation

from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value

ASCEND_QUANT_OP = {"Conv", "ConvTranspose", "Gemm", "AveragePool"}

class MNNExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        quant_info_json = {}

        # opHasBeenVisited = []
        op_tensor_scales = []
        op_tensor_names = []

        for op in graph.topological_sort():
            # process the first conv
            if op.type == "Conv":
                op_tensor_scales.clear()
                op_tensor_names.clear()
                for cfg, var in op.config_with_variable:
                    if not cfg.can_export(export_overlapped=True): 
                        continue
                    if var.is_parameter: 
                        continue
                    op_tensor_scales.append(cfg.scale.item())
                    op_tensor_names.append(var.name)
                    
                assert len(op_tensor_scales)==len(op_tensor_names)
                
                base_name = op_tensor_names[1]
                input_tensor_name = base_name + "_input_tensor_0"
                output_tensor_name = base_name + "_output_tensor_0"
                quant_info_json[input_tensor_name] = op_tensor_scales[0]
                quant_info_json[output_tensor_name] = op_tensor_scales[1]

        json_qparams_str = json.dumps(quant_info_json, indent=4)
        with open(config_path, "w") as json_file:
            json_file.write(json_qparams_str)


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

        # import pdb
        # pdb.set_trace()

        _, ext = os.path.splitext(file_path)
        if ext == '.onnx':
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
        elif ext in {'.prototxt', '.caffemodel'}:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)
        
        # no pre-determined export format, we export according to the
        # original model format
        elif graph._built_from == NetworkFramework.CAFFE:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)

        elif graph._built_from == NetworkFramework.ONNX:
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
