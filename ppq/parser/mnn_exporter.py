import os
from typing import List
import json

from ppq.core import NetworkFramework
from ppq.IR import BaseGraph,GraphExporter
from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter


class MNNExporter(GraphExporter):
    def export_onnx_quantization_config(self, config_path: str, graph: BaseGraph):
        quant_info_json = {}
        shape = {}
        op_tensor_scales = []
        op_tensor_names = []
        for input_name in graph.inputs.keys():
            quant_var = graph.inputs[input_name]
            shape["channels"] = quant_var.shape[1]
            shape["height"] = quant_var.shape[2]
            shape["width"] = quant_var.shape[3]
        quant_info_json["shape"] = shape

        for op in graph.topological_sort():
            if op.type in {"Conv", 'Add'}:
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
                if op.type == "Conv":
                    base_name = op_tensor_names[1]
                    input_tensor_name = base_name + " input_tensor_0"
                    output_tensor_name = base_name + " output_tensor_0"
                    quant_info_json[input_tensor_name] = op_tensor_scales[0]
                    quant_info_json[output_tensor_name] = op_tensor_scales[1]
                if op.type == "Add":
                    base_name = op_tensor_names[2]
                    output_tensor_name = base_name + " output_tensor_0"
                    quant_info_json[output_tensor_name] = op_tensor_scales[1]
        json_qparams_str = json.dumps(quant_info_json, indent=4)
        with open(config_path, "w") as json_file:
            json_file.write(json_qparams_str)

    def export_caffe_quantization_config(self, config_path: str, graph: BaseGraph):
        quant_info_json = {}
        shape = {}
        op_tensor_scales = []
        op_tensor_names = []
        for input_name in graph.inputs.keys():
            quant_var = graph.inputs[input_name]
            shape["channels"] = quant_var.shape[1]
            shape["height"] = quant_var.shape[2]
            shape["width"] = quant_var.shape[3]
        quant_info_json["shape"] = shape

        for op in graph.topological_sort():
            if op.type in {"Conv", 'Add', 'Gemm'}:
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
                if op.type in {"Conv","Gemm"}:
                    base_name = op.name
                    input_tensor_name = base_name + " input_tensor_0"
                    output_tensor_name = base_name + " output_tensor_0"
                    quant_info_json[input_tensor_name] = op_tensor_scales[0]
                    quant_info_json[output_tensor_name] = op_tensor_scales[1]
                if op.type == "Add":
                    base_name = op.name
                    output_tensor_name = base_name + " output_tensor_0"
                    quant_info_json[output_tensor_name] = op_tensor_scales[1]
        json_qparams_str = json.dumps(quant_info_json, indent=4)
        with open(config_path, "w") as json_file:
            json_file.write(json_qparams_str)


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):

        if graph._built_from == NetworkFramework.CAFFE:
            if config_path is not None:
                self.export_caffe_quantization_config(config_path, graph)

            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)

        elif graph._built_from == NetworkFramework.ONNX:
            if config_path is not None:
                self.export_onnx_quantization_config(config_path, graph)
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)