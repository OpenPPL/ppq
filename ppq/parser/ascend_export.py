import json
from typing import Dict
import os

import torch
from ppq.core import (DataType, OperationMeta, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, TensorMeta,
                      TensorQuantizationConfig, convert_any_to_torch_tensor,
                      ppq_warning)
from ppq.IR import BaseGraph
from ppq.IR.quantize import QuantableOperation, QuantableVariable
from ppq.utils.round import ppq_tensor_round

from typing import List

from ppq.core import (DataType, NetworkFramework, QuantizationProperty,
                      QuantizationStates)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation

from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value

ASCEND_QUANT_OP = {"Conv", "ConvTranspose", "Gemm", "AveragePool"}

FLT_EPSILON = 1.1920929e-10

def adapt_scale(scale):
    min = FLT_EPSILON
    max = 1.0 / FLT_EPSILON

    if scale < min:
        scale = FLT_EPSILON
        ppq_warning(f'scale is too small: {scale}.')
    elif scale > max:
        scale = max
        ppq_warning(f'scale is too large: {scale}.')
    return scale

class AscendExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        new_config_path = os.path.splitext(config_path)[0] + ".txt"
        matched_nodes = []

        for op in graph.topological_sort():

            if op.type == "Conv":
                quant_unit_list = []
                quant_unit_list.append("record {\n")
                quant_unit_list.append("  key: " + op.name + "\n")
                quant_unit_list.append("  value {\n")
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)

                scale_d = input_cfg.scale.item()
                offset_d = 0
                quant_unit_list.append("    scale_d: " + str(adapt_scale(scale_d)) + "\n")
                quant_unit_list.append("    offset_d: " + str(offset_d) + "\n")
                
                kernel_nums = op.parameters[0].shape[0]
                for num in range(kernel_nums):
                    kernel_value = op.parameters[0].value[num]
                    max_value = max(torch.abs(kernel_value.min()).item(), torch.abs(kernel_value.max()).item())
                    scale_w = max_value / 127.0
                    quant_unit_list.append("    scale_w: " + str(adapt_scale(scale_w)) + "\n")
                
                for num in range(kernel_nums):
                    quant_unit_list.append("    offset_w: " + "0" + "\n")
                
                _, channels, height, width = op.inputs[0].shape
                quant_unit_list.append("    channels: " + str(channels) + "\n")
                quant_unit_list.append("    height: " + str(height) + "\n")
                quant_unit_list.append("    width: " + str(width) + "\n")
                quant_unit_list.append("  }" + "\n")
                quant_unit_list.append("}" + "\n")
                matched_nodes.append(quant_unit_list)
            
            elif op.type == "Gemm":
                quant_unit_list = []
                quant_unit_list.append("record {\n")
                quant_unit_list.append("  key: " + op.name + "\n")
                quant_unit_list.append("  value {\n")
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)

                scale_d = input_cfg.scale.item()
                offset_d = 0
                quant_unit_list.append("    scale_d: " + str(adapt_scale(scale_d)) + "\n")
                quant_unit_list.append("    offset_d: " + str(offset_d) + "\n")
                param_values = op.parameters[0].value
                max_value = max(torch.abs(param_values.min()).item(), torch.abs(param_values.max()).item())
                scale_w = max_value / 127.0
                quant_unit_list.append("    scale_w: " + str(adapt_scale(scale_w)) + "\n")
                quant_unit_list.append("    offset_w: " + "0" + "\n")
                _, channels, height, width = op.inputs[0].shape
                quant_unit_list.append("    channels: " + str(channels) + "\n")
                quant_unit_list.append("    height: " + str(height) + "\n")
                quant_unit_list.append("    width: " + str(width) + "\n")
                quant_unit_list.append("  }" + "\n")
                quant_unit_list.append("}" + "\n")
                matched_nodes.append(quant_unit_list)

            elif op.type == {"AveragePool", "ConvTranspose"}:
                quant_unit_list = []
                quant_unit_list.append("record {\n")
                quant_unit_list.append("  key: " + op.name + "\n")
                quant_unit_list.append("  value {\n")
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                scale_d = input_cfg.scale.item()
                offset_d = 0
                quant_unit_list.append("    scale_d: " + str(adapt_scale(scale_d)) + "\n")
                quant_unit_list.append("    offset_d: " + str(offset_d) + "\n")
                _, channels, height, width = op.inputs[0].shape
                quant_unit_list.append("    channels: " + str(channels) + "\n")
                quant_unit_list.append("    height: " + str(height) + "\n")
                quant_unit_list.append("    width: " + str(width) + "\n")
                quant_unit_list.append("  }" + "\n")
                quant_unit_list.append("}" + "\n")
                matched_nodes.append(quant_unit_list)
            else:
                continue
        fd = open(new_config_path, 'w+')
        for tem_list in matched_nodes:
            for tem_str in tem_list:
                fd.write(tem_str)
        fd.close()

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

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
