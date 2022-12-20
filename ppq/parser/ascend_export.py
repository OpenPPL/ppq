
import os
from typing import List

from ppq.core import (DataType, NetworkFramework, QuantizationProperty,
                      QuantizationStates, ppq_warning)
from ppq.IR import BaseGraph, GraphExporter

from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value

ASCEND_QUANT_OP = {"Conv", "ConvTranspose", "Gemm", "AveragePool"}

FLT_EPSILON = 1.1920929e-7

def adapt_scale(op, scale):
    min = FLT_EPSILON
    max = 1.0 / FLT_EPSILON
    if scale < min:
        scale = FLT_EPSILON
        ppq_warning(f'{op.name} scale is too small: {scale}.')
    elif scale > max:
        scale = max
        ppq_warning(f'{op.name} scale is too large: {scale}.')
    return scale

def check_offset(offset):
    if offset>127 or offset < -128:
        raise RuntimeError(f'This offset value {offset} does not belong to the range [-128,127].')

def generate_shape(shape):
    channels = ""
    height = ""
    width = ""
    if len(shape) == 2:
        channels = shape[1]
        height = 1
        width = 1
    elif len(shape) == 4:
        _, channels, height, width = shape
    else:
        raise RuntimeError(f'Please design this shape yourself.')
    return channels, height, width

class AscendExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        new_config_path = os.path.splitext(config_path)[0] + ".txt"
        matched_nodes = []

        for op in graph.topological_sort():
            if op.type in {"Conv", "ConvTranspose", "Gemm"}:
                if not hasattr(op, 'config'):
                    ppq_warning(f'This op does not write quantization parameters: {op.name}.')
                    continue
                op_name = '''\"''' + op.name + '''\"'''
                quant_unit_list = []
                quant_unit_list.append("record {\n")
                quant_unit_list.append("  key: " + op_name + "\n")
                quant_unit_list.append("  value {\n")

                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)

                scale_d = input_cfg.scale.item()
                offset_d = int(input_cfg.offset.item()) - 128

                check_offset(offset_d)
                quant_unit_list.append("    scale_d: " + str(adapt_scale(op, scale_d)) + "\n")
                quant_unit_list.append("    offset_d: " + str(offset_d) + "\n")

                weight_config = op.config.input_quantization_config[1]
                scale_list = convert_value(weight_config.scale, False, DataType.FP32)

                if op.type == "Gemm":
                    assert isinstance(scale_list,float), 'Gemm can only have one scale.'
                    scale_list = [scale_list]

                for scale_w in scale_list:
                    quant_unit_list.append("    scale_w: " + str(adapt_scale(op, scale_w)) + "\n")

                for _ in range(len(scale_list)):
                    quant_unit_list.append("    offset_w: " + "0" + "\n")

                channels, height, width = generate_shape(op.inputs[0].shape)

                quant_unit_list.append("    channels: " + str(channels) + "\n")
                quant_unit_list.append("    height: " + str(height) + "\n")
                quant_unit_list.append("    width: " + str(width) + "\n")
                quant_unit_list.append("  }" + "\n")
                quant_unit_list.append("}" + "\n")
                matched_nodes.append(quant_unit_list)


            elif op.type == "AveragePool":
                if not hasattr(op, 'config'):
                    ppq_warning(f'This op does not write quantization parameters: {op.name}.')
                    continue
                op_name = '''\"''' + op.name + '''\"'''
                quant_unit_list = []
                quant_unit_list.append("record {\n")
                quant_unit_list.append("  key: " + op_name + "\n")
                quant_unit_list.append("  value {\n")
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                scale_d = input_cfg.scale.item()
                
                offset_d = int(input_cfg.offset.item()) - 128
                check_offset(offset_d)
                quant_unit_list.append("    scale_d: " + str(adapt_scale(op, scale_d)) + "\n")
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
