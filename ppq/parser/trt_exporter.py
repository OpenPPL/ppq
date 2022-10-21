import os
import sys
import json
import struct
from typing import List
from ppq.core import (DataType, PPQ_CONFIG, NetworkFramework, QuantizationProperty,
                      QuantizationStates)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation
from ppq.IR.morph import GraphDeviceSwitcher
from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value
from ppq.core import ppq_warning
from ppq.core import ppq_info

class TensorrtExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        quant_info = {}
        act_quant_info = {}
        quant_info["act_quant_info"] = act_quant_info

        topo_order =  graph.topological_sort()

        for index, op in enumerate(topo_order):
            
            if op.type in {"Shape", "Gather", "Unsqueeze", "Concat", "Reshape"}:
               continue
            
            if index == 0:
                assert graph.inputs.__contains__(op.inputs[0].name)
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                trt_range_input = input_cfg.scale.item() * (input_cfg.quant_max - input_cfg.quant_min) / 2
                act_quant_info[op.inputs[0].name] = trt_range_input
                output_cfg = op.config.output_quantization_config[0]
                trt_range_output = output_cfg.scale.item() * (output_cfg.quant_max - output_cfg.quant_min) / 2
                act_quant_info[op.outputs[0].name] = trt_range_input

            else:
                if not hasattr(op, 'config'):
                    ppq_warning(f'This op does not write quantization parameters: {op.name}.')
                    continue
                else:
                    output_cfg = op.config.output_quantization_config[0]
                    trt_range_output = output_cfg.scale.item() * (output_cfg.quant_max - output_cfg.quant_min) / 2
                    act_quant_info[op.outputs[0].name] = trt_range_output

        json_qparams_str = json.dumps(quant_info, indent=4)
        with open(config_path, "w") as json_file:
            json_file.write(json_qparams_str)

    def export_weights(self, graph: BaseGraph, config_path: str = None):
        topo_order =  graph.topological_sort()
        weights_list = []
        for index, op in enumerate(topo_order):
            if op.type in {"Conv", "Gemm"}:
                weights_list.extend(op.parameters)

        weight_file_path = os.path.join(os.path.dirname(config_path), "quantized.wts")

        f = open(weight_file_path, 'w')
        f.write("{}\n".format(len(weights_list)))

        for param in weights_list:
            weight_name = param.name
            weight_value = param.value.reshape(-1).cpu().numpy()
            f.write("{} {}".format(weight_name, len(weight_value)))
            for value in weight_value:
                f.write(" ")
                f.write(struct.pack(">f", float(value)).hex())
            f.write("\n")
        ppq_info(f'Parameters have been saved to file: {weight_file_path}')


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        if not PPQ_CONFIG.EXPORT_DEVICE_SWITCHER:
            processor = GraphDeviceSwitcher(graph)
            processor.remove_switcher()

        self.export_weights(graph, config_path)

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
