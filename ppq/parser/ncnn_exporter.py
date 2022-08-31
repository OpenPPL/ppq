import os
from typing import List

from ppq.core import (DataType, NetworkFramework, QuantizationProperty,
                      QuantizationStates)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation

from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value


class NCNNExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        fd = open(config_path, 'w+')
        topo_order =  graph.topological_sort()
        for op in topo_order:
            if op.is_computing_op and isinstance(op, QuantableOperation):
                fd.write(f'{op.name}_param_0 ')
                param_cfg = op.config.input_quantization_config[1]
                if not param_cfg.can_export(): continue

                assert param_cfg.state in {QuantizationStates.BAKED, QuantizationStates.ACTIVATED}\
                    and param_cfg.observer_algorithm in {'minmax', 'Minmax'} and \
                        param_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL)
                # a workaround for depthwise conv in ncnn
                # will cause mis-alignment between ppq and ncnn
                if op.type == 'Conv' and op.attributes.get('group', 1) > 1:
                    group  = op.attributes.get('group', 1)
                    scale  = param_cfg.scale.reshape(group, -1).max(dim=1)[0]
                else:
                    scale  = param_cfg.scale
                scale = convert_value(1 / scale, False, DataType.FP32)
                for s in scale:
                    fd.write('%f '% s)
                fd.write('\n')

        for op in topo_order:
            if op.is_computing_op and isinstance(op, QuantableOperation):
                fd.write(f'{op.name} ')
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                scale = convert_value(1 / input_cfg.scale, True, DataType.FP32)
                fd.write('%f '% scale)
                fd.write('\n')
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
