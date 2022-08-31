import json
import os
from copy import deepcopy
from typing import List

import torch
from ppq.core import (PPQ_CONFIG, DataType, QuantizationProperty,
                      QuantizationStates, TargetPlatform,
                      TensorQuantizationConfig, convert_any_to_torch_tensor)
from ppq.executor.torch import TorchExecutor
from ppq.IR import BaseGraph, GraphExporter
from ppq.IR.morph import GraphDeviceSwitcher
from ppq.IR.quantize import QuantableOperation, QuantableVariable
from ppq.log import NaiveLogger

from .caffe import ppl_caffe_pb2
from .caffe.caffe_export_utils import CaffeOpExporter, caffe_export_map
from .caffe.caffe_graph_optim import optimize_for_export
from .util import convert_value

logger = NaiveLogger.get_logger('PPQ')

def convert_type(platform: TargetPlatform) -> str:
    if platform == TargetPlatform.PPL_CUDA_INT8: return 'INT8'
    if platform == TargetPlatform.PPL_DSP_INT8: return 'INT8'
    if platform == TargetPlatform.SHAPE_OR_INDEX: return None
    if platform == TargetPlatform.FP32: return None
    raise TypeError(f'Unsupported platform type. ({str(platform)})')

OPEARTION_PLATFORM_FORMAT = \
"""
{name}:(
    platform:\t{platform},
)
"""

class CaffeExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        var_quant_info_recorder, op_platform_recorder = {}, {}
        for operation in graph.operations.values():
            if not isinstance(operation, QuantableOperation): continue
            for config, var in operation.config_with_variable:
                if not config.can_export(): continue

                # PATCH 2021.11.25
                # REMOVE BIAS FROM CONFIGURATION
                if config.num_of_bits > 8:
                        continue

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

    def prepare_model(self, graph: BaseGraph, input_shapes: List[List[int]]):
        # remove device switcher if necessary
        if not PPQ_CONFIG.EXPORT_DEVICE_SWITCHER:
            processor = GraphDeviceSwitcher(graph)
            processor.remove_switcher()

        # trace model for exporting.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if all([var.value is not None for var in graph.inputs.values()]):
            inputs = {var.name: convert_any_to_torch_tensor(var.value).to(device) for var in graph.inputs.values()}
        elif all([var.meta is not None for var in graph.inputs.values()]):
            inputs = {var.name: torch.randn(*var.meta.shape, dtype=DataType.to_torch(var.meta.dtype), device=device)\
                for var in graph.inputs.values()}
        else:
            assert len([input_shapes]) == len(graph.inputs), (
                'must provide equal number of input shapes for caffe export without quantization')
            # assume all fp32 type, because that's the usual case
            inputs = [torch.randn(*shape, device=device) for shape in input_shapes]
        tracer = TorchExecutor(graph, device=device)
        tracer.tracing_operation_meta(inputs, list(graph.outputs.keys()))

        # build caffe protobuf
        caffe_model = ppl_caffe_pb2.NetParameter()
        caffe_model.name = graph._name if graph._name else 'PPQ Exported Caffe Model'

        # add caffe input info
        for name, var in graph.inputs.items():
            caffe_model.input.append(name)
            input_shape = ppl_caffe_pb2.BlobShape()
            var.meta.shape[0] = 1
            input_shape.dim.extend(var.meta.shape)
            caffe_model.input_shape.extend([input_shape])

        # export op
        for op in graph.topological_sort():
            if op.type not in caffe_export_map:
                raise NotImplementedError(
                    f'{op.type} converted to Caffe OP is not supported in PPQ export parser yet')

            caffe_op = caffe_export_map[op.type](op)
            assert isinstance(caffe_op, CaffeOpExporter)

            layer = caffe_op.parse()
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            caffe_model.layer.extend(layer)

        caffe_model = optimize_for_export(caffe_model)
        caffe_proto = deepcopy(caffe_model)
        for layer in caffe_proto.layer: del layer.blobs[:]

        return caffe_model, caffe_proto

    def dump_to_file(self, caffe_model: ppl_caffe_pb2.NetParameter,
                     caffe_proto: ppl_caffe_pb2.NetParameter, file_path: str):

        # it's stupid but for keeping api the same
        # 就是说 caffemodel 总是要保存两个东西出来，但是我们的 api 只留了一个地方
        # 所以我们就只能这样了
        prefix, _       = os.path.splitext(file_path)
        prototxt_path   = prefix + '.prototxt'
        caffemodel_path = prefix + '.caffemodel'

        # Save prototxt, caffemodel
        with open(prototxt_path, 'w') as f:    f.write(str(caffe_proto))
        with open(caffemodel_path, 'wb') as f: f.write(caffe_model.SerializeToString())

    def export(self, file_path: str, graph: BaseGraph, config_path: str=None, input_shapes: List[List[int]]=[[1,3,224,224]]):
        # dump config
        if config_path is not None:
            self.export_quantization_config(config_path=config_path, graph=graph)

        # dump model
        caffe_model, caffe_proto = self.prepare_model(graph, input_shapes)
        self.dump_to_file(caffe_model = caffe_model,
            caffe_proto = caffe_proto, file_path = file_path)

class SNPECaffeExporter(CaffeExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        var_quant_info_recorder = {}
        for operation in graph.operations.values():
            if not isinstance(operation, QuantableOperation): continue
            for config, var in operation.config_with_variable:
                if not QuantizationStates.can_export(config.state):
                    raise PermissionError(
                        'Can not export quant config cause not all quantization configurations '
                        'have been correctly initialized(or some of them has been deactivated). '
                        f'Operation {operation.name} has an invalid quantization config({config.state}) '
                        f'at variable {var.name}.')

                # PATCH 2021.11.25
                # REMOVE BIAS FROM CONFIGURATION
                if config.num_of_bits > 8:
                        continue

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
                'tensor_min' : convert_value(config.scale * (config.quant_min - config.offset), tensorwise, DataType.FP32),
                'tensor_max' : convert_value(config.scale * (config.quant_max - config.offset), tensorwise, DataType.FP32)
            }

        with open(file=config_path, mode='w') as file:
            json.dump(var_quant_info_recorder, file, indent=4)

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        # dump config
        if config_path is not None:
            self.export_quantization_config(config_path=config_path, graph=graph)

        caffe_model, caffe_proto = self.prepare_model(graph, input_shapes)

        # can not edit structure of protobuf with python, have to edit it as pure string.
        caffe_proto, str_buffer = str(caffe_proto), ''
        lines = caffe_proto.split('\n')
        for idx in range(len(lines)):
            line = lines[idx]
            # snpe do not want hole and ceil_mode
            if 'hole' in line or 'ceil_mode' in line: continue
            # snpe do not want quantize_param
            if 'quantize_param' in line:
                idx += 4
                continue
            str_buffer = (str_buffer + line) + '\n'
        caffe_proto = str_buffer

        # dump model
        self.dump_to_file(
            caffe_model = caffe_model,
            caffe_proto = caffe_proto,
            file_path = file_path)


class PPLDSPCaffeExporter(CaffeExporter):
    def export(self, file_path: str, graph: BaseGraph, config_path: str = None,
               input_shapes: List[List[int]] = [[1, 3, 224, 224]], write_weight=False):
        # PPL3 DSP do not need a json config file, all quantization configuration will be merged into protobuf
        caffe_model, caffe_proto = self.prepare_model(graph, input_shapes)
        for idx in range(len(caffe_proto.layer)):
            layer = caffe_proto.layer[idx]
            layer_name = layer.name

            assert isinstance(layer, ppl_caffe_pb2.LayerParameter)
            assert isinstance(layer_name, str)
            # layer is a caffe data structure, corresponding to operation in ppq.
            # following code write ppq quantization configuration to caffe layer.

            # step - 1, find corresponding op
            if layer_name not in graph.operations:
                # PATCH FOR Slice
                if layer.type == 'Slice':
                    # slice0 --> (ppq parse, export, combine) --> slice0_0_0, everything else
                    # is the same with original model
                    # layer_name is slice0_0_0, obtain original_name slice0
                    original_name = ''.join(layer_name.split('_')[:-2])
                    for bottom in layer.bottom:
                        var = graph.variables.get(bottom, None)
                        if var is not None and isinstance(var, QuantableVariable) and not var.is_parameter:
                            cfg = None
                            for dest_op, dest_cfg in zip(var.dest_ops, var.dest_op_configs):
                                # dest_op.name is slice0_0
                                if ''.join(dest_op.name.split('_')[:-1]) == original_name:
                                    cfg = dest_cfg
                                    break
                            assert cfg is not None
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            layer.quantize_param.add(type='bottom', range_min=qt_min, range_max=qt_max)

                    for top in layer.top:
                        var = graph.variables.get(top, None)
                        if var is not None and isinstance(var, QuantableVariable) and not var.is_parameter:
                            cfg = var.source_op_config
                            assert cfg is not None
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            layer.quantize_param.add(type='top', range_min=qt_min, range_max=qt_max)
                    continue
                else:
                    raise KeyError(f'Can not find operation {layer_name} with current graph.')

            op = graph.operations[layer_name]
            if not isinstance(op, QuantableOperation): continue

            # step - 2, dump parameter quantization config
            if layer.type in {'Convolution', 'Deconvolution'}:
                for cfg, var in op.config_with_variable:
                    if not var.is_parameter: continue
                    if cfg.num_of_bits > 8: continue # skip bias

                    if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), False, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), False, DataType.FP32)
                        for _min, _max in zip(qt_min, qt_max):
                            p = layer.convolution_param.perchannel_quantize_param.add()
                            p.type = 'filter'
                            p.range_min = _min
                            p.range_max = _max

                    if cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                        p = layer.convolution_param.quantize_param
                        p.type = 'filter'
                        p.range_min = qt_min
                        p.range_max = qt_max

            if layer.type == 'InnerProduct':
                for cfg, var in op.config_with_variable:
                    if not var.is_parameter: continue
                    if cfg.num_of_bits > 8: continue # skip bias

                    if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), False, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), False, DataType.FP32)
                        for _min, _max in zip(qt_min, qt_max):
                            p = layer.inner_product_param.perchannel_quantize_param.add()
                            p.type = 'filter'
                            p.range_min = _min
                            p.range_max = _max

                    if cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                        p = layer.inner_product_param.quantize_param
                        p.type = 'filter'
                        p.range_min = qt_min
                        p.range_max = qt_max

            if layer.type == 'PReLU':
                for cfg, var in op.config_with_variable:
                    if not var.is_parameter: continue

                    if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), False, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), False, DataType.FP32)
                        for _min, _max in zip(qt_min, qt_max):
                            p = layer.prelu_param.perchannel_quantize_param.add()
                            p.type = 'slope'
                            p.range_min = _min
                            p.range_max = _max

                    if cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                        p = layer.prelu_param.quantize_param
                        p.type = 'slope'
                        p.range_min = qt_min
                        p.range_max = qt_max

            # step - 3 dump input config
            # only ops related with graph inputs record bottom param
            # usually first conv op
            record_bottom = False
            for input_var in op.inputs:
                if input_var.name in graph.inputs:
                    record_bottom = True
                    break
            if record_bottom:
                for bottom in layer.bottom:
                    for cfg, var, in zip(op.config.input_quantization_config, op.inputs):
                        if var.name == str(bottom):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            if var.name in graph.inputs:
                                range_min = convert_value(cfg.detail.get('range_min', -1), True, DataType.FP32)
                                # no negative values in input
                                if range_min >= 0.0:
                                    qt_min = 0.0
                            layer.quantize_param.add(type='bottom', range_min=qt_min, range_max=qt_max)

            # step - 4 dump output config
            for top in layer.top:
                for cfg, var, in zip(op.config.output_quantization_config, op.outputs):
                    assert cfg.policy.has_property(QuantizationProperty.PER_TENSOR)

                    if var.name == str(top):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                        layer.quantize_param.add(type='top', range_min=qt_min, range_max=qt_max)

        # dump model
        self.dump_to_file(
            caffe_model = caffe_model,
            caffe_proto = caffe_proto,
            file_path = file_path)


class PPLDSPTICaffeExporter(CaffeExporter):
    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, \
                input_shapes: List[List[int]] = [[1, 3, 224, 224]], write_weight=False):
        # PPL3 DSP TI do not need a json config file, all quantization configuration will be merged into protobuf
        # DSP TI differs from DSP in the perchannel quantization param for computing ops
        caffe_model, caffe_proto = self.prepare_model(graph, input_shapes)
        for idx in range(len(caffe_proto.layer)):
            layer = caffe_proto.layer[idx]
            layer_name = layer.name

            assert isinstance(layer, ppl_caffe_pb2.LayerParameter)
            assert isinstance(layer_name, str)
            # layer is a caffe data structure, corresponding to operation in ppq.
            # following code write ppq quantization configuration to caffe layer.

             # step - 1, find corresponding op
            if layer_name not in graph.operations:
                # PATCH FOR Slice
                if layer.type == 'Slice':
                    # slice0 --> (ppq parse, export, combine) --> slice0_0_0, everything else
                    # is the same with original model
                    # layer_name is slice0_0_0, obtain original_name slice0
                    original_name = ''.join(layer_name.split('_')[:-2])
                    for bottom in layer.bottom:
                        var = graph.variables.get(bottom, None)
                        if var is not None and isinstance(var, QuantableVariable) and not var.is_parameter:
                            cfg = None
                            for dest_op, dest_cfg in zip(var.dest_ops, var.dest_op_configs):
                                # dest_op.name is slice0_0
                                if ''.join(dest_op.name.split('_')[:-1]) == original_name:
                                    cfg = dest_cfg
                                    break
                            assert cfg is not None
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            layer.quantize_param.add(type='bottom', range_min=qt_min, range_max=qt_max)

                    for top in layer.top:
                        var = graph.variables.get(top, None)
                        if var is not None and isinstance(var, QuantableVariable) and not var.is_parameter:
                            cfg = var.source_op_config
                            assert cfg is not None
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            layer.quantize_param.add(type='top', range_min=qt_min, range_max=qt_max)
                    continue
                else:
                    raise KeyError(f'Can not find operation {layer_name} with current graph.')

            op = graph.operations[layer_name]
            if not isinstance(op, QuantableOperation): continue

            # don't dump by default
            if write_weight:
                if layer.type in {'Convolution', 'Deconvolution'}:
                    for cfg, var in op.config_with_variable:
                        if not var.is_parameter: continue
                        if cfg.num_of_bits > 8: continue # skip bias

                        if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), False, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), False, DataType.FP32)
                            for _min, _max in zip(qt_min, qt_max):
                                p = layer.convolution_param.perchannel_quantize_param.add()
                                p.type = 'filter'
                                p.range_min = _min
                                p.range_max = _max

                        if cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            p = layer.convolution_param.quantize_param
                            p.type = 'filter'
                            p.range_min = qt_min
                            p.range_max = qt_max

                if layer.type == 'InnerProduct':
                    for cfg, var in op.config_with_variable:
                        if not var.is_parameter: continue
                        if cfg.num_of_bits > 8: continue # skip bias

                        if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), False, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), False, DataType.FP32)
                            for _min, _max in zip(qt_min, qt_max):
                                p = layer.inner_product_param.perchannel_quantize_param.add()
                                p.type = 'filter'
                                p.range_min = _min
                                p.range_max = _max

                        if cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            p = layer.inner_product_param.quantize_param
                            p.type = 'filter'
                            p.range_min = qt_min
                            p.range_max = qt_max

                if layer.type == 'PReLU':
                    for cfg, var in op.config_with_variable:
                        if not var.is_parameter: continue

                        if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), False, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), False, DataType.FP32)
                            for _min, _max in zip(qt_min, qt_max):
                                p = layer.prelu_param.perchannel_quantize_param.add()
                                p.type = 'slope'
                                p.range_min = _min
                                p.range_max = _max

                        if cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            p = layer.prelu_param.quantize_param
                            p.type = 'slope'
                            p.range_min = qt_min
                            p.range_max = qt_max

            # step - 3 dump input config
            # only ops related with graph inputs record bottom param
            # usually first conv op
            record_bottom = False
            for input_var in op.inputs:
                if input_var.name in graph.inputs:
                    record_bottom = True
                    break
            if record_bottom:
                for bottom in layer.bottom:
                    for cfg, var, in zip(op.config.input_quantization_config, op.inputs):
                        if var.name == str(bottom):
                            qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                            qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                            if var.name in graph.inputs:
                                range_min = convert_value(cfg.detail.get('range_min', -1), True, DataType.FP32)
                                # no negative values in input
                                if range_min >= 0.0:
                                    qt_min = 0.0
                            layer.quantize_param.add(type='bottom', range_min=qt_min, range_max=qt_max)

            # step - 4 dump output config
            for top in layer.top:
                for cfg, var, in zip(op.config.output_quantization_config, op.outputs):
                    assert cfg.policy.has_property(QuantizationProperty.PER_TENSOR)

                    if var.name == str(top):
                        qt_min = convert_value(cfg.scale * (cfg.quant_min - cfg.offset), True, DataType.FP32)
                        qt_max = convert_value(cfg.scale * (cfg.quant_max - cfg.offset), True, DataType.FP32)
                        layer.quantize_param.add(type='top', range_min=qt_min, range_max=qt_max)

                    if op.is_computing_op:
                        for _min, _max in zip(cfg.detail['range_min'], cfg.detail['range_max']):
                            layer.quantize_param.add(type='topperchannel', range_min=_min, range_max=_max)

        # dump model
        self.dump_to_file(
            caffe_model = caffe_model,
            caffe_proto = caffe_proto,
            file_path = file_path)
