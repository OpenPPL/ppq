import json
import logging
import os
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
from ppq.core import (DataType, QuantizationProperty, QuantizationStates,
                      TargetPlatform, TensorQuantizationConfig,
                      convert_any_to_numpy,
                      convert_any_to_torch_tensor, ppq_legacy)
from ppq.executor.torch import TorchExecutor
from ppq.IR import BaseGraph, GraphExporter
from ppq.IR.quantize import QuantableOperation

from .caffe import ppl_caffe_pb2
from .caffe.caffe_export_utils import caffe_export_map
from .caffe.caffe_graph_optim import dump_optimize
from .util import convert_value

logger = logging.getLogger('PPQ')

def convert_type(platform: TargetPlatform) -> str:
    if platform == TargetPlatform.PPL_CUDA_INT8: return "INT8"
    if platform == TargetPlatform.DSP_INT8: return "INT8"
    if platform == TargetPlatform.SHAPE_OR_INDEX: return None
    if platform == TargetPlatform.FP32: return None
    raise TypeError(f'Unsupported platform type. ({str(platform)})')

OPEARTION_PLATFROM_FORMAT = \
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
                # we do not support mix presicion quantization for CUDA backend now.
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
            "quant_info": var_quant_info_recorder,
            "op_info": op_platform_recorder}

        with open(file=config_path, mode='w') as file:
            json.dump(exports, file, indent=4)

    def trace(self, graph: BaseGraph, input_shapes: List[List[int]]):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if all([var.value is not None for var in graph.inputs.values()]):
            inputs = {var.name: convert_any_to_torch_tensor(var.value).to(device) for var in graph.inputs.values()}
        elif all([var.meta is not None for var in graph.inputs.values()]):
            inputs = {var.name: torch.randn(*var.meta.shape, dtype=DataType.to_torch(var.meta.dtype), device=device)\
                for var in graph.inputs.values()}
        else:
            assert len([input_shapes]) == len(graph.inputs), "must provide equal number of input shapes for caffe export without quantization"
            # assume all fp32 type, because that's the usual case
            inputs = [torch.randn(*shape, device=device) for shape in input_shapes]
        tracer = TorchExecutor(graph, device=device)
        tracer.tracing_operation_meta(inputs, list(graph.outputs.keys()))

    def export(self, file_path: str, graph: BaseGraph, config_path: str=None, input_shapes: List[List[int]]=[[1,3,224,224]]):
        # it's stupid but for keeping api the same
        prefix,_ = os.path.splitext(file_path)
        prototxt_path = prefix + '.prototxt'
        caffemodel_path = prefix + '.caffemodel'

        self.trace(graph, input_shapes)

        if config_path is not None:
            self.export_quantization_config(config_path, graph)
        caffe_net = ppl_caffe_pb2.NetParameter()
        caffe_net.name = graph._name if graph._name else prefix.split('/')[-1]

        # add caffe input info
        for name, var in graph.inputs.items():
            caffe_net.input.append(name)
            input_shape = ppl_caffe_pb2.BlobShape()
            input_shape.dim.extend(var.meta.shape)
            caffe_net.input_shape.extend([input_shape])

        # export op
        for op in graph.operations.values():
            if op.type not in caffe_export_map:
                logger.error(f'{op.type} converted to Caffe OP is not supported in PPQ export parser yet')
                raise NotImplementedError(f'{op.type} converted to Caffe OP is not supported in PPQ export parser yet')
            caffe_op = caffe_export_map[op.type](op)
            layer = caffe_op.parse()
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            caffe_net.layer.extend(layer)

        dump_optimize(caffe_net)
        prototxt = deepcopy(caffe_net)
        for layer in prototxt.layer:
            del layer.blobs[:]
        logger.info(f'Successfully dump PPQ graph to CAFFE model')
        # Save prototxt, caffemodel
        with open(prototxt_path, 'w') as f:
            f.write(str(prototxt))
        logger.info(f'Save model structure to {prototxt_path}')
        with open(caffemodel_path, 'wb') as f:
            byte = f.write(caffe_net.SerializeToString())
        logger.info(f'Save caffe model to {caffemodel_path} with size {byte / 1000000:.2f}MB')
