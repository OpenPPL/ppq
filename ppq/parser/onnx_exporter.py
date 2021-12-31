import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from ppq.core import (PPQ_NAME, DataType, convert_any_to_numpy,
                      convert_any_to_string)
from ppq.core.config import EXPORT_DEVICE_SWITCHER, EXPORT_PPQ_INTERNAL_INFO
from ppq.IR import (BaseGraph, GraphExporter, Operation, QuantableVariable,
                    Variable)
from ppq.IR.morph import GraphDeviceSwitcher

QUANTIZATION_CONFIG_FORMAT = \
"""
{name}:(
    num_of_bits:\t{num_of_bits},
    scale:\t{scale},
    offset:\t{offset},
    quant_min:\t{quant_min},
    quant_max:\t{quant_max},
    state:\t{state},
    policy:\t{policy},
    detail:\t{detail},
    observer:\t{observer},
    dominator:\t{dominated_by}
)
"""

OPEARTION_PLATFROM_FORMAT = \
"""
{name}:(
    platform:\t{platform},
)
"""

class OnnxExporter(GraphExporter):
    def __init__(self) -> None:
        super().__init__()

    def export_quantization_config(
        self,
        config_path: str, graph: BaseGraph):
        file_handler = open(config_path, mode='w', encoding='utf-8')
        quantable_vars = [var for var in graph.variables.values() if isinstance(var, QuantableVariable)]
        quantable_vars = sorted(quantable_vars, key=lambda x: x.name)
        for var in quantable_vars:
            related_configs = var.dest_op_configs + [var.source_op_config]
            related_op      = var.dest_ops + [var.source_op]
            for op, config in zip(related_op, related_configs):
                if config is not None:
                    file_handler.write(QUANTIZATION_CONFIG_FORMAT.format(
                        name = op.name + ': ' + var.name + '(' + str(config.__hash__()) + ')',
                        num_of_bits = config.num_of_bits,
                        scale = convert_any_to_string(config.scale), 
                        offset = convert_any_to_string(config.offset),
                        quant_min = config.quant_min,
                        quant_max = config.quant_max,
                        state = str(config.state.name),
                        policy = config.policy._policy,
                        detail = config.detail,
                        observer = config.observer_algorithm,
                        dominated_by = config.dominated_by.__hash__()
                    ))

        ops = [op for op in graph.operations.values()]
        ops = sorted(ops, key=lambda x: x.name)
        for op in ops:
            file_handler.write(OPEARTION_PLATFROM_FORMAT.format(
                name = op.name,
                platform = str(op.platform.name)
            ))
        file_handler.close()

    def export_operation(self, operation: Operation) -> onnx.OperatorProto:
        attributes = operation.attributes.copy()
        # PATCH 20211203, ConstantOfShape Op causes an export error.
        # 这一问题是由 ConstantOfShape 中的 value 格式问题引发的，下面的代码将导出正确的格式
        if operation.type == 'ConstantOfShape':
            attributes['value'] = numpy_helper.from_array(attributes['value'])

        # PATCH 20211206, MMCVRoiAlign operation must have a domain attribute.
        if operation.type in {'MMCVRoiAlign'}:
            attributes['domain'] = 'mmcv'
        
        # PATCH 20211206, grid_sampler operation must have a domain attribute.
        if operation.type in {'grid_sampler'}:
            attributes['domain'] = 'mmcv'
            
        # PATCH 20211216, interp op can not export input_shape attribute.
        if operation.type == 'Interp':
            attributes.pop('input_shape')

        for key in attributes:
            value = attributes[key]
            if isinstance(value, DataType):
                attributes[key] = value.value
            if isinstance(value, torch.Tensor):
                attributes[key] = convert_any_to_numpy(value)

        if EXPORT_PPQ_INTERNAL_INFO:
            attributes['platform'] = operation.platform.name
        op_proto = helper.make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            **attributes)
        return op_proto

    def export_var(self, variable: Variable) -> onnx.TensorProto:
        if variable.meta is not None:
            shape = variable.meta.shape
            dtype = variable.meta.dtype.value
        else:
            shape, dtype = None, None

        if not variable.is_parameter:
            tensor_proto = helper.make_tensor_value_info(
                name=variable.name,
                # PPQ data type has exact same eunm value with onnx.
                elem_type=dtype,
                shape=shape)
        else:
            value = variable.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    value = []
                elif value.ndim >= 1:
                    value = convert_any_to_numpy(variable.value).flatten()
                elif value.ndim == 0:
                    value = [value.item(), ] # it is fine for onnx, cause shape for this value will be []
            else: value = value # value is numpy.ndarray or python primary type.
            tensor_proto = helper.make_tensor(
                name=variable.name, data_type=dtype,
                dims=shape, vals=value)
        return tensor_proto

    def export(self, file_path: str, graph: BaseGraph, config_path: str = None):
        # during export we will remove all boundary operations from graph.
        # we do not want to change the sturcture of original graph,
        # so there have to take a clone of it.
        # graph = graph.copy()
        # remove switchers.
        if not EXPORT_DEVICE_SWITCHER:
            processer = GraphDeviceSwitcher(graph)
            processer.remove_switcher()

        name = graph._name
        if not name: name = 'PPL Quantization Tool - Onnx Export'

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            self.export_quantization_config(config_path, graph)
        
        # Ready to export onnx graph defination.
        _inputs, _outputs, _initilizers, _nodes = [], [], [], []
        for operation in graph.topological_sort():
            _nodes.append(self.export_operation(operation))

        for variable in graph.variables.values():
            tensor_proto = self.export_var(variable)
            if variable.name in graph.inputs:
                _inputs.append(tensor_proto)
            if variable.name in graph.outputs:
                _outputs.append(tensor_proto)
            if variable.is_parameter:
                _initilizers.append(tensor_proto)

        graph_def = helper.make_graph(
            name=name,
            nodes=_nodes,
            inputs=_inputs,
            outputs=_outputs,
            initializer=_initilizers,
        )

        # force opset to 11
        if 'opsets' not in graph._detail:
            op = onnx.OperatorSetIdProto()
            op.version = 11
            opsets = [op]
        else:
            opsets = []
            for opset in graph._detail['opsets']:
                op = onnx.OperatorSetIdProto()
                op.domain = opset['domain']
                op.version = opset['version']
                opsets.append(op)

        onnx_model = helper.make_model(
            graph_def, producer_name=PPQ_NAME, opset_imports=opsets)
        onnx_model.ir_version = 6
        # onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, file_path)
