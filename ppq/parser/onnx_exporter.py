import json
from typing import Union

import numpy as np
import torch
from ppq.core import (EXPORT_DEVICE_SWITCHER, EXPORT_PPQ_INTERNAL_INFO,
                      ONNX_EXPORT_NAME, ONNX_EXPORT_OPSET, ONNX_VERSION,
                      PPQ_NAME, DataType, convert_any_to_numpy)
from ppq.IR import BaseGraph, GraphExporter, Operation, Variable
from ppq.IR.morph import GraphDeviceSwitcher
from ppq.IR.quantize import QuantableOperation

import onnx
from onnx import helper, numpy_helper


def convert_value(value: Union[int, float, np.ndarray, torch.Tensor]) -> str:
    if type(value) in {int, float}: return value
    else:
        value = convert_any_to_numpy(value, accepet_none=True)
        if value is None: return value # SOI config has Nona as its scale and
        return value.tolist()


class OnnxExporter(GraphExporter):
    def __init__(self) -> None:
        super().__init__()

    def export_quantization_config(
        self, config_path: str, graph: BaseGraph):
        
        render_buffer = {
            'configs': {},
            'dispatchings' : {},
            'values': {}
        }
        
        # Render quantization config.
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                op_dict = {
                    var.name: {
                        'bit_width'  : config.num_of_bits,
                        'policy'     : config.policy.to_dict(),
                        'state'      : config.state.name,
                        'quant_min'  : config.quant_min,
                        'quant_max'  : config.quant_max,
                        'hash'       : config.__hash__(),
                        'dominator'  : config.dominated_by.__hash__()
                    }
                    for config, var in operation.config_with_variable
                }
                
                for config, _ in operation.config_with_variable:
                    if config.dominated_by == config:
                        render_buffer['values'][config.__hash__()] = {
                            'scale'      : convert_value(config.scale),
                            'zero_point' : convert_value(config.offset),
                        }

                render_buffer['configs'][operation.name] = op_dict
                render_buffer['dispatchings'][operation.name] = operation.platform.name
        
        with open(file=config_path, mode='w') as file:
            json.dump(render_buffer, file, indent=4)

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
            else: value = value # value is python primary type.
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
        if not name: name = ONNX_EXPORT_NAME

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
            op.version = ONNX_EXPORT_OPSET
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
        onnx_model.ir_version = ONNX_VERSION
        # onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, file_path)
