import onnx
from onnx import helper, numpy_helper
from ppq.core import NetworkFramework, QuantizationStates, convert_any_to_numpy
from ppq.IR import (BaseGraph, GraphExporter, Operation, QuantableVariable,
                    Variable)

MIN_SUFFIX = '_min'
MAX_SUFFIX = '_max'

class NxpExporter(GraphExporter):
    def __init__(self) -> None:
        super().__init__()

    def export_operation(self, operation: Operation) -> onnx.OperatorProto:
        op_proto = helper.make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            **operation.attributes)
        return op_proto

    def export_var(self, variable: Variable) -> onnx.TensorProto:
        if not variable.is_parameter:
            tensor_proto = helper.make_tensor_value_info(
                name=variable.name,
                # PPQ data type has exact same eunm value with onnx.
                elem_type=variable.meta.dtype.value,
                shape=variable.meta.shape)
        else:
            value = convert_any_to_numpy(variable.value)
            if value is None: value = []
            else: value = value.flatten()
            tensor_proto = helper.make_tensor(
                name=variable.name, data_type=variable.meta.dtype.value,
                dims=variable.meta.shape,
                vals=value
            )
        return tensor_proto

    def export(self, file_path: str, graph: BaseGraph,
        config_path: str = None, export_param: bool = False):
        onnx_graph = onnx.GraphProto()
        onnx_graph.name = graph._name

        for operation in graph.operations.values():
            onnx_graph.node.append(self.export_operation(operation))

        for variable in graph.variables.values():
            tensor_proto = self.export_var(variable)

            if variable.name in graph.inputs:
                onnx_graph.input.append(tensor_proto)

            if variable.name in graph.outputs:
                onnx_graph.output.append(tensor_proto)

            if variable.is_parameter:
                onnx_graph.initializer.append(tensor_proto)

            if isinstance(variable, QuantableVariable):
                configs = variable.dest_op_configs + [variable.source_op_config]
                if variable.is_parameter and not export_param: continue
                for config in configs:
                    if config is None: continue # source_op can be None
                    if config.can_export():

                        tensor_range = config.scale * pow(2, config.num_of_bits - 1)
                        min_val, max_val = -tensor_range, tensor_range - config.scale
                        min_tensor = numpy_helper.from_array(
                            convert_any_to_numpy(min_val).astype('float32'), variable.name + MIN_SUFFIX)
                        min_info = helper.make_tensor_value_info(
                            variable.name + MIN_SUFFIX, min_tensor.data_type, min_tensor.dims)

                        max_tensor = numpy_helper.from_array(
                            convert_any_to_numpy(max_val).astype('float32'), variable.name + MAX_SUFFIX)
                        max_info = helper.make_tensor_value_info(
                            variable.name + MAX_SUFFIX, max_tensor.data_type, max_tensor.dims)

                        onnx_graph.initializer.append(min_tensor)
                        onnx_graph.initializer.append(max_tensor)
                        onnx_graph.input.append(min_info)
                        onnx_graph.input.append(max_info)
                        break

        onnx_model = helper.make_model(onnx_graph)
        # onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, file_path)
