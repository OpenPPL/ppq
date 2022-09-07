from typing import Any, Dict, Iterable, List

import onnx
from onnx import helper, mapping, numpy_helper
from ppq.core import (DEFAULT_OPSET_DOMAIN, DEFAULT_OPSET_VERSION,
                      GRAPH_OPSET_ATTRIB, NetworkFramework, is_file_exist)
from ppq.IR import BaseGraph, GraphBuilder, Operation, Opset, Variable

class OnnxParser(GraphBuilder):
    def build_variables(
        self, graph: BaseGraph,
        graph_inputs: List[str], graph_outputs: List[str],
        op_inputs: Dict[str, list], op_outputs: Dict[str, list]) -> BaseGraph:
        var_list = []

        for op_name, _ in graph.operations.items():
            for var_name in op_inputs[op_name]: var_list.append(var_name)
            for var_name in op_outputs[op_name]: var_list.append(var_name)

        # create all variable at once.
        for var_name in set(var_list):
            graph.variables[var_name] = Variable(name=var_name)

        # build graph's input, output variables.
        try:
            for var_name in graph_inputs:
                if var_name not in graph.variables: continue
                graph.inputs[var_name] = graph.variables[var_name]
            for var_name in graph_outputs:
                graph.outputs[var_name] = graph.variables[var_name]
        except KeyError as e:
            raise KeyError(
                'seems you got an input/output variable that is not linked to any operation.')

        # build operation inputs, outputs variables.
        for op in graph.operations.values():
            for var_name in op_inputs[op.name]:
                var = graph.variables[var_name]
                var.dest_ops.append(op)
                op.inputs.append(graph.variables[var_name])
            for var_name in op_outputs[op.name]:
                var = graph.variables[var_name]
                var.source_op = op
                op.outputs.append(graph.variables[var_name])
        return graph

    def initialize_params(self, graph: BaseGraph, initializer: Dict[str, Any]) -> BaseGraph:
        for var in graph.variables.values():
            if var.name in initializer:
                for dest_op in var.dest_ops:
                    assert isinstance(dest_op, Operation)
                    dest_op.parameters.append(var)
                var.value = initializer[var.name]
                var.is_parameter = True
        return graph


    def de_inplace(self, graph: BaseGraph) -> BaseGraph:
        """Remove inplace layer in netdef If the names of bottom and top are same,
        it means the computation of this layer is in place."""
        def new_name(_name):
            if _name == '': return ''
            if current_write_times[_name] == total_write_times[_name]:
                return _name
            else:
                return f'{_name}_ver{current_write_times[_name]}'

        total_write_times = {}
        for op in graph.operations.values():
            for top in op.outputs:
                total_write_times.setdefault(top._name, 0)
                total_write_times[top._name] += 1

        current_write_times = {}
        for name in graph.inputs.keys():
            total_write_times[name] = 0
            current_write_times[name] = 0

        for op in graph.operations.values():
            for bottom in op.inputs:
                if bottom.is_parameter:
                    continue
                bottom._name = new_name(bottom._name)
            for top in op.outputs:
                current_write_times.setdefault(top._name, 0)
                current_write_times[top._name] += 1
                top._name = new_name(top._name)


    def refine_graph(self, graph: BaseGraph) -> BaseGraph:
        for op in graph.operations.values():
            for key, value in op.attributes.items():
                if isinstance(value, bytes):
                    # Change bytes to string
                    value = value.decode('utf-8')
                if op.type == 'Constant' or op.type == 'ConstantOfShape':
                    # The attribute of 'Constant' node is a value, needs to convert to numpy array
                    value = numpy_helper.to_array(value).copy()
                if op.type == 'Cast':
                    # The attribute of 'Cast' node is data type (represented in int), need to convert to numpy data type
                    value = mapping.TENSOR_TYPE_TO_NP_TYPE[value]
                op.attributes[key] = value

        graph_initializers = []
        for input_var in graph.inputs.values():
            # remove initilizer from graph.inputs
            if input_var.value is not None:
                graph_initializers.append(input_var.name)
        for non_input_var in graph_initializers: graph.inputs.pop(non_input_var)
        return graph

    def convert_opsets_to_str(self, opsets: Iterable) -> List[Dict[str, str]]:
        results = []
        for opset in opsets:
            results.append({'domain': opset.domain, 'version': opset.version})
        return results

    def build(self, file_path: str) -> BaseGraph:
        _rand_seed = 0 # used for name generation.
        if not is_file_exist(file_path):
            raise FileNotFoundError(f'file {file_path} does not exist, or it is a directory.')
        model_pb = onnx.load(file_path)
        opsets = model_pb.opset_import

        assert isinstance(model_pb, onnx.ModelProto), \
            f'onnx load failed, only ProtoBuffer object is expected here, while {type(model_pb)} is loaded.'
        graph_pb = model_pb.graph
        graph = BaseGraph(name=graph_pb.name, built_from=NetworkFramework.ONNX)
        graph._detail[GRAPH_OPSET_ATTRIB] = self.convert_opsets_to_str(opsets)
        graph._detail['ir_version'] = model_pb.ir_version

        onnx_import_opset = DEFAULT_OPSET_VERSION
        for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
            if opset['domain'] == DEFAULT_OPSET_DOMAIN or opset['domain'] == '':
                onnx_import_opset = opset['version']
                break

        # a temporary storage for operation's inputs and outputs
        op_inputs_dict, op_outputs_dict = {}, {}
        for node in graph_pb.node:
            op_name = node.name
            if len(op_name) == 0: # some operation do not have a name, we just generate one.
                op_name = 'generated_name_' + str(_rand_seed)
                _rand_seed += 1

            if op_name in graph.operations:
                raise KeyError(f'Duplicated operation {op_name} was found.')

            graph.operations[op_name] = Operation(
                name=op_name, op_type=node.op_type,
                attributes={item.name: helper.get_attribute_value(item) for item in node.attribute},
                opset=Opset(domain=DEFAULT_OPSET_DOMAIN, version=onnx_import_opset)
            )
            op_inputs_dict[op_name] = [var_name for var_name in node.input]
            op_outputs_dict[op_name] = [var_name for var_name in node.output]

        initializer = {}
        for item in graph_pb.initializer:
            init_name = item.name
            value = numpy_helper.to_array(item)
            initializer[init_name] = value

        inputs  = [item.name for item in graph_pb.input]
        outputs = [item.name for item in graph_pb.output]
        graph = self.build_variables(
            graph, graph_inputs=inputs, graph_outputs=outputs,
            op_inputs=op_inputs_dict, op_outputs=op_outputs_dict)
        graph = self.initialize_params(graph, initializer)
        self.de_inplace(graph)
        return self.refine_graph(graph)
