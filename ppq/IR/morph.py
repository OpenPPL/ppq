from typing import Any, List

import numpy as np
import torch
from ppq.core import (DataType, convert_any_to_python_primary_type,
                      convert_any_to_torch_tensor, ppq_warning)
from ppq.IR.search import SearchableGraph

from .base.command import (GraphCommand, GraphCommandType,
                           ReplaceOperationCommand, ReplaceVariableCommand,
                           TruncateGraphCommand)
from .base.graph import Operation, Variable
from .processer import GraphCommandProcessor


class GraphReplacer(GraphCommandProcessor):
    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.REPLACE_OP:
            assert isinstance(command, ReplaceOperationCommand), \
                'Use ReplaceOperationCommand instead of GraphCommand'
            return self.replace_op(command.op_name, command.replace_to)
        if command.command_type == GraphCommandType.REPLACE_VAR:
            assert isinstance(command, ReplaceVariableCommand), \
                'Use ReplaceOperationCommand instead of GraphCommand'
            return self.replace_var(command.op_name, command.replace_to)

    def replace_op(self, op_name: str, replace_to: Operation):
        if op_name not in self._graph.operations:
            raise KeyError(f'Operation {op_name} is not in current graph')
        operation = self._graph.operations[op_name]

        replace_to.inputs.clear()
        replace_to.inputs.extend(operation.inputs)
        for input_var in operation.inputs:
            dest_idx = input_var.dest_ops.index(operation)
            input_var.dest_ops[dest_idx] = replace_to

        replace_to.outputs.clear()
        replace_to.outputs.extend(operation.outputs)
        for output_var in operation.outputs:
            output_var.source_op = replace_to

        replace_to.parameters.clear()
        replace_to.parameters.extend(operation.parameters)

        self._graph.operations[op_name] = replace_to

    def replace_var(self, var_name: str, replace_to: Variable):
        if var_name not in self._graph.variables:
            raise KeyError(f'Variable {var_name} is not in current graph')
        variable = self._graph.variables[var_name]

        replace_to.dest_ops.clear()
        replace_to.dest_ops.extend(variable.dest_ops)
        for dest_op in replace_to.dest_ops:
            dest_idx = dest_op.inputs.index(variable)
            dest_op.inputs[dest_idx] = replace_to

        replace_to.source_op = variable.source_op
        if variable.source_op is not None:
            source_idx = variable.source_op.outputs.index(variable)
            variable.source_op.outputs[source_idx] = replace_to

        self._graph.variables[var_name] = replace_to
        if var_name in self._graph.inputs:
            self._graph.inputs[var_name] = replace_to
        if var_name in self._graph.outputs:
            self._graph.outputs[var_name] = replace_to

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.REPLACE_OP,
            GraphCommandType.REPLACE_VAR,
        ]


class GraphFormatter(GraphCommandProcessor):
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.FORMAT_CLIP,
            GraphCommandType.FORMAT_PAD,
            GraphCommandType.FORMAT_GATHER,
            GraphCommandType.FORMAT_CAST,
            GraphCommandType.FORMAT_INT64_CONSTANT,
            GraphCommandType.DELETE_ISOLATED,
            GraphCommandType.REPLACE_SUB,
            GraphCommandType.FORMAT_PARAMETERS,
            GraphCommandType.FORMAT_CONSTANT_INPUT,
            GraphCommandType.FORMAT_SLICE,
            GraphCommandType.TRUNCATE_ON_VAR,
            GraphCommandType.FORMAT_RESIZE,
            GraphCommandType.FORMAT_SNG_BN
        ]

    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.FORMAT_CLIP:
            return self.format_clip()
        if command.command_type == GraphCommandType.FORMAT_PAD:
            return self.format_pad()
        if command.command_type == GraphCommandType.FORMAT_GATHER:
            return self.format_gather()
        if command.command_type == GraphCommandType.FORMAT_CAST:
            return self.format_cast()
        if command.command_type == GraphCommandType.DELETE_ISOLATED:
            return self.delete_isolated()
        if command.command_type == GraphCommandType.FORMAT_INT64_CONSTANT:
            return self.format_int64_constant()
        if command.command_type == GraphCommandType.REPLACE_SUB:
            return self.replace_substraction()
        if command.command_type == GraphCommandType.FORMAT_PARAMETERS:
            return self.format_parameter_variables()
        if command.command_type == GraphCommandType.FORMAT_CONSTANT_INPUT:
            return self.format_constant_input()
        if command.command_type == GraphCommandType.FORMAT_SLICE:
            return self.format_slice()
        if command.command_type == GraphCommandType.FORMAT_RESIZE:
            return self.format_resize()
        if command.command_type == GraphCommandType.FORMAT_SNG_BN:
            return self.format_sng_bn()
        if command.command_type == GraphCommandType.TRUNCATE_ON_VAR:
            assert isinstance(command, TruncateGraphCommand), f'Use TruncateGraphCommand here.'
            return self.truncate_on_var(command.var, command.mark_as_output)

    def format_slice(self) -> None:
        """
            Slice: opset1 格式跟其他的不太一样，这个 pass 将 opset1 的 slice 强行转换为 opset 11
        """
        interested_ops = []
        for operation in self.graph.operations.values():
            if operation.type == 'Slice':
                if 'starts' in operation.attributes:
                    assert 'starts' in operation.attributes and 'ends' in operation.attributes, (
                        f'Invalid Slice Operation Format, Slice operation is expected to have axes, '
                        'starts and ends attributes with opset 1, '
                        f'however your operation {operation.name}, do not have completed attributes')
                    interested_ops.append(operation)

        for slice in interested_ops:
            assert isinstance(slice, Operation)
            axes   = slice.attributes.get('axes', None)
            starts = slice.attributes['starts']
            ends   = slice.attributes['ends']
            if axes == None: axes = [_ for _ in range(starts)]

            slice.attributes.pop('starts')
            slice.attributes.pop('ends')
            if 'axes' in slice.attributes: slice.attributes.pop('axes')
            self.__add_constant_input(slice, convert_any_to_torch_tensor(starts))
            self.__add_constant_input(slice, convert_any_to_torch_tensor(ends))
            self.__add_constant_input(slice, convert_any_to_torch_tensor(axes))

    def format_pad(self) -> None:
        """
            对于不同的模型格式, pad 算子将有两种不同的输入格式：
            for different models, possibly Pad op has the following input formats
                1. pads 参数由第二个输入变量给出
                   pads parameter is given by the second input variable
                2. pads 参数被放置于 operation.attribute 中
                   pads parameter is set in attribute
            此函数统一 pad 算子行为：所有 pad 算子的 pads 参数均由第二个输入给出
        """
        for op in self.graph.operations.values():
            if op.type == 'Pad' and 'pads' in op.attributes:
                self.graph.create_link_with_op(
                    variable=self.graph.create_variable(
                        value=torch.tensor(op.attributes['pads']), is_parameter=True),
                    upstream_op=None, downstream_op=op)
                op.attributes.clear()

    def format_resize(self) -> None:
        """
            升级 opset 10 的 resize 到 opset 11
        """
        for op in self.graph.operations.values():
            if op.type == 'Resize' and len(op.inputs) == 2:
                self.graph.create_link_with_op(
                    variable=self.graph.create_variable(value=None, is_parameter=False),
                    upstream_op=None, downstream_op=op)
                op.inputs[1], op.inputs[2] = op.inputs[2], op.inputs[1]

    def format_clip(self) -> None:
        """
            对于不同的模型格式, clip 算子将有两种不同的输入格式：
            for different models, possibly clip op has the following input formats
                1. min, max 参数由 第二、第三个输入变量给出
                   min, max parameter will be given by the second and third input variable
                2. min, max 参数由 attribute 给出
                   min, max parameter will be given by the attribute
            此函数统一 clip 算子行为：所有 clip 算子的 min, max 参数第二第三个变量给出
            this func unifies behaviors of clip op: min, max parameter will be given by input vars
            针对可能存在的 min, max 为空的情况，将其直接置为 2 << 30（保证处理后非空）

            当 min, max 参数由 第二、第三个输入变量给出时，其中一个为空时直接返回 ValueError
            ValueError will be raised when any of min, max parameters is null
        """

        interested_ops = []
        for _, operation in self.graph.operations.items():
            if operation.type == 'Clip' and ('min' in operation.attributes or 'max' in operation.attributes):
                interested_ops.append(operation)
        for op in interested_ops:
            assert isinstance(op, Operation)
            min = op.attributes.get('min', - 2 << 30)
            max = op.attributes.get('max', + 2 << 30)
            min_var = Variable(name=op.name + '_min', value=min, is_parameter=True, dest_ops=[op])
            max_var = Variable(name=op.name + '_max', value=max, is_parameter=True, dest_ops=[op])
            self.graph.append_variable(min_var)
            self.graph.append_variable(max_var)
            op.inputs.append(min_var)
            op.inputs.append(max_var)
            if 'min' in op.attributes: op.attributes.pop('min')
            if 'max' in op.attributes: op.attributes.pop('max')

    def format_gather(self) -> None:
        """gather op 的参数 index 可能由 input variable 给出 但 index
        参数不可以被量化，同时后端运算需要其作为Python 原生类型 因此将其转移到 gather op 的属性上。 index parameter
        of gather op can be given by input variable, however, it can't be
        quantized, thus we transfer index parameter to attribute of gather op.

        axis is set to 0 when it's not given gather op 的参数 axis 可能不存在，此时强制植入 0
        作为 axis 参数
        """
        interested_ops = []
        for _, operation in self.graph.operations.items():
            if operation.type == 'Gather': interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            if len(operation.inputs) == 2:
                index_op = operation.inputs[1].source_op
                if index_op.type == 'Constant':
                    index = index_op.attributes['value']
                    self.__delete_constant_input(operation, 1)
                    operation.attributes['gather_index'] = convert_any_to_python_primary_type(index)
            if 'axis' not in operation.attributes:
                operation.attributes['axis'] = 0

            if 'indices' in operation.attributes:
                operation.attributes['gather_index'] = operation.attributes['indices']
                operation.attributes.pop('indices')
                
    def format_sng_bn(self) -> None:
        """
        batchnormal: 这个算子独立出现的时候（前面没有Conv），我们考虑把它合进一个1x1的conv
        """
        interested_ops = []
        for operation in self.graph.operations.values():
            if operation.type == "BatchNormalization":
                upstream_ops = self.graph.get_upstream_operations(operation)  # 假设bn上游只有一个op或者没有op

                if len(upstream_ops) == 0:
                    interested_ops.append(operation)
                    continue
                assert len(upstream_ops) == 1, f"BN op({operation.name}) should have only one input"
                if upstream_ops[0].type not in ["Conv", "Gemm", "ConvTranspose"]:
                    interested_ops.append(operation)
                else:
                    down_ops = self.graph.get_downstream_operations(upstream_ops[0])
                    if len(down_ops) != 1:
                        interested_ops.append(operation)

        for bn_op in interested_ops:
            assert isinstance(bn_op, Operation)
            assert len(bn_op.parameters) == 4, "BatchNorm should have 4 parameters, namely alpha, beta, mean, var"
            alpha = bn_op.parameters[0].value
            beta  = bn_op.parameters[1].value
            mean  = bn_op.parameters[2].value
            var   = bn_op.parameters[3].value
            epsilon = bn_op.attributes.get("epsilon", 1e-5)

            input_var = bn_op.inputs[0]
            input_channel = alpha.shape[0]
            bn_out_var = bn_op.outputs[0]
            w = np.ones((input_channel, 1, 1, 1), dtype=np.float32)  # 1x1卷积 group卷积
            b = np.zeros(
                shape=w.shape[1] * input_channel,
                dtype=np.float32,
            )
            scale = alpha / np.sqrt(var + epsilon)
            w = w * scale.reshape([-1, 1, 1, 1])
            b = alpha * (b - mean) / np.sqrt(var + epsilon) + beta
            conv_op = Operation(
                name=bn_op.name + "_conv",
                op_type="Conv",
                attributes=dict(
                    kernel_shape=[1, 1], strides=[1, 1], dilations=[1, 1], pads=[0, 0, 0, 0], group=input_channel
                ),
                inputs=[input_var],
                outputs=[bn_out_var],
                platform=bn_op.platform,
            )
            self.graph.append_operation(conv_op)

            w_var = self.graph.create_variable(
                name=conv_op.name + "_1x1_weight", value=w, is_parameter=True, dest_ops=[conv_op]
            )
            b_var = self.graph.create_variable(
                name=conv_op.name + "_1x1_bias", value=b, is_parameter=True, dest_ops=[conv_op]
            )
            conv_op.inputs.extend([w_var, b_var])
            self.graph.variables[w_var.name] = w_var
            self.graph.variables[b_var.name] = b_var

            if bn_op.outputs[0].name in self.graph.outputs:
                del self.graph.outputs[bn_op.outputs[0].name]
                self.graph.outputs.update({conv_op.outputs[0].name: conv_op.outputs[0]})

            self.graph.remove_operation(bn_op)

            bn_out_var.source_op = conv_op
            input_var.dest_ops.append(conv_op)
        
    def format_cast(self) -> None:
        """cast op 的参数 to 默认为 int，使用该函数将其封装为 ppq.core.DataType."""
        interested_ops = []
        for _, operation in self.graph.operations.items():
            assert isinstance(operation, Operation)
            if operation.type == 'Cast': interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            assert 'to' in operation.attributes
            operation.attributes['to'] = DataType.convert_from_numpy(operation.attributes['to'])

    def format_int64_constant(self) -> None:
        """convert all int64 constants to int32, check if direct dtype cast is
        available 将所有 int64 的 Constant 转换为 int32 将检查所有 Constant value, 如果 value
        范围在 int32 表示范围内则执行转换。"""
        for operation in self.graph.operations.values():
            if operation.type == 'Constant':
                assert 'value' in operation.attributes
                value = operation.attributes['value']

                assert isinstance(value, torch.Tensor)
                if value.dtype != torch.int64: continue

                pvalue = convert_any_to_python_primary_type(value)
                check = [0xFFFFFFFF > v >= -0xFFFFFFFF for v in pvalue]

                if all(check): value = value.int()

    def format_constant_input(self) -> None:
        """部分部署平台不支持 Constant Op，在这种情况下我们使用这个 pass 把 Constant Op 的输入切换成
        parameter variable 的形式 some backend platform doesn't support Constant
        Op, we use this pass to replace it by forcing its value to be a
        parameter variable."""
        constant_ops = []
        for operation in self.graph.operations.values():
            if operation.type == 'Constant':
                assert len(operation.outputs) == 1, (
                    f'Constant Operation {operation.name} has more than 1 output, is there a network parsing error?')
                constant_ops.append(operation)

        for operation in constant_ops:
            assert isinstance(operation, Operation)
            constant_value = operation.attributes['value']
            output_var = operation.outputs[0]
            output_var._is_parameter = True
            output_var.value = constant_value
            self.graph.remove_operation(removing_op=operation)

    def truncate_on_var(self, var: Variable, mark_as_output: bool):
        """从一个指定位置将图截断.

        Args:
            var (Variable): _description_
            mark_as_output (bool): _description_

        Raises:
            TypeError: _description_
            KeyError: _description_
        """
        graph = self.graph
        if not isinstance(var, Variable):
            raise TypeError(f'Except variable instance here, however {type(var)} was given.')
        if var.name not in graph.variables:
            raise KeyError(f'Can not find vairiable {var.name} in current graph')

        mark_to_delete, delete_queue, didx = set(), [], 0
        delete_queue.extend(var.dest_ops)
        while didx < len(delete_queue):
            first_op = delete_queue[didx]
            if first_op not in mark_to_delete:
                mark_to_delete.add(first_op)
                delete_queue.extend(graph.get_downstream_operations(first_op))
            didx += 1

        for operation in mark_to_delete:
            graph.remove_operation(operation)

        if mark_as_output:
            graph.mark_variable_as_graph_output(var)

        self.delete_isolated()

    def delete_isolated(self):
        blacklist = [None]
        while len(blacklist) > 0:
            blacklist = []
            # delete all operations which are not links to a valid graph output
            for op in self.graph.operations.values():
                if len(self.graph.get_downstream_operations(op)) == 0:
                    output_names = [var.name for var in op.outputs]
                    if all([name not in self.graph.outputs for name in output_names]):
                        blacklist.append(op)

            for op in blacklist:
                for var in op.outputs:
                    self.graph.remove_variable(var)
                self.graph.remove_operation(op)

        var_blacklist = [None]
        while len(var_blacklist) > 0:
            var_blacklist = set()
            # delete all variables that links to invalid operations:
            for var in self.graph.variables.values():
                # 删除无根无输出的变量
                if var.source_op is None and len(var.dest_ops) == 0:
                    var_blacklist.add(var)

                # 删除根节点不在图中的变量
                if var.source_op is not None and var.source_op.name not in self.graph.operations:
                    var_blacklist.add(var)

                # 删除连接到未知节点的变量
                for op in var.dest_ops:
                    if op.name not in self.graph.operations:
                        var_blacklist.add(var)

                # 删除孤立变量
                if var.source_op is None and var.name not in self.graph.inputs:
                    if len(var.dest_ops) == 0: var_blacklist.add(var)

                # 没有输出的不能删...会影响算子输出顺序...

            for var in var_blacklist:
                self.graph.remove_variable(var)

    def format_parameter_variables(self) -> None:
        vars = []
        for var in self.graph.variables.values():
            if var.is_parameter and len(var.dest_ops) > 1:
                # found parameter with multiple destination operations
                # split parameter variable
                vars.append(var)

        for var in vars:
            assert isinstance(var, Variable)
            for idx, dest_op in enumerate(var.dest_ops.copy()):
                # create variables
                sub_var = Variable(
                    name=var.name + '_' + str(idx),
                    value=var.value, is_parameter=True,
                    dest_ops=[dest_op], source_op=None)
                self.graph.append_variable(sub_var)

                # replace original variable with splited one.
                dest_op.inputs[dest_op.inputs.index(var)] = sub_var
                var.dest_ops.remove(dest_op)

            # pop variable from graph
            self.graph.remove_variable(var)

    def replace_substraction(self) -> None:
        substractions = []
        for operation in self.graph.operations.values():
            if operation.type == 'Sub':
                substractions.append(operation)

        for operation in substractions:
            assert isinstance(operation, Operation)
            subtractor = operation.inputs[-1].source_op
            substractor_var = operation.inputs[-1]

            # create a neg operation
            neg_op = Operation(name=subtractor.name + '_neg', op_type='Neg', attributes={})
            self.graph.append_operation(neg_op)

            # create related variables
            neg_var = Variable(name=subtractor.name + '_neg_1', dest_ops=[operation], source_op=neg_op)

            # link var to op
            neg_op.inputs.append(substractor_var)
            neg_op.outputs.append(neg_var)

            operation.inputs.remove(substractor_var)
            operation.inputs.append(neg_var)

            substractor_var.dest_ops.remove(operation)
            substractor_var.dest_ops.append(neg_op)

            # add var to graph
            self.graph.append_variable(neg_var)

            # replace sub to add
            operation._type = 'Add'

    def __delete_constant_input(self, op: Operation, input_idx: int):
        op_name = op.name
        if op_name not in self._graph.operations:
            raise KeyError(f'Operation {op_name} not in current graph.')
        operation = self._graph.operations[op_name]
        assert input_idx < len(operation.inputs), 'Trying to delete an out-of-range input variable, '\
                f'has graph been manually changed? Error at Operation {op_name}, input_idx: {input_idx}'
        input_var = operation.inputs[input_idx]
        if input_var.source_op.type != 'Constant':
            raise ValueError(f'Trying to delete an non-const input, '\
                f'Error at Operation {op_name}, inputs[{input_idx}]')
        input_var.dest_ops.pop(input_var.dest_ops.index(operation))
        operation.inputs.pop(input_idx)

        if len(input_var.dest_ops) == 0:
            self.graph.remove_operation(input_var.source_op)
            self.graph.remove_variable(input_var)

    def __add_constant_input(self, op: Operation, value: torch.Tensor):
        op_name = op.name
        if op_name not in self._graph.operations:
            raise KeyError(f'Operation {op_name} not in current graph.')
        operation = self._graph.operations[op_name]
        var = Variable(name=f'{op_name}_{len(op.inputs) + 1}', value=value, is_parameter=True)
        self.graph.append_variable(var)
        var.dest_ops.append(operation)
        operation.inputs.append(var)


class GraphMerger(GraphCommandProcessor):
    """Graph Merger implements all graph fusion related functions."""
    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            # add more extensions in the future
            GraphCommandType.FUSE_BN
        ]

    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.FUSE_BN:
            return self.fuse_bn()
    
    def fuse_bn(self):
        search_engine = SearchableGraph(graph=self.graph)
        paths = search_engine.path_matching(
            sp_expr=lambda x: x.type in {'Conv', 'Gemm', 'ConvTranspose'},
            rp_expr=lambda x, y: False,
            ep_expr=lambda x: x.type == 'BatchNormalization',
            direction='down')

        for path in paths:
            path = path.tolist()
            assert len(path) == 2, ('Oops seems we got something unexpected.')

            computing_op, bn_op = path
            assert isinstance(computing_op, Operation) and isinstance(bn_op, Operation)

            if (len(self.graph.get_downstream_operations(computing_op)) != 1 or
                len(self.graph.get_upstream_operations(bn_op)) != 1):
                ppq_warning(f'PPQ can not merge operation {computing_op.name} and {bn_op.name}, '
                            'this is not suppose to happen with your network, '
                            'network with batchnorm inside might not be able to quantize and deploy.')
                continue

            assert len(bn_op.parameters) == 4, 'BatchNorm should have 4 parameters, namely alpha, beta, mean, var'
            alpha = bn_op.parameters[0].value
            beta  = bn_op.parameters[1].value
            mean  = bn_op.parameters[2].value
            var   = bn_op.parameters[3].value
            epsilon = bn_op.attributes.get('epsilon', 1e-5)

            if computing_op.num_of_parameters == 1:
                w = computing_op.parameters[0].value  # no bias.
                assert isinstance(w, np.ndarray), 'values of parameters are assumed numpy arrays'
                if computing_op.type == 'ConvTranspose':
                    b = np.zeros(shape=w.shape[1] * computing_op.attributes.get('group', 1), dtype=np.float32)
                elif computing_op.type == 'Gemm' and computing_op.attributes.get('transB', 0) == 0:
                    b = np.zeros(shape=w.shape[1], dtype=np.float32)
                else:
                    b = np.zeros(shape=w.shape[0], dtype=np.float32)
            else:
                w, b = [var.value for var in computing_op.parameters[: 2]]  # has bias.

            if computing_op.type == 'Conv':

                # calculate new weight and bias
                scale = alpha / np.sqrt(var + epsilon)
                w = w * scale.reshape([-1] + [1] * (w.ndim - 1))
                b = alpha * (b - mean) / np.sqrt(var + epsilon) + beta

            elif computing_op.type == 'Gemm':

                # calculate new weight and bias
                scale = alpha / np.sqrt(var + epsilon)
                if computing_op.attributes.get('transB', 0):
                    w = w * scale.reshape([-1, 1])
                else:
                    w = w * scale.reshape([1, -1])
                b = alpha * (b - mean) / np.sqrt(var + epsilon) + beta

            elif computing_op.type == 'ConvTranspose':

                scale = alpha / np.sqrt(var + epsilon)
                group = computing_op.attributes.get('group', 1)
                scale = scale.reshape([group, 1, -1, 1, 1])
                w = w.reshape([group, -1, w.shape[1], w.shape[2], w.shape[3]]) * scale
                w = w.reshape([w.shape[0] * w.shape[1], w.shape[2], w.shape[3], w.shape[4]])
                b = alpha * (b - mean) / np.sqrt(var + epsilon) + beta
            else:
                raise TypeError(
                    f'Unexpected op type {computing_op.type}. '
                    f'Can not merge {computing_op.name} with {bn_op.name}')

            # create new op and variable
            merged_op  = Operation(computing_op.name, op_type=computing_op.type,
                                   attributes=computing_op.attributes.copy())
            weight_var = Variable(computing_op.name + '_weight', w, True, [merged_op])
            bias_var   = Variable(computing_op.name + '_bias', b, True, [merged_op])

            # replace & dirty work
            input_var  = computing_op.inputs[0]
            output_var = bn_op.outputs[0]

            input_var.dest_ops.remove(computing_op)
            input_var.dest_ops.append(merged_op)

            output_var.source_op = merged_op

            # delete old operations
            computing_op.inputs.pop(0)
            bn_op.outputs.clear()
            self.graph.remove_operation(computing_op)

            # insert new
            self.graph.append_operation(merged_op)
            merged_op.inputs.extend([input_var, weight_var, bias_var])
            merged_op.outputs.extend([output_var])

            self.graph.append_variable(weight_var)
            self.graph.append_variable(bias_var)


    def fuse_gemm(self):
        """Fuse MatMul + add into a singal Gemm
            Single Matmul will be replaced with Gemm

        Returns:
            _type_: _description_
        """

        def _is_replaceable(op: Operation) -> bool:
            if op.inputs[0].is_parameter == False and op.inputs[1].is_parameter == False:
                return False
            else:
                return True

        search_engine = SearchableGraph(graph=self.graph)
        patterns = search_engine.pattern_matching(patterns=["MatMul", "Add"], edges=[[0, 1]], exclusive=True)
        for pattern in patterns:
            matmul, add = pattern

            if _is_replaceable(matmul) == False:
                continue

            matmul.type = "Gemm"

            matmul_out = matmul.outputs[0]
            add_out = add.outputs[0]

            if matmul.inputs[0].is_parameter:
                temp = matmul.inputs[0]
                matmul.inputs[0] = matmul.inputs[1]
                matmul.inputs[1] = temp

            assert len(add.inputs) == 2, "Oops, seems we got some problem here."
            var1, var2 = add.inputs
            bias_var = None

            if var1.source_op == matmul and var2.is_parameter:
                bias_var = var2

            if var2.source_op == matmul and var1.is_parameter:
                bias_var = var1

            # can not find a valid bias, just skip add.
            if bias_var is None:
                continue

            if len(bias_var.value.shape) == 1:
                if bias_var.value.shape[0] == matmul.parameters[0].value.shape[-1]:
                    matmul.attributes["transB"] = 1
                    weight_val = matmul.parameters[0].value

                    matmul.parameters[0].value = weight_val.transpose(-1, -2)

                    bias_var.dest_ops.clear()
                    add.inputs.remove(bias_var)

                    # remove bias add, move bias to matmul
                    self.graph.remove_operation(add)
                    self.graph.create_link_with_op(variable=bias_var, upstream_op=None, downstream_op=matmul)
                    self.graph.create_link_with_var(upstream_variable=matmul_out, downstream_variable=add_out)
                elif bias_var.value.shape[0] == matmul.parameters[0].value.shape[-2]:

                    bias_var.dest_ops.clear()
                    add.inputs.remove(bias_var)
                    # remove bias add, move bias to matmul
                    self.graph.remove_operation(add)
                    self.graph.create_link_with_op(variable=bias_var, upstream_op=None, downstream_op=matmul)
                    self.graph.create_link_with_var(upstream_variable=matmul_out, downstream_variable=add_out)

        # process single gemm
        for op in self.graph.operations.values():
            if op.type == "MatMul":
                if _is_replaceable(op) == False:
                    continue
                op.type = "Gemm"


    def fuse_layernorm(self, exclusive_search: bool = False):
        """Fuse Layernormalization with pattern matching."""
        
        def _fuse(rm1: Operation, rm2: Operation, 
                  eps: Operation, scale: torch.Tensor, 
                  bias: torch.Tensor, layernorm_input_var: Variable, 
                  layernorm_output_var: Variable) -> Operation:

            if rm2.type == rm1.type == 'ReduceMean':
                if 'axes' not in rm1.attributes: return None
                if 'axes' not in rm2.attributes: return None
                if rm1.attributes['axes'] != rm2.attributes['axes']: return None
                layernorm_axis = rm1.attributes['axes']
                if isinstance(layernorm_axis, list): 
                    if len(layernorm_axis) != 1: return None
                    layernorm_axis = layernorm_axis[0]
                if not isinstance(layernorm_axis, int): return None
            else: layernorm_axis = -1

            if not eps.inputs[-1].is_parameter: return None
            value = eps.inputs[-1].value
            value = convert_any_to_torch_tensor(value).cpu()
            if value.numel() != 1: return None
            layernorm_eps = value.item()

            layernorm_output_var.source_op.outputs.clear()
            layernorm = self.graph.create_operation(
                op_type = 'LayerNormalization',
                attributes = {'axis': layernorm_axis, 'epsilon': layernorm_eps, 'stash_type': 0},
                inputs = [layernorm_input_var, self.graph.create_variable(value=scale, is_parameter=True)],
                outputs = [layernorm_output_var])

            if bias is not None:
                self.graph.create_link_with_op(
                    variable=self.graph.create_variable(value=bias, is_parameter=True), 
                    upstream_op=None, downstream_op=layernorm)
            return layernorm

        search_engine = SearchableGraph(graph=self.graph)
        
        # pattern 1:
        #                                 ---     ---     ---      ---        ---       ---    ---    --
        #                               |                                                              |
        # ***(0) --- ReduceMean(1) --- Sub(2) --- Pow(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Div(7) --- Mul(8) --- (Add)(9)
        #      |                     |
        #       ---   ---   ---   ---       
        matches = search_engine.pattern_matching(
            patterns=[lambda x: True, lambda x: x.type in {'ReduceMean', 'GlobalAveragePool'}, 'Sub', 'Pow', 
                      lambda x: x.type in {'ReduceMean', 'GlobalAveragePool'}, 'Add', 'Sqrt', 'Div', 'Mul'],
            edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]], exclusive=exclusive_search)

        for _, rm1, sub, pow, rm2, add, sqrt, div, mul in matches:
            layernorm_ops = [rm1, sub, pow, rm2, add, sqrt, div, mul]

            layernorm_scale = mul.inputs[-1].value
            layernorm_output_var = div.outputs[0]
            layernorm_input_var = sub.inputs[0]

            # bias check
            layernorm_bias = None
            next_op = self.graph.get_downstream_operations(mul)
            if len(next_op) == 1 and (next_op[0].type == 'Add'):
                bias_op = next_op[0]
                if bias_op.inputs[-1].is_parameter:
                    layernorm_bias = bias_op.inputs[-1].value
                    layernorm_output_var = bias_op.outputs[0]
                    layernorm_ops.append(bias_op)

            layernorm = _fuse(rm1=rm1, rm2=rm2, eps=add, scale=layernorm_scale, 
                bias=layernorm_bias, layernorm_input_var=layernorm_input_var, 
                layernorm_output_var=layernorm_output_var)
            
            if layernorm is not None:
                # delete merged ops
                for op in layernorm_ops:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var != layernorm_input_var and var != layernorm_output_var:
                            self.graph.remove_variable(var)
                    self.graph.remove_operation(op)

        # pattern 2:
        #  ---   ---  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  -- Mul(11)   ---   ---   Add(12) ---
        #  ^                             |                                                                                ^                     ^
        #  |                             v                                                                                |                     |
        # ***(0) --- ReduceMean(1) --- Sub(2) --- Mul(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Reciprocal(7) --- Mul(8) --- Mul(9) --- Sub(10)
        #             |                                                                                                             |
        #             v                                                                                                             ^
        #             --   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  ---  --- --
        matches = search_engine.pattern_matching(
            patterns=[lambda x: True, lambda x: x.type in {'ReduceMean', 'GlobalAveragePool'}, 'Sub', 
                      'Mul', lambda x: x.type in {'ReduceMean', 'GlobalAveragePool'}, 'Add', 'Sqrt', 
                      'Reciprocal', 'Mul', 'Mul', 
                      'Sub', 'Mul', 'Add'],
            edges=[[0, 1], [0, 2], [0, 11], [1, 2], [1, 9], [2, 3], [3, 4], 
                   [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], 
                   [8, 11], [9, 10], [11, 12], [10, 12]], exclusive=exclusive_search)

        for _, rm1, sub1, mul, rm2, add1, sqrt, recipro, mul2, mul3, sub2, mul4, add2 in matches:
            layernorm_ops = [rm1, sub1, mul, rm2, add1, sqrt, recipro, mul2, mul3, sub2, mul4, add2]

            # mul check
            if not mul2.inputs[-1].is_parameter: continue
            layernorm_scale      = mul2.inputs[-1].value
            layernorm_output_var = add2.outputs[0]
            layernorm_input_var  = sub1.inputs[0]
            layernorm_bias       = sub2.inputs[0].value

            layernorm = _fuse(rm1=rm1, rm2=rm2, eps=add1, scale=layernorm_scale, 
                bias=layernorm_bias, layernorm_input_var=layernorm_input_var, 
                layernorm_output_var=layernorm_output_var)

            if layernorm is not None:
                # delete merged ops
                for op in layernorm_ops:
                    assert isinstance(op, Operation)
                    for var in op.inputs + op.outputs:
                        if var != layernorm_input_var and var != layernorm_output_var:
                            self.graph.remove_variable(var)
                    self.graph.remove_operation(op)


class GraphDecomposer(GraphCommandProcessor):
    """Since PPQ 0.6.4, GraphDecomposer is introduced to split some complex
    operations For example, Gemm can be split with MatMul with Bias add.

    Gemm
    General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

    A' = transpose(A) if transA else A

    B' = transpose(B) if transB else B

    Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M), 
        input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), 
        and output tensor Y has shape (M, N). A will be transposed before doing the computation if attribute transA is non-zero, 
        same for B and transB. 
    
    This operator supports unidirectional broadcasting (tensor C should be unidirectional broadcastable to tensor A * B); 
        for more details please check the doc. This operator has optional inputs/outputs. 
        
    See the doc for more details about the representation of optional arguments. 
    An empty string may be used in the place of an actual argument's name to indicate a missing argument. 
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

    Attributes
        alpha : float (default is 1.0)
        Scalar multiplier for the product of input tensors A * B.
    
        beta : float (default is 1.0)
        Scalar multiplier for input tensor C.
    
        transA : int (default is 0)
        Whether A should be transposed
    
        transB : int (default is 0)
        Whether B should be transposed
    """

    def process(self, command: GraphCommand) -> Any:
        return super().process(command)

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return super()._acceptable_command_types

    def decompose_gemm(self):
        graph = self.graph
        interested_ops = []
        for operation in graph.operations.values():
            if operation.type == 'Gemm':
                interested_ops.append(operation)

        for op in interested_ops:
            assert isinstance(op, Operation)
            output_var = op.outputs[0]

            if op.num_of_input == 3:
                bias_add  = graph.create_operation(op_type='Add', platform=op.platform)
                bias_var  = op.inputs[-1]

                graph.create_link_with_op(
                    variable=graph.create_variable(),
                    upstream_op=op, downstream_op=bias_add)

                graph.create_link_with_op(
                    variable=graph.create_variable(
                        value=bias_var.value * op.attributes.get('beta', 1), is_parameter=True),
                    upstream_op=None, downstream_op=bias_add)

                graph.remove_variable(bias_var)
                output_var.source_op = bias_add
                bias_add.outputs.append(output_var)
                op.outputs.remove(output_var)

            if op.attributes.get('transA', 0) == 1:
                raise ValueError(f'Can not process with operation {op.name}, transA=1 is not allowed.')
            if op.attributes.get('alpha', 1) != 1:
                op.parameters[0].value *= op.attributes.get('alpha')
            if op.attributes.get('transB', 0) == 1:
                op.inputs[1].value = op.inputs[1].value.permute(1, 0)

            op.type = 'MatMul'
            op.attributes.clear()

    def decompose_gru(self):
        pass

class GraphDeviceSwitcher(GraphCommandProcessor):
    """Graph Device Switcher insert necessary switcher operation for graph
    split and device dispatching.

    See also ppq scheduler for more information.

    All SOI operations are supposed to be executed on cpu.
        while other operations are supposed to be executed on cuda.
        Therefore switching operation will be inserted between SOI operations and fp32(quant) operations.
        to transfer cuda tensor to cpu tensor, vice versa.

    However some operations receive SOI input(cpu tensor) naturally, such as reshape, slice, etc.
    PPQ uses a tracing function for judging whether it is necessary to insert a
        switcher between operations like that.

    Before invoking this class, all operations must have been dispatched by a dispatcher.

    Args:
        GraphCommandProcessor ([type]): [description]
    """
    def insert_switcher(self):
        ppq_warning('This function has been removed from PPQ Since 0.6.6')

    def remove_switcher(self):
        ppq_warning('This function has been removed from PPQ Since 0.6.6')

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.INSERT_SWITCHER,
            GraphCommandType.REMOVE_SWITCHER
        ]

    def process(self, command: GraphCommand) -> Any:
        if command.command_type == GraphCommandType.INSERT_SWITCHER:
            return self.insert_switcher()
        if command.command_type == GraphCommandType.REMOVE_SWITCHER:
            return self.remove_switcher()