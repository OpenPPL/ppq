from typing import Any, List

import numpy as np
import torch
from ppq.core import (DataType, TargetPlatform,
                      convert_any_to_python_primary_type, ppq_warning)
from ppq.IR.quantize import DeviceSwitchOP
from ppq.IR.search import SearchableGraph
from ppq.core.data import convert_any_to_torch_tensor
from ppq.scheduler import value_tracing_pattern

from .base.command import (GraphCommand, GraphCommandType,
                           ReplaceOperationCommand, ReplaceVariableCommand)
from .base.graph import Operation, Variable
from .processer import GraphCommandProcesser


class GraphReplacer(GraphCommandProcesser):
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
            raise KeyError(f'Opeartion {op_name} is not in current graph')
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


class GraphFormatter(GraphCommandProcesser):
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
            GraphCommandType.FORMAT_SLICE
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
            return self.replace_substarction()
        if command.command_type == GraphCommandType.FORMAT_PARAMETERS:
            return self.format_parameter_variables()
        if command.command_type == GraphCommandType.FORMAT_CONSTANT_INPUT:
            return self.format_constant_input()
        if command.command_type == GraphCommandType.FORMAT_SLICE:
            return self.format_slice()

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
            此函数统一 pad 算子行为：所有 pad 算子的 pads 参数均由 operation.attribute 给出
            this func unifies behaviors of Pad op: pads paramter will always given in
            attribute
            同时当 padding mode 设置为 constant 时，pads 将存在一个用来确定 padding value 的值
            存在该值时，该函数返回 ValueError
            when the padding mode is set to constant, its constant input will be used as
            padding value
        """
        interested_ops = []
        for _, operation in self.graph.operations.items():
            if operation.type == 'Pad': interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            padding_value = operation.attributes.get('pads_value', 0)
            padding_mode = operation.attributes.get('mode', 'constant')
            if padding_mode == 'constant' and len(operation.inputs) == 3:
                pads_variable = operation.inputs[1]
                pads_constant_op = pads_variable.source_op
                padding_value = pads_constant_op.attributes['value']
                self.__delete_constant_input(operation, 1)
            if len(operation.inputs) > 1:
                # here exist a pattern: constant -> pad
                pads_variable = operation.inputs[1]
                pads_constant_op = pads_variable.source_op
                pads = pads_constant_op.attributes['value']
                self.__delete_constant_input(operation, 1)
                operation.attributes['pads'] = convert_any_to_python_primary_type(pads)
            if padding_mode == 'constant': operation.attributes['pads_value'] = padding_value

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
        """
            gather op 的参数 index 可能由 input variable 给出
            但 index 参数不可以被量化，同时后端运算需要其作为Python 原生类型
            因此将其转移到 gather op 的属性上。
            index parameter of gather op can be given by input variable,
            however, it can't be quantized, thus we transfer index parameter
            to attribute of gather op

            axis is set to 0 when it's not given
            gather op 的参数 axis 可能不存在，此时强制植入 0 作为 axis 参数
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

    def format_cast(self) -> None:
        """
            cast op 的参数 to 默认为 int，使用该函数将其封装为 ppq.core.DataType
        """
        interested_ops = []
        for _, operation in self.graph.operations.items():
            assert isinstance(operation, Operation)
            if operation.type == 'Cast': interested_ops.append(operation)
        for operation in interested_ops:
            assert isinstance(operation, Operation)
            assert 'to' in operation.attributes
            operation.attributes['to'] = DataType.convert_from_numpy(operation.attributes['to'])

    def format_int64_constant(self) -> None:
        """
            convert all int64 constants to int32, check if direct dtype cast is available
            将所有 int64 的 Constant 转换为 int32
            将检查所有 Constant value, 如果 value 范围在 int32 表示范围内则执行转换。
        """
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
        """
        部分部署平台不支持 Constant Op，在这种情况下我们使用这个 pass 把 Constant Op 的输入切换成 parameter variable 的形式
        some backend platform doesn't support Constant Op, we use this pass to replace it by forcing its value to 
        be a parameter variable
        """
        constant_ops = []
        for operation in self.graph.operations.values():
            if operation.type == 'Constant':
                assert len(operation.outputs) == 1, (
                    f"Constant Operation {operation.name} has more than 1 output, is there a network parsing error?")
                constant_ops.append(operation)

        for operation in constant_ops:
            assert isinstance(operation, Operation)
            output_var = operation.outputs[0]

            constant_value = operation.attributes['value']
            output_var.value = constant_value
            # force output variable to a parameter.
            output_var._is_parameter = True
            
            operation.outputs.clear()
            output_var.source_op = None
            self.graph.delete_operation(op_name=operation.name)

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
                    self.graph.delete_variable(var.name, force_delete=True)
                self.graph.delete_operation(op.name, force_delete=True)
        
        var_blacklist = [None]
        while len(var_blacklist) > 0:
            var_blacklist = set()
            # delete all variables that links to invalid operations:
            for var in self.graph.variables.values():
                if var.source_op is not None and var.source_op.name not in self.graph.operations:
                    var_blacklist.add(var.name)
                for op in var.dest_ops:
                    if op.name not in self.graph.operations:
                        var_blacklist.add(var.name)
    
            for var in var_blacklist:
                self.graph.delete_variable(var, force_delete=True)

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
            self.graph.delete_variable(var.name)

    def replace_substarction(self) -> None:
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
                f'has graph been manully changed? Error at Opeartion {op_name}, input_idx: {input_idx}'
        input_var = operation.inputs[input_idx]
        if input_var.source_op.type != 'Constant':
            raise ValueError(f'Trying to delete an non-const input, '\
                f'Error at Opeartion {op_name}, inputs[{input_idx}]')
        input_var.dest_ops.pop(input_var.dest_ops.index(operation))
        operation.inputs.pop(input_idx)
        if len(input_var.dest_ops) == 0:
            self.graph.delete_variable(input_var.name)
            self.graph.delete_operation(input_var.source_op.name)

    def __add_constant_input(self, op: Operation, value: torch.Tensor):
        op_name = op.name
        if op_name not in self._graph.operations:
            raise KeyError(f'Operation {op_name} not in current graph.')
        operation = self._graph.operations[op_name]
        var = Variable(name=f'{op_name}_{len(op.inputs) + 1}', value=value, is_parameter=True)
        self.graph.append_variable(var)
        var.dest_ops.append(operation)
        operation.inputs.append(var)


class GraphMerger(GraphCommandProcesser):
    """
    Graph Merger implements all graph fusion related functions.
    """
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

            assert len(bn_op.parameters) == 4, 'BatchNorm should have 4 parameters, namely alpha, beta, mean, var'
            alpha = bn_op.parameters[0].value
            beta  = bn_op.parameters[1].value
            mean  = bn_op.parameters[2].value
            var   = bn_op.parameters[3].value
            epsilon = bn_op.attributes.get('epislon', 1e-5)

            if computing_op.num_of_parameters == 1:
                w = computing_op.parameters[0].value  # no bias.
                if isinstance(w, torch.Tensor):
                    b = torch.zeros(size=w.shape[0], dtype=torch.float, device=w.device)
                if isinstance(w, np.ndarray):
                    b = np.zeros(shape=w.shape[0], dtype=np.float32)
            else:
                w, b = [var.value for var in computing_op.parameters[: 2]]  # has bias.

            if computing_op.type == 'Conv':

                # calculate new weight and bias
                scale = alpha / np.sqrt(var + epsilon)
                w = w * scale.reshape([-1, 1, 1, 1])
                b = alpha * (b - mean) / np.sqrt(var + epsilon) + beta

            elif computing_op.type == 'Gemm':

                # calculate new weight and bias
                scale = alpha / np.sqrt(var + epsilon)
                w = w * scale.reshape([-1, 1])
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
            self.graph.delete_operation(computing_op.name, cascade=True)

            # insert new
            self.graph.append_operation(merged_op)
            merged_op.inputs.extend([input_var, weight_var, bias_var])
            merged_op.outputs.extend([output_var])

            self.graph.append_variable(weight_var)
            self.graph.append_variable(bias_var)


class GraphDeviceSwitcher(GraphCommandProcesser):
    """
    Graph Device Switcher insert necessary switcher operation for
        graph split and device dispatching.
    
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
        GraphCommandProcesser ([type]): [description]
    """
    def insert_switcher(self):
        """
        Insert all necessary switchers into current graph.
            Before invoking this function, all operations must have been dispatched by a dispatcher.
        
        THIS IS AN NOT-REENTRANT FUNCTION!
        """
        def can_pass_shape(from_op: Operation, to_op: Operation) -> bool:
            if to_op.platform == TargetPlatform.SHAPE_OR_INDEX: return True
            else: return not value_tracing_pattern(from_where=from_op, to_where=to_op)
        
        soi_ops = []
        for operation in self.graph.operations.values():
            if operation.platform == TargetPlatform.SHAPE_OR_INDEX:
                soi_ops.append(operation)

        for operation in soi_ops:
            assert isinstance(operation, Operation)
            for var in operation.outputs:
                if all([can_pass_shape(operation, op) for op in var.dest_ops]): continue
                # else there is at least one opeartion needs a device converter.

                if all([not can_pass_shape(operation, op) for op in var.dest_ops]):
                    boundary_op = DeviceSwitchOP(name=var.name + '_Switcher')
                    self._graph.insert_op_on_var(inserting_op=boundary_op, var=var.name)
                    boundary_op.platform = TargetPlatform.FP32
                else:
                    for dest_op in var.dest_ops:
                        if can_pass_shape(operation, dest_op): continue
                        boundary_op = DeviceSwitchOP(name=operation.name + '_' + dest_op.name)
                        self._graph.insert_op_between_ops(inserting_op=boundary_op, up_op=operation, down_op=dest_op)
                        boundary_op.platform = TargetPlatform.FP32
            
            for var in operation.inputs:
                source_op = var.source_op
                # TODO refine here.
                if source_op is None: continue
                if source_op.platform != TargetPlatform.SHAPE_OR_INDEX and not source_op.is_soi_generator:
                    boundary_op = DeviceSwitchOP(name=source_op.name + '_' + operation.name)
                    self._graph.insert_op_between_ops(inserting_op=boundary_op, up_op=source_op, down_op=operation)
                    boundary_op.platform = TargetPlatform.SHAPE_OR_INDEX

    def remove_switcher(self):
        """
        remove all switchers from current graph.
        """
        removing_collection = []
        for operation in self.graph.operations.values():
            if operation.type == 'PPQDeviceSwitch':
                removing_collection.append(operation)

        for op in removing_collection:
            self.graph.remove_operation(removing_op=op)

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
