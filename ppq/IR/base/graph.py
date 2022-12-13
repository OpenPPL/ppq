from abc import abstractmethod
from collections import deque
from typing import Any, Dict, List, Text, Union

import torch
import numpy as np
from ppq.core import (LINEAR_ACTIVATIONS, DataType, NetworkFramework,
                      Serializable, SingletonMeta, TargetPlatform, TensorMeta,
                      convert_any_to_torch_tensor, ppq_warning)

from .opdef import (DEFAULT_SOCKET_CREATOR, DEFAULT_SOCKET_TABLE,
                    OperationBase, Opset, OpSocket)


class Variable(Serializable):
    def __init__(self, name: str, value: Any = None, is_parameter: bool = False,
                 dest_ops: List[OperationBase] = None, source_op: OperationBase = None,
                 shape: List[int] = None, dtype: DataType= DataType.FP32) -> None:
        super().__init__()
        self._name = name
        self._value = None if value is None else value
        self._dest_ops = [] if dest_ops is None else dest_ops
        self._source_op = None if source_op is None else source_op
        self._is_parameter = is_parameter
        self._shape = shape
        self._dtype = dtype

    @ property
    def is_parameter(self) -> bool:
        return self._is_parameter

    @ is_parameter.setter
    def is_parameter(self, value: bool):
        self._is_parameter = value

    @ property
    def name(self) -> str:
        return self._name

    @ property
    def value(self) -> torch.Tensor:
        return self._value

    @ value.setter
    def value(self, value):
        self._value = value

    @ property
    def dest_ops(self) -> List[OperationBase]:
        return self._dest_ops

    @ property
    def source_op(self) -> OperationBase:
        return self._source_op

    @ source_op.setter
    def source_op(self, source_op: OperationBase) -> OperationBase:
        self._source_op = source_op

    @ property
    def dest_idx(self) -> List[int]:
        _dest_idx = []
        for op in self.dest_ops:
            _dest_idx.append(op.inputs.index(self))
        return _dest_idx

    @ property
    def src_idx(self) -> int:
        if self.source_op is not None:
            return self.source_op.outputs.index(self)
        else: return None

    @ property
    def meta(self) -> TensorMeta:
        raise Exception('PPQ Variable.meta has been removed since 0.6.6, use Variable.shape, Variable.dtype instead.')

    def __hash__(self) -> int:
        return self._name.__hash__()

    def __str__(self) -> str:
        return f'Variable ({self._name})'

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['_dest_ops'] = [op.name for op in self.dest_ops]
        state['_source_op'] = self.source_op.name if self.source_op is not None else None
        return state

    @ property
    def shape(self) -> List[int]:
        """ Return tensor shape of this variable
        It is modifiable when current variable is not a paramter.
        """
        if self.value is not None: 
            return self.value.shape
        return self._shape

    @ shape.setter
    def shape(self, new_shape: List[Union[Text, int, None]]):
        self._shape = new_shape

    @ property
    def dtype(self) -> DataType:
        """ Return tensor shape of this variable
        It is modifiable when current variable is not a paramter.
        """
        if self.value is not None: 
            return DataType.convert_from_torch(self.value.dtype)
        return self._dtype
        
    @ dtype.setter
    def dtype(self, T: DataType):
        if isinstance(T, np.dtype):
            self._dtype = DataType.convert_from_numpy(T)
        elif isinstance(T, torch.dtype):
            self._dtype = DataType.convert_from_torch(T)
        elif isinstance(T, DataType):
            self._dtype = T
        else:
            raise TypeError(f'Invalid Dtype: {T} was given.')

    def copy(self, copy_value: bool = False):
        if not copy_value or self.value is None:
            return Variable(name=self.name, value=self.value, is_parameter=self.is_parameter)
        
        if not isinstance(self.value, torch.Tensor):
            ppq_warning(f'You are requiring to copy variable {self.name}, '
                        'however its value is not an instance of torch.Tensor, '
                        'ppq will automaticall convert it to torch.Tensor now.')
            self.value = convert_any_to_torch_tensor(self.value)
        if isinstance(self.value, torch.Tensor):
            value = self.value.clone()
        return Variable(name=self.name, value=value, is_parameter=self.is_parameter, shape=self.shape, dtype=self.dtype)


class Operation(OperationBase, Serializable):
    def __init__(
        self, name: str, op_type: str,
        attributes: Dict[str, Any], platform: TargetPlatform = TargetPlatform.UNSPECIFIED,
        inputs: List[Variable] = None, outputs: List[Variable] = None, opset: Opset = None) -> None:
        OperationBase.__init__(self, name, op_type, attributes, platform=platform, opset=opset)
        Serializable.__init__(self)
        self._input_vars    = [] if inputs is None else inputs
        self._output_vars   = [] if outputs is None else outputs

    @ property
    def socket(self) -> OpSocket:
        if self.type in DEFAULT_SOCKET_TABLE:
            return DEFAULT_SOCKET_TABLE[self.type](self)
        else: 
            return DEFAULT_SOCKET_CREATOR(self)

    @ property
    def inputs(self) -> List[Variable]:
        return self._input_vars

    @ property
    def outputs(self) -> List[Variable]:
        return self._output_vars

    @ property
    def parameters(self) -> List[Variable]:
        return [var for var in self.inputs if var.is_parameter]

    @ property
    def num_of_parameters(self) -> int:
        return len(self.parameters)

    @ property
    def is_linear_activation(self) -> bool:
        return self.type in LINEAR_ACTIVATIONS

    @ property
    def num_of_parameter(self) -> int:
        return len([var for var in self.inputs if var.is_parameter])

    @ property
    def is_boundary(self) -> bool:
        up_ops, down_ops = [], []
        for var in self.inputs:
            up_ops.append(var.source_op)
        for var in self.outputs:
            down_ops.extend(var.dest_ops)
        return all([op is None for op in up_ops]) or len(down_ops) == 0

    def __hash__(self) -> int:
        return self._name.__hash__()

    def __str__(self) -> str:
        return f'{self._name}({self.platform}) ' \
               f'- inputs:{[var.name for var in self.inputs]}, outputs:{[var.name for var in self.outputs]}'

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['_input_vars'] = [var.name for var in self.inputs]
        state['_output_vars'] = [var.name for var in self.outputs]
        return state

    def copy(self):
        clone = Operation(
            name=self.name, 
            op_type=self.type, 
            attributes=self.attributes.copy(), 
            platform=self.platform, 
            opset=self.opset)
        clone._detail = self._detail.copy()
        return clone


class BaseGraph(Serializable):
    """Graph is a PPQ Internal Represtation Data Structure.

    A computational graph is a directed graph where the nodes correspond to operations or variables.
    Variables can feed their value into operations, and operations can feed their output into other operations.
        This way, every node in the graph defines a function of the variables.

    The values that are fed into the nodes and come out of the nodes are called tensors,
        which is just a fancy word for a multi-dimensional array.
    Hence, it subsumes scalars, vectors and matrices as well as tensors of a higher rank.

    The computational graph created by PPQ contains quantization info as well as operations and variables.
        So to say it is a computational graph designed for quantization.

    All quantization related infos are stored within graph and its operations.
        See ppq.IR.quantize for more information.

    Args:
        Serializable ([type]): [description]
    """
    def __init__(self, name: str, built_from: NetworkFramework = NetworkFramework.NATIVE) -> None:
        super().__init__()
        self._operations    = {}
        self._variables     = {}
        self._graph_inputs  = {}
        self._graph_outputs = {}
        self._name = name
        self._built_from = built_from
        self._detail        = {}
        self._num_of_generated_var = 0
        self._num_of_generated_op  = 0


    @ property
    def operations(self) -> Dict[str, Operation]:
        return self._operations

    @ property
    def variables(self) -> Dict[str, Variable]:
        return self._variables

    @ property
    def inputs(self) -> Dict[str, Variable]:
        return self._graph_inputs

    @ property
    def outputs(self) -> Dict[str, Variable]:
        return self._graph_outputs

    def parameters(self) -> List[torch.Tensor]:
        parameters = []
        for var in self.variables.values():
            if var.is_parameter: 
                parameters.append(var.value)
        return parameters

    def set_extension_attrib(self, attrib: str, value: Any):
        self._detail[attrib] = value

    @ property
    def extension_attrib(self):
        return self._detail

    def append_operation(self, operation: Operation):
        if not isinstance(operation, Operation):
            raise TypeError(f'You can only insert operations via this function, however {type(operation)} was given.')
        if not all([var.name in self.variables for var in operation.inputs + operation.outputs]):
            raise KeyError(f'Inserting Operation {operation} has a related variable '\
                'which are not included in this graph yet, insert such variables before inserting this.')
        if operation.name in self.operations:
            raise KeyError(f'Duplicated Operation({operation}) was found, rename your Operation before inserting.')
        self.operations[operation.name] = operation

    def append_variable(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(f'You can only insert variable via this function, however {type(var)} was given.')
        if not all([dest_op.name in self.operations for dest_op in var.dest_ops]):
            raise KeyError(f'Inserting Variable {var} has a related Operation(dest_op) '\
                'which are not included in this graph yet, insert such Operations before inserting this.')
        if var.name in self.variables:
            raise KeyError(f'Duplicated Variable({var}) was found, rename your Variable before inserting.')
        self.variables[var.name] = var

    def get_downstream_operations(self, operation: Operation) -> List[Operation]:
        if not isinstance(operation, Operation):
            raise TypeError(f'Expect an operation instance, however {type(operation)} is given.')
        if operation.name not in self.operations:
            raise KeyError(f'Operation {operation.name} not in current graph.')
        downstream_ops = []
        for output_var in operation.outputs:
            downstream_ops.extend(output_var.dest_ops)
        return downstream_ops

    def get_upstream_operations(self, operation: Operation) -> List[Operation]:
        if not isinstance(operation, Operation):
            raise TypeError(f'Expect an operation instance, however {type(operation)} is given.')
        if operation.name not in self.operations:
            raise KeyError(f'Operation {operation.name} not in current graph.')
        upstream_ops = []
        for input_var in operation.inputs:
            if input_var.source_op is not None:
                upstream_ops.append(input_var.source_op)
        return upstream_ops

    def topological_sort(self) -> List[Operation]:
        visited = {operation.name: False for operation in self.operations.values()}
        sort_ret, pop_list = [], deque()
        num_of_inputs = {
            operation.name: len(self.get_upstream_operations(operation))
            for operation in self.operations.values()}

        # initialization
        for op_name, n_input in num_of_inputs.items():
            if n_input == 0: pop_list.append(op_name)

        # topological sort
        for _ in range(len(visited)):
            if len(pop_list) == 0: break
            op_name = pop_list.popleft()
            op = self.operations[op_name]
            for post_op in self.get_downstream_operations(op):
                num_of_inputs[post_op.name] -= 1
                if num_of_inputs[post_op.name] == 0:
                    pop_list.append(post_op.name)
            visited[op.name] = True
            sort_ret.append(op)
        if all(visited.values()):
            return sort_ret
        else:
            raise RuntimeError(
                'Topological Sort failed. Some operation can not be sorted (might due to circular reference).\n'
                ''.join(str(self.operations[op_name]) + '\n' for op_name in visited if visited[op_name] == False)
            )

    def insert_op_on_var(self, inserting_op: Operation, var: str):
        """Insert one operation to current graph. Inserting operation will
        replace var.dest_ops and automatically connect to inserting_op.

        Before insertion:
            op1 -> var -> op2

        After insertion:
            op1 -> var -> inserting_op -> link_var(generated) -> op2

        ATTENTION: Inserting operation must be an empty operation with no input and output variables linked to it.

        Args:
            inserting_op (Operation): [description]
            upstream_op (Operation): [description]
            downstream_op (Operation): [description]
        """
        if not isinstance(var, str):
            raise TypeError(f'Needs a variable name(str) here, however {type(var)} was given')
        if var not in self.variables:
            raise KeyError(f'Can not inserting operation at variable {var}, variable not found.')
        if len(inserting_op.inputs) != 0 or len(inserting_op.outputs) != 0:
            raise PermissionError('Can only insert operation with no input and output variables.')

        variable = self.variables[var]

        # add to graph.
        if inserting_op.name not in self.operations.keys():
            self.append_operation(inserting_op)

        # create all links.
        link_var = self.create_variable(
            name=None, value=None, is_parameter=False,
            dest_ops=variable.dest_ops.copy(), source_op=inserting_op)

        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)

        variable.dest_ops.clear()
        variable.dest_ops.append(inserting_op)

        for op in link_var.dest_ops:
            op.inputs[op.inputs.index(variable)] = link_var

        if var in self.outputs:
            self.outputs.pop(var)
            self.outputs[link_var.name] = link_var

    def insert_op_between_ops(self, inserting_op: Operation, up_op: Operation, down_op: Operation):
        """Insert one operation to current graph. Inserting operation will just
        between up_op and down_op.

        Example1(Insert Conv3 between Conv1 and Conv2):
            Before insertion: Conv1 -- Conv2
            After insertion:  Conv1 -- Conv3 -- Conv1

        Example2(Insert Conv3 between Conv1 and Conv2):

            Before insertion: Conv1 ----- Conv2
                                      |
                                      --- Conv4

            After insertion:  Conv1 ----- Conv3 -- Conv2
                                      |
                                      --- Conv4

        ATTENTION: Inserting operation must be an empty operation with no input and output variables linked to it.

        Args:
            inserting_op (Operation): [description]
            up_op (Operation): [description]
            down_op (Operation): [description]
        """
        if up_op.name not in self.operations:
            raise KeyError(f'Can not inserting operation behind {up_op.name}, operation not found.')
        if down_op.name not in self.operations:
            raise KeyError(f'Can not inserting operation behind {down_op.name}, operation not found.')
        if down_op not in self.get_downstream_operations(up_op):
            raise PermissionError(f'operation {up_op.name} and {down_op.name} are not linked,'
                                  ' there is no way to insert an op between them.')
        if len(inserting_op.inputs) != 0 or len(inserting_op.outputs) != 0:
            raise PermissionError('Can only insert operation with no input and output variables.')

        variables = []
        for var in down_op.inputs:
            if var.source_op == up_op:
                variables.append(var)
        assert len(variables) == 1, (f'Can not insert operation between {up_op.name} and {down_op.name},'
                                     ' graph is too complex.')
        [variable] = variables

        # add to graph.
        if inserting_op.name not in self.operations.keys():
            self.append_operation(inserting_op)

        # create all links.
        link_var = self.create_variable(
            name=None, value=None, is_parameter=False,
            dest_ops=[down_op], source_op=inserting_op)

        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)

        assert isinstance(variable, Variable)
        variable.dest_ops[variable.dest_ops.index(down_op)] = inserting_op
        down_op.inputs[down_op.inputs.index(variable)] = link_var

    def insert_op_between_var_and_op(self, inserting_op: Operation, up_var: Variable, down_op: Operation):
        """Insert one operation to current graph. Inserting operation will just
        between up_var and down_op.

        ATTENTION: Inserting operation must be an empty operation with no input and output variables linked to it.

        Args:
            inserting_op (Operation): [description]
            up_op (Operation): [description]
            down_op (Operation): [description]
        """
        if up_var.name not in self.variables:
            raise KeyError(f'Can not inserting operation behind {up_var.name}, variable not found.')
        if down_op.name not in self.operations:
            raise KeyError(f'Can not inserting operation behind {down_op.name}, operation not found.')
        if down_op.name not in [op.name for op in up_var.dest_ops]:
            raise PermissionError(f'variable {up_var.name} and {down_op.name} are not linked,'
                                  ' there is no way to insert an op between them.')
        if len(inserting_op.inputs) != 0 or len(inserting_op.outputs) != 0:
            raise PermissionError('Can only insert operation with no input and output variables.')

        variables = []
        for var in down_op.inputs:
            if var == up_var:
                variables.append(var)
        assert len(variables) == 1, (f'Can not insert operation between {var.name} and {down_op.name},'
                                     ' graph is too complex.')

        # add to graph.
        if inserting_op.name not in self.operations.keys():
            self.append_operation(inserting_op)

        # create all links.
        link_var = self.create_variable(
            name=None, value=None, is_parameter=False,
            dest_ops=[down_op], source_op=inserting_op)

        inserting_op.inputs.append(up_var)
        inserting_op.outputs.append(link_var)

        up_var.dest_ops[up_var.dest_ops.index(down_op)] = inserting_op
        down_op.inputs[down_op.inputs.index(up_var)] = link_var

    def create_link_with_op(self, variable: Variable, upstream_op: Operation, downstream_op: Operation):
        """Create a link with given variable from upstream_op to downstream_op
        variable will be appended to upstream_op's output and downstream_op's
        input given variable must have empty source_op or its source_op ==
        upstream_op.

        Sometime you may want to link a single upstream_op to many downstream_ops with a same variable,
            you are supposed to invoke this function for each downstream_op then.

        You can set upstream_op = None if your variable is a parameter variable.

        Example:
            create_link_with_op(var1, op1, op2)
            create_link_with_op(var1, op1, op3)

        Will makes:
                  --> op2
            op1 --|
                  --> op3

        Args:
            link_variable (Variable): _description_
            upstream_op (Operation): _description_
            downstream_op (Operation): _description_
        """
        if variable.name not in self.variables:
            raise KeyError(f'Can not find your variable {variable.name} in current graph.')
        if upstream_op is not None and upstream_op.name not in self.operations:
            raise KeyError(f'Can not find your operation {upstream_op.name} in current graph.')
        if downstream_op is not None and downstream_op.name not in self.operations:
            raise KeyError(f'Can not find your operation {downstream_op.name} in current graph.')

        if variable.source_op is None: variable.source_op = upstream_op
        if variable.source_op != upstream_op:
            raise PermissionError(f'Can not create link with variable {variable}, '
                                  f'cause its source operations != {upstream_op}')

        # For complex graph, following logic might have some error.
        if upstream_op is not None and variable not in upstream_op.outputs:
            upstream_op.outputs.append(variable)
        if downstream_op is None: return
        if downstream_op is not None and variable not in downstream_op.inputs:
            variable.dest_ops.append(downstream_op)
            downstream_op.inputs.append(variable)
        else: 
            variable.dest_ops.append(downstream_op)
            downstream_op.inputs.append(variable)
            ppq_warning(f'You are trying to link variable with operation, '
                          f'however Variable {variable.name} has already linked with downstream op {downstream_op.name}')

    def create_link_with_var(self, upstream_variable: Variable, downstream_variable: Variable):
        """connect upstream_variable.source_op with
        downstream_variable.dest_ops, downstream variable will be eliminated by
        this function.

        downstream_variable must have None as its source_op.

        Args:
            upstream_variable (_type_): _description_
            downstream_variable (_type_): _description_
        """
        if downstream_variable.source_op is not None:
            raise PermissionError(
                f'Can not create link with variable {upstream_variable.name} & {downstream_variable.name}, '
                'Cause downstream variable has a non-empty source op')

        dest_ops = downstream_variable.dest_ops
        for dest_op in dest_ops:
            dest_op.inputs[dest_op.inputs.index(downstream_variable)] = upstream_variable
            upstream_variable.dest_ops.append(dest_op)
        downstream_variable.dest_ops.clear()
        self.remove_variable(downstream_variable)
        return self

    def remove_operation(self, removing_op: Operation, keep_coherence: bool = False):
        """Remove operation from graph, this function will unlink removing
        operation from current graph, pop it from graph.operations, and remove
        it from all its input and output variables.

        Parameters of this removing operations will be removed from graph by this function, without warning.

        Args:
            removing_op (Operation): [description]
            
            keep_coherence (bool): if keep_coherence = True, 
                PPQ will link downstream operations of removing op to the upstream operation.
        """
        if removing_op.name not in self.operations:
            raise KeyError(f'Can not remove operation {removing_op.name}, operation not found.')

        # removing all parameters first.
        for parameter in removing_op.inputs.copy():
            if keep_coherence and removing_op.type in {'Constant', 'Identity'}: break
            if parameter.is_parameter:

                parameter.dest_ops.clear()
                parameter.value = None # clear memory.
                removing_op.inputs.remove(parameter)

                self.variables.pop(parameter.name)

        if not keep_coherence:
            # remove operation from its output variables
            for output_var in removing_op.outputs:
                output_var.source_op = None
            removing_op.outputs.clear()

            # remove operation from its input variables
            for input_var in removing_op.inputs:
                if removing_op in input_var.dest_ops:
                    input_var.dest_ops.remove(removing_op)
            removing_op.inputs.clear()

        else:
            if removing_op.num_of_input != 1:
                raise PermissionError(f'Can not remove operation {removing_op.name} with keep_coherence = True, '
                                      'operation must has exactly 1 input variable.')
            if removing_op.num_of_output != 1:
                raise PermissionError(f'Can not remove operation {removing_op.name} with keep_coherence = True, '
                                      'operation must has exactly 1 output variable.')
            
            input_var    = removing_op.inputs[0]
            removing_var = removing_op.outputs[0]
            dest_ops     = removing_var.dest_ops
            is_graph_output = removing_var.name in self.outputs
            
            for op in dest_ops:
                op.inputs[op.inputs.index(removing_var)] = input_var
                input_var.dest_ops.append(op)
            removing_var.dest_ops.clear()
            removing_var.source_op = None
            input_var.dest_ops.remove(removing_op)
            self.remove_variable(removing_var)

            if is_graph_output:
                self.mark_variable_as_graph_output(input_var)

        self.operations.pop(removing_op.name)
        return self

    def remove_variable(self, removing_var: Variable):
        """Remove variable from graph, this function will unlink removing
        variable from current graph, pop it from graph.variables, and remove it
        from its source op and dest ops.

        Args:
            removing_var (Variable): [description]
        """
        if removing_var.name not in self.variables:
            raise KeyError(f'Can not remove variable {removing_var.name}, variable not found.')

        # remove from source operation
        source_op = removing_var.source_op
        if source_op is not None:
            assert isinstance(source_op, Operation), (
                f'Can not remove variable {removing_var.name}, it links to a unexpected source operation.')
            if removing_var in source_op.outputs:
                source_op.outputs.remove(removing_var)
            removing_var.source_op = None

        # remove from all dest ops
        for dest_op in removing_var.dest_ops:
            assert isinstance(dest_op, Operation), (
                f'Can not remove variable {removing_var.name}, it links to a unexpected dest operation.')
            if removing_var in dest_op.inputs:
                dest_op.inputs.remove(removing_var)
        removing_var.dest_ops.clear()

        if removing_var.name in self.outputs:
            self.outputs.pop(removing_var.name)

        if removing_var.name in self.inputs:
            self.inputs.pop(removing_var.name)

        self.variables.pop(removing_var.name)
        return self

    def create_operation(self, op_type: str,  name: str = None,
        attributes: Dict[str, Any] = None, platform: TargetPlatform = TargetPlatform.UNSPECIFIED,
        inputs: List[Variable] = None, outputs: List[Variable] = None, **kwargs) -> Operation:
        """Create an operation and attach it it current graph. op_type is
        mandatory here, however op_name is not required. PPQ will automatically
        generates a name for your operation:
        PPQ_Operation_{self._num_of_generated_op}.

        Use this function carefully, cause once your network is quantized,
            simply create an operation via this function might cause unexpected error.
        Beawre that operation created by this function has no meta data and quantization info,
            which is needed to export and executing your graph.

        Do not set inputs and outputs via this function,
            to link your operation with others, use graph.create_link_with_var instead.

        Args:
            op_type (str): _description_
            name (str, optional): _description_. Defaults to None.
            attributes (Dict[str, Any], optional): _description_. Defaults to None.
            platform (TargetPlatform, optional): _description_. Defaults to TargetPlatform.UNSPECIFIED.
            inputs (List[Variable], optional): _description_. Defaults to None.
            outputs (List[Variable], optional): _description_. Defaults to None.

        Returns:
            Operation: _description_
        """
        if inputs is None:  inputs = []
        if outputs is None: outputs = []

        if name is None:
            name = f'PPQ_Operation_{self._num_of_generated_op}'
            self._num_of_generated_op += 1

        if not isinstance(inputs, list):
            raise TypeError(f'A list of input variable is required for creating operation, '
                            f'however {type(inputs)} was given')

        if attributes is None: attributes = {}
        created = Operation(
            name=name,
            op_type=op_type,
            attributes=attributes,
            platform=platform,
            inputs=inputs,
            outputs=outputs
        )
        self.append_operation(created)
        
        for item in inputs:
            if not isinstance(item, Variable):
                raise TypeError(f'A list contains variables is required for creating operation, '
                                f'however there is a {type(item)} in your input list.')
            item.dest_ops.append(created)
        
        if not isinstance(outputs, list):
            raise TypeError(f'A list of output variable is required for creating operation, '
                            f'however {type(inputs)} was given')
        for item in outputs:
            if not isinstance(item, Variable):
                raise TypeError(f'A list contains variables is required for creating operation, '
                                f'however there is a {type(item)} in your output list.')
            item.source_op = created
        return created

    def create_variable(self, name: str = None, value: Any = None, is_parameter: bool = False,
        dest_ops: List[OperationBase] = None, source_op: OperationBase = None, **kwargs) -> Variable:
        """Create a variable and attach it it current graph. PPQ will
        automatically generates a name for your variable:
        PPQ_Variable_{self._num_of_generated_op}.

        Use this function carefully, cause once your network is quantized,
            simply create an variable via this function might cause unexpected error.
        You'd better invoke this function before running your quantizer.

        Do not set dest_ops and source_op via this function,
            to link this variable with others, use graph.create_link_with_var instead.

        Args:
            name (str, optional): _description_. Defaults to None.
            value (Any, optional): _description_. Defaults to None.
            is_parameter (bool, optional): _description_. Defaults to False.
            dest_ops (List[OperationBase], optional): _description_. Defaults to None.
            source_op (OperationBase, optional): _description_. Defaults to None.

        Returns:
            Variable: _description_
        """
        if name is None:
            name = f'PPQ_Variable_{self._num_of_generated_var}'
            self._num_of_generated_var += 1

        created = Variable(
            name=name,
            value=value,
            is_parameter=is_parameter,
            dest_ops=dest_ops,
            source_op=source_op,
        )
        self.append_variable(created)
        return created

    def mark_variable_as_graph_input(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(f'Except a variable here, however {type(var)} was given.')
        var_name = var.name
        if var_name not in self.variables:
            raise KeyError(f'Can not find variable {var_name} within current graph.')
        if var_name in self.inputs: return
        if var_name in self.outputs:
            raise KeyError(f'Can not mark variable {var_name} as graph input, cause it is graph output.')
        self.inputs[var_name] = self.variables[var_name]

    def mark_variable_as_graph_output(self, var: Variable):
        if not isinstance(var, Variable):
            raise TypeError(f'Except a variable here, however {type(var)} was given.')
        var_name = var.name
        if var_name not in self.variables:
            raise KeyError(f'Can not find variable {var_name} within current graph.')
        if var_name in self.outputs: return
        self.outputs[var_name] = self.variables[var_name]

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['_graph_inputs'] = [var for var in self.inputs]
        state['_graph_outputs'] = [var for var in self.outputs]
        return state

    def copy(self, copy_value: bool = False):
        """Clone current graph. 
        Use parameter copy_value to control whether to do a Shallow Copy or Deep Copy.
        
        For copy_value = True, there will be a copy of each parameter in your network.
            ATTENTION: it might cause gpu memory overflow.
        For copy_value = False, cloned network will share the same parameter tensor of current one.
        
        ATTENTION: all quantization config will be cloned, 
            all scales and offsets will be cloned even with copy_valye = False.

        Shallow Copy: Shallow repetition is quicker.
        However, it's “lazy” it handles pointers and references.
        Rather than creating a contemporary copy of the particular knowledge the pointer points to,
            it simply copies over the pointer price.
        So, each the first and therefore the copy can have pointers that reference constant underlying knowledge.

        Deep Copy: Deep repetition truly clones the underlying data.
        It is not shared between the first and therefore the copy.
        """
        from ppq.IR.quantize import QuantableOperation
        cloned = BaseGraph(name=self._name, built_from=self._built_from)
        for op in self.operations.values(): cloned.append_operation(op.copy())
        for var in self.variables.values(): cloned.append_variable(var.copy(copy_value=copy_value))
        
        # notice that all operations is copyed without link, so do all variables
        # relink them with following code
        config_dict = {}
        for op in self.operations.values():
            assert op.name in cloned.operations, (
                f'Graph Copy Error, Operation {op.name} is not correctly cloned')
            c_op = cloned.operations[op.name]
            for i_var in op.inputs:
                assert i_var.name in cloned.variables, (
                    f'Graph Copy Error, Variable {i_var.name} is not correctly cloned')
                ci_var = cloned.variables[i_var.name]
                cloned.create_link_with_op(
                    variable=ci_var, 
                    upstream_op=ci_var.source_op, 
                    downstream_op=c_op)
            for o_var in op.outputs:
                assert o_var.name in cloned.variables, (
                    f'Graph Copy Error, Variable {o_var.name} is not correctly cloned')
                co_var = cloned.variables[o_var.name]
                c_op.outputs.append(co_var)
                co_var.source_op = c_op
            if isinstance(op, QuantableOperation):
                for cfg, var in op.config_with_variable:
                    config_dict[cfg._hash] = (op, var)

        # relink config to there cloned master.
        for c_op in cloned.operations.values():
            if isinstance(c_op, QuantableOperation):
                for cfg, var in c_op.config_with_variable:
                    if cfg.dominated_by != cfg:
                        assert cfg.dominated_by._hash in config_dict, (
                            'Graph Copy Error, can not find a corresponding master config.')
                        op, var = config_dict[cfg.dominated_by._hash]

                        op = cloned.operations[op.name]
                        assert isinstance(op, QuantableOperation), (
                            'Graph Copy Error, Unexpected Master Operation Type.')
                        for mcfg, mvar in op.config_with_variable:
                            if mvar.name == var.name: cfg._dominator = mcfg

        # recreate input, output
        for name in self.inputs:
            cloned.inputs[name] = cloned.variables[name]
        for name in self.outputs:
            cloned.outputs[name] = cloned.variables[name]

        # check integrity
        for op in self.operations.values():
            if op.name not in cloned.operations:
                raise KeyError(f'Graph Copy Error, Operation {op.name} is Missing')
        for var in self.variables.values():
            if var.name not in cloned.variables:
                raise KeyError(f'Graph Copy Error, Variable {var.name} is Missing')
        for name in self.inputs:
            if name not in cloned.inputs:
                raise KeyError(f'Graph Copy Error, Input {var.name} is Missing')
        for name in self.outputs:
            if name not in cloned.outputs:
                raise KeyError(f'Graph Copy Error, Output {var.name} is Missing')
        cloned._num_of_generated_op  = self._num_of_generated_op
        cloned._num_of_generated_var = self._num_of_generated_var
        cloned._detail = self._detail.copy()
        return cloned


class GraphBuilder(metaclass=SingletonMeta):
    @ abstractmethod
    def build(self, file_path: str, **kwargs) -> BaseGraph: pass


class GraphExporter(metaclass=SingletonMeta):
    @ abstractmethod
    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, **kwargs): pass


class OperationExporter(metaclass=SingletonMeta):
    @ abstractmethod
    def export(self, operation:Operation, graph: BaseGraph, **kwargs) -> Operation: pass
