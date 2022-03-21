from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque
from typing import Any, Dict, List

from ppq.core import (COMPUTING_OP, LINEAR_ACTIVATIONS, SOI_OP,
                      NetworkFramework, OperationMeta, Serializable,
                      SingletonMeta, TargetPlatform, TensorMeta)


class OperationBase(metaclass=ABCMeta):
    def __init__(self, 
                 name: str, op_type: str, 
                 attributes: Dict[str, Any], 
                 platform: TargetPlatform=TargetPlatform.UNSPECIFIED) -> None:
        self._name = name
        self._type = op_type
        self._attributes = attributes
        self._platform = platform
        self._meta = None

    @ abstractproperty
    def inputs(self) -> List[Any]: pass

    @ abstractproperty
    def outputs(self) -> List[Any]: pass

    @ abstractproperty
    def parameters(self) -> List[Any]: pass

    @ property
    def name(self) -> str:
        return self._name

    @ property
    def type(self) -> str:
        return self._type

    @ type.setter
    def type(self, type: str):
        self._type = type

    @ property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    @ property
    def platform(self) -> TargetPlatform:
        return self._platform

    @ platform.setter
    def platform(self, platform: TargetPlatform):
        self._platform = platform

    @ property
    def meta_data(self) -> OperationMeta:
        return self._meta

    @ meta_data.setter
    def meta_data(self, meta: OperationMeta) -> OperationMeta:
        self._meta = meta

    def __hash__(self) -> int:
        return self._name.__hash__()


class Variable(Serializable):
    def __init__(self, name: str, value: Any = None, is_parameter: bool = False,
                 dest_ops: List[OperationBase] = None, source_op: OperationBase = None) -> None:
        super().__init__()
        self._name = name
        self._value = None if value is None else value
        self._dest_ops = [] if dest_ops is None else dest_ops
        self._source_op = None if source_op is None else source_op
        self._is_parameter = is_parameter

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
    def value(self) -> Any:
        return self._value

    @ value.setter
    def value(self, value) -> None:
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
        if self.source_op is not None:
            if self.source_op.meta_data is None: return None 
            return self.source_op.meta_data.output_metas[self.src_idx]
        elif len(self.dest_ops) > 0:
            dest_op = self.dest_ops[0]
            dest_idx = self.dest_idx[0]
            if dest_op.meta_data is None: return None
            return dest_op.meta_data.input_metas[dest_idx]
        else:
            raise RuntimeError(f'Seems you got an isolated variable {self.name}, '\
                'PPQ is not able to infer its meta data yet.')

    def __hash__(self) -> int:
        return self._name.__hash__()
    
    def __str__(self) -> str:
        return f'Variable ({self._name})'
    
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['_dest_ops'] = [op.name for op in self.dest_ops]
        state['_source_op'] = self.source_op.name if self.source_op is not None else None
        return state


class Operation(OperationBase, Serializable):
    def __init__(
        self, name: str, op_type: str, 
        attributes: Dict[str, Any], platform: TargetPlatform = TargetPlatform.UNSPECIFIED,
        inputs: List[Variable] = None, outputs: List[Variable] = None) -> None:
        OperationBase.__init__(self, name, op_type, attributes, platform=platform)
        Serializable.__init__(self)
        self._input_vars    = [] if inputs is None else inputs
        self._output_vars   = [] if outputs is None else outputs

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
    def is_computing_op(self) -> bool:
        return self.type in COMPUTING_OP

    @ property
    def is_soi_generator(self) -> bool:
        return self.type in SOI_OP

    @ property
    def is_boundary(self) -> bool:
        up_ops, down_ops = [], []
        for var in self.inputs:
            up_ops.append(var.source_op)
        for var in self.outputs:
            down_ops.extend(var.dest_ops)
        return all([op is None for op in up_ops]) or len(down_ops) == 0

    @ property
    def is_linear_activation(self) -> bool:
        return self.type in LINEAR_ACTIVATIONS
    
    @ property
    def num_of_input(self) -> int:
        return len(self.inputs)

    @ property
    def num_of_output(self) -> int:
        return len(self.outputs)
    
    @ property
    def num_of_parameter(self) -> int:
        return len([var for var in self.inputs if var.is_parameter])

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


class BaseGraph(Serializable):
    """
    Graph is a PPQ Internal Represtation Data Structure.
    
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
    def __init__(self, name: str, built_from: NetworkFramework) -> None:
        super().__init__()
        self._operations    = {}
        self._variables     = {}
        self._graph_inputs  = {}
        self._graph_outputs = {}
        self._name = name
        self._built_from = built_from
        self._detail        = {}
        self._num_of_generated_var = 0


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

    def delete_operation(self, op_name: str, cascade: bool = False, force_delete: bool = False):
        if not isinstance(op_name, str): 
            raise TypeError(f'This function needs a operation name as parameter, '\
                f'while {type(op_name)} was given')
        if op_name not in self.operations: return
        operation = self.operations[op_name]
        if len(operation.outputs) != 0 and not cascade and not force_delete:
            raise PermissionError(f'It is not safe to delete opeartion {op_name}, '\
                f'cause it still has output variable(s) {[str(output_var) for output_var in operation.outputs]}')
        for input_var in operation.inputs:
            dest_idx = input_var.dest_ops.index(operation)
            input_var.dest_ops.pop(dest_idx)
            # once variable is isolated, delete it.
            if len(input_var.dest_ops) == 0 and input_var.name not in self.outputs:
                self.delete_variable(input_var.name, force_delete=force_delete)
        self.operations.pop(operation.name)

        if cascade:
            for output_var in operation.outputs:
                for cascade_op in output_var.dest_ops:
                    self.delete_operation(cascade_op.name, cascade=True, force_delete=force_delete)

    def delete_variable(self, var_name: str, force_delete: bool = False):
        if not isinstance(var_name, str): 
            raise TypeError(f'This function need a variable name to delete variable from graph, '\
                f'while {type(var_name)} was given')
        if var_name in self.inputs or var_name in self.outputs:
            raise PermissionError('Can not delete graph input and output variables.')
        if var_name not in self.variables:
            raise KeyError(f'Variable {var_name} not in current graph.')
        variable = self.variables[var_name]
        if len(variable.dest_ops) != 0 and not force_delete:
            raise PermissionError(f'It is not safe to delete variable {variable}, '\
                f'cause it still has output operation(s) {[dest_op.name for dest_op in variable.dest_ops]}')
        if variable.source_op is not None and len(variable.source_op.outputs) != 1 and not force_delete:
            raise PermissionError(
                f'It is not safe to delete variable {variable}, Cause its source operation {variable.source_op.name} '
                'has more than 1 output, this deletion will change output order.')
        source_op = variable.source_op
        if source_op is not None:
            output_idx = source_op.outputs.index(variable)
            source_op.outputs.pop(output_idx)
        self.variables.pop(variable.name)

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

        # initilization
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
                'Topological Sort failed. Some operation can not be sorted (might due to circular reference).\n' + \
                ''.join(str(self.operations[op_name]) + '\n' for op_name in visited if visited[op_name] == False)
            )

    def insert_op_on_var(self, inserting_op: Operation, var: str):
        """
        Insert one operation to current graph.
            Inserting operation will replace var.dest_ops and automatically connect to var.source_op.

        ATTENTION: Inserting opeartion must be an empty opeartion with no input and output variables linked to it.

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
        self.append_operation(inserting_op)

        # create all links.
        link_var = Variable(name=f'PPQ_Generated_Var_{self._num_of_generated_var}', 
                            dest_ops=variable.dest_ops.copy(), 
                            source_op=inserting_op)
        self.append_variable(link_var)

        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)

        variable.dest_ops.clear()
        variable.dest_ops.append(inserting_op)
        
        for op in link_var.dest_ops:
            op.inputs[op.inputs.index(variable)] = link_var
        
        if var in self.outputs:
            self.outputs.pop(var)
            self.outputs[link_var.name] = link_var

        self._num_of_generated_var += 1

    def insert_op_between_ops(self, inserting_op: Operation, up_op: Operation, down_op: Operation):
        """
        Insert one operation to current graph.
            Inserting operation will just between up_op and down_op.

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

        ATTENTION: Inserting opeartion must be an empty opeartion with no input and output variables linked to it.

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
        assert len(variables) == 1, (f'Can not insert opeartion between {up_op.name} and {down_op.name},'
                                     ' graph is too complex.')
        [variable] = variables
        
        # add to graph.
        self.append_operation(inserting_op)

        # create all links.
        link_var = Variable(name=f'PPQ_Generated_Var_{self._num_of_generated_var}', 
                            dest_ops=[down_op], 
                            source_op=inserting_op)
        self.append_variable(link_var)

        inserting_op.inputs.append(variable)
        inserting_op.outputs.append(link_var)

        assert isinstance(variable, Variable)
        variable.dest_ops[variable.dest_ops.index(down_op)] = inserting_op
        down_op.inputs[down_op.inputs.index(variable)] = link_var
        self._num_of_generated_var += 1

    def remove_operation(self, removing_op: Operation):
        """
        Remove opeartion from graph, this function will unlink removing operation from
            current graph, and delete it from graph.
        
        removing operation is supposed to have only one output variable and one input variable,
            and its output variable should not be the output of current graph. 
        Otherwise errors will be thrown by this function.

        This function will auto link the source variable of removing operation to all 
            downstream operations(as their input instead).

        removing operation and its output variable will be deleted from current graph.

        Args:
            removing_op (Operation): [description]
        """
        if removing_op.name not in self.operations:
            raise KeyError(f'Can not remove operation {removing_op.name}, operation not found.')
        if len(removing_op.outputs) != 1 or len(removing_op.inputs) != 1:
            raise PermissionError(f'Can not remove operation {removing_op.name},'
                                  ' it has more than 1 input or output variable.')
        if removing_op.outputs[0].name in self._graph_outputs:
            raise PermissionError(f'Can not remove operation {removing_op.name},'
                        ' its output variable is listed in graph output.')

        removing_var = removing_op.outputs[0]
        input_var    = removing_op.inputs[0]
        downstream_ops = self.get_downstream_operations(removing_op)

        input_var.dest_ops.remove(removing_op)
        input_var.dest_ops.extend(downstream_ops)

        removing_op.inputs.clear()
        removing_op.outputs.clear()

        removing_var.dest_ops.clear()
        for downstream_op in downstream_ops:
            downstream_op.inputs[downstream_op.inputs.index(removing_var)] = input_var

        self.operations.pop(removing_op.name)
        self.variables.pop(removing_var.name)

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state['_graph_inputs'] = [var for var in self.inputs]
        state['_graph_outputs'] = [var for var in self.outputs]
        return state

    def copy(self):
        """
        Clone this graph. Here a shallow copy will be returned as result.

        Shallow Copy: Shallow repetition is quicker. 
        However, it’s “lazy” it handles pointers and references. 
        Rather than creating a contemporary copy of the particular knowledge the pointer points to, 
            it simply copies over the pointer price. 
        So, each the first and therefore the copy can have pointers that reference constant underlying knowledge.

        Deep Copy: Deep repetition truly clones the underlying data.
        It is not shared between the first and therefore the copy.
        """
        cloned = BaseGraph(name=self._name, built_from=self._built_from)
        cloned._graph_inputs  = self.inputs.copy()
        cloned._graph_outputs = self.outputs.copy()
        cloned._export_value  = self._export_value
        cloned._operations    = self.operations.copy()
        cloned._variables     = self.variables.copy()
        return cloned


class GraphBuilder(metaclass=SingletonMeta):
    @ abstractmethod
    def build(self, file_path: str, **kwargs) -> BaseGraph: pass


class GraphExporter(metaclass=SingletonMeta):
    @ abstractmethod
    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, **kwargs): pass
