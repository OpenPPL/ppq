from typing import Any, List

import torch
from ppq.core import (TargetPlatform, convert_any_to_numpy,
                      convert_any_to_torch_tensor)

from .base.command import GraphCommand, GraphCommandType, GraphDeployCommand
from .base.graph import BaseGraph, Operation, Variable
from .processer import GraphCommandProcessor
from .quantize import QuantableOperation


class RunnableGraph(GraphCommandProcessor):
    def __init__(self, graph: BaseGraph, device: str = None):
        """RunnableGraph object aims at dealing things related with graph
        executing.

        Literally it helps you move values of your graph towards device and vice versa.
            And give an executable order of all operations in your graph which actual executor will follow.
        Args:
            graph (BaseGraph): BaseGraph instance.
            device (str, optional): This attribute is only used by with RunnableGraph(graph, device) syntactic.
            next_command_processor (Callable, optional): next processor in processing chain.
        """
        super().__init__(graph_or_processor=graph)
        self._device = device  # only used in "with RunnableGraph(graph, device):"

    def process(self, command: GraphCommand) -> Any:

        if command.command_type == GraphCommandType.DEPLOY_TO_CPU:
            return self.deploy('cpu')

        elif command.command_type == GraphCommandType.DEPLOY_TO_CUDA:
            if isinstance(command, GraphDeployCommand):
                device = command._device
                return self.deploy(device)
            else:
                return self.deploy('cuda')

        elif command.command_type == GraphCommandType.DEPLOY_TO_NUMPY:
            return self.retrieve()

    def __enter__(self):
        self.deploy(self._device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.retrieve()

    def _acceptable_command_types(self) -> List[GraphCommandType]:
        return [
            GraphCommandType.DEPLOY_TO_CPU,
            GraphCommandType.DEPLOY_TO_CUDA,
            GraphCommandType.DEPLOY_TO_NUMPY
        ]

    def retrieve(self):

        for _, operator in self._graph.operations.items():

            assert isinstance(operator, Operation), \
                f'Failed to retrieve graph to numpy, incorrect operator {operator} found.'

            # in onnx format, some constant values are warpped with operation's attributes['value']
            # To move those constant value from numpy to device,
            # we have to move all the attributes['value'] of operation to device(if there is any).
            if operator.type == 'Constant':
                operator.attributes['value'] = \
                    convert_any_to_numpy(operator.attributes['value'])

        for _, variable in self._graph.variables.items():
            assert isinstance(variable, Variable), \
                f'Failed to send graph to device, incorrect variable {variable} found.'
            variable.value = convert_any_to_numpy(variable.value, accept_none=True)

        return self

    def deploy(self, device: str):

        for _, operator in self._graph.operations.items():
            assert isinstance(operator, Operation), \
                f'Failed to send graph to device, incorrect operator {operator} found.'

            # in onnx format, some constant values are warpped with operation's attributes['value']
            # To move those constant value from device to numpy,
            # we have to move all the attributes['value'] of operation to numpy(if there is any).
            if operator.type == 'Constant' and operator.platform != TargetPlatform.SHAPE_OR_INDEX:
                operator.attributes['value'] = \
                    convert_any_to_torch_tensor(
                        operator.attributes['value'], accept_none=False).to(device)

            if operator.type == 'Constant' and operator.platform == TargetPlatform.SHAPE_OR_INDEX:
                value = operator.attributes['value']
                operator.attributes['value'] = convert_any_to_torch_tensor(
                    value, accept_none=False, device='cpu')

            # PATCH 20220706, send quantization config to device.
            if isinstance(operator, QuantableOperation):
                for cfg, var in operator.config_with_variable:
                    if isinstance(cfg._scale, torch.Tensor): cfg._scale = cfg._scale.to(device)
                    if isinstance(cfg._offset, torch.Tensor): cfg._offset = cfg._offset.to(device)

        for _, variable in self._graph.variables.items():
            assert isinstance(variable, Variable), \
                f'Failed to send graph to device, incorrect variable {variable} found.'
            # graph output variable has no destinations
            if len(variable.dest_ops) == 0: continue
            if variable.value is None: continue

            # check all destination operations platform are same.
            platforms = [op.platform for op in variable.dest_ops]
            if not all([_ == platforms[0] for _ in platforms]):
                raise TypeError(f'Not all down-steram operations for var {variable} '
                                'share a same target platform, split this variable first.')
            platform = platforms[-1]

            # if all downstream operations are shape related operations, send value to cpu
            if platform == TargetPlatform.SHAPE_OR_INDEX:
                variable.value = convert_any_to_torch_tensor(
                    variable.value, accept_none=True).to('cpu')
            else:
                variable.value = convert_any_to_torch_tensor(
                    variable.value, accept_none=True).to(device=device)

            # if variable is a shape-related variable, send it to cpu.
            if variable.is_parameter:
                if len(variable.dest_ops) > 1:
                    raise PermissionError(
                        f'PPQ can not process parameter variable({variable.name})'
                        f' with multiple destinations({[op.name for op in variable.dest_ops]}), split it first.')
                dest_op = variable.dest_ops[0]
                dest_idx = dest_op.inputs.index(variable)

                if dest_op.type in {'Squeeze', 'Unsqueeze', 'ReduceSum', 'Reshape', 'Slice', 'Gather', 'Pad', 'Resize', 
                    'Split', 'TopK', 'Tile', 'Expand'}:
                    if dest_idx >= 1 and len(variable.dest_ops) == 1:
                        variable.value = convert_any_to_torch_tensor(
                            variable.value, accept_none=True).to('cpu')
        return self
