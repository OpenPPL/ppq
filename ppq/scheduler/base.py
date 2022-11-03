from abc import abstractmethod
from typing import Dict, List, Set
from ppq.IR import BaseGraph, Operation
from ppq.core import TargetPlatform

class GraphDispatcher:
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable operations,
        Shape-Or-Index related operations and remaining operations, all sub classes of GraphDispatcher will
        give an implementation of function "dispatch" to send all operations to their proper platform.

    ATTENTION: platform attribute will greatly affect quantizer's quantization logic, and the execution result.
        If operation is sent to a quantable platform, then its inputs and outputs will be quantized if necessary.
        if operation is classified as shape-or-index related operation, then its execution will be taken with cpu.
        if operation is sent to a fp32 platform, then its inputs and outputs shall never be quantized.
    """
    @ abstractmethod
    def dispatch(self, quant_types: Set[str], **kwargs) -> Dict[str, TargetPlatform]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.
        """
        raise NotImplementedError('Impl this first.')

def value_tracing_pattern(from_where: Operation, to_where: Operation) -> bool:
    if to_where.type in {'Reshape', 'Slice', 'Gather', 'Pad', 'Resize',
                         'Split', 'TopK', 'Tile', 'Expand', 'RoiAlign', 'MMCVRoiAlign'}:
        # shape can go through above operations as a input, under this circumstance, their output should still be a tensor of shape.
        # however if shape was feed as a parameter for those operations, then their outputs are irrelevant with shape flow.
        return to_where.inputs[0].source_op == from_where
    if to_where.type == 'ScatterND':
        # ScatterND has 2 quant input.
        return to_where.inputs[0].source_op == from_where or to_where.inputs[-1].source_op == from_where

    if to_where.type in {'ConstantOfShape', 'Shape', 'NonMaxSuppression'}:
        # Inputs: (1)
        #   input : T
        #       1D tensor. The shape of the expected output tensor.
        #       If empty tensor is given, the output would be a scalar. All values must be >= 0.
        # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape
        return False

    return True

def reverse_tracing_pattern(from_where: Operation, to_where: Operation) -> bool:
    if to_where.type in {'Shape'}: return False
    if to_where.type == 'TopK':
        return False
    if from_where.type in {'Reshape', 'Slice', 'Gather', 'Pad', 'Resize',
                            'Split', 'TopK', 'Tile', 'Expand', 'RoiAlign', 'MMCVRoiAlign'}:
        return to_where == from_where.inputs[0].source_op
    if from_where.type == 'ScatterND':
        return to_where == from_where.inputs[0].source_op or to_where == from_where.inputs[-1].source_op
    if from_where.type in {'NonMaxSuppression', 'Shape'}: # 'ConstantOfShape'}:
        # remove constant of shape from here can speed up.
        return False
    return True

def SOI_receivers(graph: BaseGraph) -> Set[Operation]:
    receivers = set()
    for operation in graph.operations.values():
        for idx, plat in enumerate(operation.socket.in_plat):
            if plat in {TargetPlatform.SOI, TargetPlatform.FP32}:
               receivers.add(operation.inputs[idx].source_op)

    if None in receivers: receivers.remove(None)
    return receivers

def SOI_generators(graph: BaseGraph) -> Set[Operation]:
    _ret_collection = set()
    for operation in graph.operations.values():
        if operation.type in {'Shape', 'NonMaxSuppression', 'Constant', 'ConstantOfShape', 'TopK'}:
            _ret_collection.add(operation)
    return _ret_collection
