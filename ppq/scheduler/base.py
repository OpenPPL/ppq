from abc import abstractstaticmethod
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
    @ staticmethod
    @ abstractstaticmethod
    def dispatch(self, graph: BaseGraph, quant_types: List[str],
                 quant_platform: TargetPlatform,
                 fp32_platform: TargetPlatform,
                 SOI_platform: TargetPlatform, **kwargs
                 ) -> Dict[str, TargetPlatform]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_platform - all quantable parts of graph will be dispatched to this platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched to this platform.
            fp32_platform  - there are some operations receiving results from both quant_platform and SOI_platform,
                they will be dispatched to fp32_platform.

        ATTENTION: Quantization follows this dispatching,
            and only the operations within quantable platform will be quantized in the future.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this dispatcher.

            quant_types(Set[str]): all quantable types for given platforms.

            quant_platform (TargetPlatform):
                platform object where quantable parts will goes to.

            SOI_platform (TargetPlatform):
                platform object where SOI parts will goes to.

            fp32_platform (TargetPlatform):
                platform object where remaining parts will goes to.

        Returns:
            Dict[str, TargetPlatform]: [description]
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
        return to_where == from_where.inputs[0].source_op or  to_where == from_where.inputs[-1].source_op
    if from_where.type in {'NonMaxSuppression', 'Shape'}: # 'ConstantOfShape'}:
        # remove constant of shape from here can speed up.
        return False
    return True

def SOI_receivers(graph: BaseGraph) -> Set[Operation]:
    _ret_collection = set()
    for operation in graph.operations.values():
        if operation.type == 'Reshape':
            # Inputs:
            #   data (differentiable) : T
            #       An input tensor.
            #   shape (non-differentiable) : tensor(int64)
            #       Specified shape for output.
            # see also https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
            _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'Slice':
            # Inputs (3 - 5)
            #   data (differentiable) : T
            #       Tensor of data to extract slices from.
            #   starts (non-differentiable) : Tind
            #       1-D tensor of starting indices of corresponding axis in `axes`
            #   ends (non-differentiable) : Tind
            #       1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
            #   axes (optional, non-differentiable) : Tind
            #       1-D tensor of axes that `starts` and `ends` apply to. Negative value means
            #       counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
            #   steps (optional, non-differentiable) : Tind
            #       1-D tensor of slice step of corresponding axis in `axes`.
            #       Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.
            # see also https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Slice-11
            for shape_var in operation.inputs[1: ]:
                _ret_collection.add(shape_var.source_op)

        if operation.type == 'Gather':
            # Inputs
            #   data (differentiable) : T
            #       Tensor of rank r >= 1.
            #   indices (non-differentiable) : Tind
            #       Tensor of int32/int64 indices, of any rank q.
            #       All index values are expected to be within bounds [-s, s-1] along axis of size s.
            #       It is an error if any of the index values are out of bounds.
            # see also https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Gather-11
            _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'Pad':
            # Inputs (2 - 3)
            #   data : T
            # Input tensor.
            #   pads : tensor(int64)
            #       Tensor of integers indicating the number of padding elements to add or remove
            #       (if negative) at the beginning and end of each axis.
            #       For 2D input tensor, it is the number of pixels. `pads` should be a 1D tensor of shape [2 * input_rank].
            #       `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
            #        where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end,
            #       the number of pad values added at the end of axis `i`.
            #   constant_value (optional) : T
            #       (Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0).
            # https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Pad-11
            if len(operation.inputs) >= 2:
                _ret_collection.add(operation.inputs[1].source_op)

        if operation.type == 'Resize':
            # Inputs (3 - 4)
            #   X : T1
            #       N-D tensor
            #   roi : T2
            #       1-D tensor given as [start1, ..., startN, end1, ..., endN],
            #       where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image.
            #       It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
            #   scales : tensor(float)
            #       The scale array along each dimension.
            #       It takes value greater than 0. If it's less than 1, it's sampling down,
            #       otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'.
            #       Only one of 'scales' and 'sizes' can be specified.
            #       If 'size' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.
            #   sizes (optional) : tensor(int64)
            #       The size of the output tensor.
            #       The number of elements of 'sizes' should be the same as the rank of input 'X'.
            #       Only one of 'scales' and 'sizes' can be specified.
            # https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Resize-11
            for shape_var in operation.inputs[1: ]:
                _ret_collection.add(shape_var.source_op)

        if operation.type == 'Split':
            # Inputs (1 - 2)
            #   input (differentiable) : T
            #       The tensor to split
            #   split (optional, non-differentiable) : tensor(int64) (opset 13)
            #       Optional length of each output.
            #       Values should be >= 0.Sum of the values must be equal to the dim value at 'axis' specified.
            # see also: https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Split-11
            # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split
            if len(operation.inputs) == 2:
                _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'TopK':
            # Inputs: (2)
            #   X (differentiable) : T
            #       Tensor of shape [a_1, a_2, ..., a_n, r]
            #   K (non-differentiable) : tensor(int64)
            #       A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
            # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK-11
            if len(operation.inputs) == 2:
                _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'Tile':
            # Inputs: (2)
            #   input : T
            #       Input tensor of any shape.
            #   repeats : T1
            #       1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.
            # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tile
            if len(operation.inputs) == 2:
                _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'Expand':
            # Inputs: (2)
            #   input : T
            #       input (differentiable) : T
            #   shape (non-differentiable) : tensor(int64)
            #      A 1-D tensor indicates the shape you want to expand to, following the broadcast rule
            # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tile
            if len(operation.inputs) == 2:
                _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'ConstantOfShape':
            # Inputs: (1)
            #   input : T
            #       1D tensor. The shape of the expected output tensor.
            # If empty tensor is given, the output would be a scalar. All values must be >= 0.
            # see also: https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape
            _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'RoiAlign':
            _ret_collection.add(operation.inputs[-1].source_op)
            _ret_collection.add(operation.inputs[-2].source_op)

        if operation.type == 'MMCVRoiAlign':
            _ret_collection.add(operation.inputs[-1].source_op)

        if operation.type == 'ScatterND':
            _ret_collection.add(operation.inputs[1].source_op)
        
        # FOR opset13
        if operation.type == 'Squeeze' or operation.type == 'Unsqueeze' or operation.type == 'ReduceSum':
            for var in operation.inputs[1:]:
                _ret_collection.add(var.source_op)


    # end for
    if None in _ret_collection: _ret_collection.remove(None)
    return _ret_collection

def SOI_generators(graph: BaseGraph) -> Set[Operation]:
    _ret_collection = set()
    for operation in graph.operations.values():
        if operation.type in {'Shape', 'NonMaxSuppression', 'Constant', 'ConstantOfShape', 'TopK'}:
            _ret_collection.add(operation)
    return _ret_collection
