import operator
from functools import reduce
from typing import List

import numpy as np
from ppq.core import (DataType, convert_any_to_python_primary_type,
                      convert_any_to_torch_tensor)
from ppq.IR import Operation

import torch
import torch.nn.functional as F

from .base import *


def Shape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Takes a tensor as input and outputs an 1D int64 tensor containing the
    shape of the input tensor.

    Version
        This version of the operator has been available since version 1 of the default ONNX operator set.

    Inputs
        data : T
        An input tensor.

    Outputs
        shape : T1
        Shape of the input tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    shape_tensor = torch.Tensor([k for k in value.shape]).long()
    return shape_tensor

def Div_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Performs element-wise binary division (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    Version
        This version of the operator has been available since version 7 of the default ONNX operator set.

    Inputs
        A : T
        First operand.

        B : T
        Second operand.

    Outputs
        C : T
        Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        input_values (list): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)

    dividend, divider = values
    quotient = dividend / divider
    if dividend.dtype in {torch.int16, torch.int64, torch.int8, torch.int32} and \
        divider.dtype in {torch.int16, torch.int64, torch.int8, torch.int32}:
        quotient = quotient.type(torch.int64)
    return quotient

def Mul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Performs element-wise binary multiplication (with Numpy-style
    broadcasting support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

    Inputs
        A (differentiable) : T
            First operand.

        B (differentiable) : T
            Second operand.

    Outputs
        C (differentiable) : T
            Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    multiplicand, multiplier = values
    return multiplicand * multiplier

def Add_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Performs element-wise binary addition (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting;
        for more details please check the doc.

    (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

    Inputs
        A (differentiable) : T
            First operand.
        B (differentiable) : T
            Second operand.

    Outputs
        C (differentiable) : T
            Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        input_values (list): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a + b

def And_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a & b

def Sub_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Performs element-wise binary subtraction (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    Version
        This version of the operator has been available since version 7 of the default ONNX operator set.

    Inputs
        A : T
        First operand.

        B : T
        Second operand.

    Outputs
        C : T
        Result, has same element type as two inputs

    Args:
        op (Operation): [description]
        input_values (list): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    minuend, subtrahend = values
    return minuend - subtrahend

def Cast_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """The operator casts the elements of a given input tensor to a data type
    specified by the 'to' argument and returns an output tensor of the same
    size in the converted type. The 'to' argument must be one of the data types
    specified in the 'DataType' enum field in the TensorProto message.

    Casting from string tensor in plain (e.g., "3.14" and "1000") and
        scientific numeric representations (e.g., "1e-5" and "1E8") to float types is supported.

    For example, converting string "100.5" to an integer may result 100.
    There are some string literals reserved for special floating-point values;
    "+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity,
        and not-a-number, respectively.

    Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite.
    Similarly, this case-insensitive rule is applied to "INF" and "NaN".
    When casting from numeric tensors to string tensors,
        plain floating-point representation (such as "314.15926") would be used.

    Converting non-numerical-literal string such as "Hello World!" is an undefined behavior.
    Cases of converting string representing floating-point arithmetic value, such as "2.718",
        to INT is an undefined behavior.

    Conversion from a numerical type to any numerical type is always allowed.
    User must be aware of precision loss and value change caused by range difference between two types.

    For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592.
    Similarly, converting an integer 36 to Boolean may produce 1
        because we truncate bits which can't be stored in the targeted type.

    Args:
        op (Operation): [description]
        values (list): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [converting] = values
    new_type = DataType.to_torch(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='to', compulsive=True))
    old_type = converting.dtype
    if new_type != old_type:
        return converting.type(new_type)
    else: return converting

def Concat_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Concatenate a list of tensors into a single tensor. All input tensors
    must have the same shape, except for the dimension size of the axis to
    concatenate on.

    Attributes
        axis : int (required)
            Which axis to concat on.
            A negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(inputs)..

    Inputs (1 - ∞)
        inputs (variadic, differentiable) : T
            List of tensors for concatenation

    Outputs
        concat_result (differentiable) : T
            Concatenated tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', compulsive=True)

    # zero-dimensional tensor cannot be concatenated
    # must extend 1 dimension for concat.
    concat_view = []
    for value in values:
        if value.ndim == 0:
            value = value.unsqueeze(0)
        concat_view.append(value)

    return torch.cat(concat_view, axis=axis)

def Slice_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Produces a slice of the input tensor along multiple axes. Similar to
    numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slices uses starts, ends, axes and steps inputs to specify the start and
    end dimension and step for each axis in the list of axes,

    it uses this information to slice the input data tensor.
    If a negative value is passed for any of the start or end indices,
        it represents number of elements before the end of that dimension.

    If the value passed to start or end is larger than the n (the number of elements in this dimension),
        it represents n. For slicing to the end of a dimension with unknown size,
        it is recommended to pass in INT_MAX when sclicing forward and 'INT_MIN' when slicing backward.

    If a negative value is passed for step, it represents slicing backward.
    However step value cannot be 0. If axes are omitted, they are set to [0, ..., ndim-1].

    If steps are omitted, they are set to [1, ..., 1] of length len(starts)

    Example 1: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] axes = [0, 1] starts = [1, 0] ends = [2, 3] steps = [1, 2] result = [ [5, 7], ]
    Example 2: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] starts = [0, 1] ends = [-1, 1000] result = [ [2, 3, 4], ]

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=5)
    data, starts, ends = values[: 3]
    axes  = values[3] if len(values) > 3 else None
    steps = values[4] if len(values) > 4 else torch.ones_like(starts)
    if axes is not None: axes = axes.tolist()
    starts, ends, steps = starts.tolist(), ends.tolist(), steps.tolist()

    slice_args = list(zip(starts, ends, steps))
    if axes is not None and all([_ != 0 for _ in axes]):
        assert len(axes) == len(slice_args)
        new_axes = [x if x >= 0 else len(data.dim()) + x for x in axes]
        full_axes = [i for i in range(data.dim())]
        slice_args = [slice_args[new_axes.index(i)] if i in new_axes else (None,) for i in full_axes]

    # slice function 里面有些时候会出现 list
    # 虽然我们也不知道是为什么，但是我们可以强行展开 list
    for args in slice_args:
        for i in range(len(args)):
            if type(args[i]) in {list, tuple}:
                args[i] = args[i][0]
    slice_func = [slice(*args) for args in slice_args]

    if any([step < 0 for step in steps]):
        negative_axes = []
        for idx, step in enumerate(steps):
            if step < 0: negative_axes.append(idx)
            slice_func[idx] = slice(- slice_func[idx].start - 1, - slice_func[idx].stop + 1, -slice_func[idx].step)
        data = torch.flip(data, dims=axes)

    output = data[slice_func]
    return output

def Constant_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """A constant tensor. Exactly one of the two attributes, either value or
    sparse_value, must be specified.

    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Attributes
        sparse_value : sparse_tensor
            The value for the elements of the output tensor in sparse format.

        value : tensor
            The value for the elements of the output tensor.
    Inputs

    Outputs
        output : T
            Output tensor containing the same value of the provided tensor.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=0, max_num_of_input=0)
    return op.attributes['value']

def Unsqueeze_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Insert single-dimensional entries to the shape of an input tensor
    (data).

    Takes one required argument axes - which contains a list of dimension indices and
        this operator will insert a dimension of value 1 into the corresponding
        index of the output tensor (expanded).

    For example: Given an input tensor (data) of shape [3, 4, 5],
        then Unsqueeze(data, axes=[0, 4]) outputs a tensor (expanded)
        containing same data as data but with shape [1, 3, 4, 5, 1].

    The attribute axes should not contain any duplicate entries.
    It is an error if it contains duplicates.

    The rank of the output tensor (output_rank) is the rank of the input tensor (data)
        plus the number of values in axes.

    Each value in axes should be within the (inclusive) range [-output_rank , output_rank - 1].
    The order of values in axes does not matter and can come in any order.

    Attributes
        axes : list of ints (required)
            List of integers indicating the dimensions to be inserted.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(expanded).

    Inputs
        data : T
            Original tensor

    Outputs
        expanded : T
            Reshaped tensor with same data as input.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
        unsqueezing_tensor, axes = values
        ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[axes])
        axes = axes.tolist()
    else:
        ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        [unsqueezing_tensor] = values
        axes = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=True)

    if isinstance(axes, list):
        for squeezing_dim in sorted(axes, reverse=True):
            unsqueezing_tensor = torch.unsqueeze(unsqueezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        unsqueezing_tensor = torch.unsqueeze(unsqueezing_tensor, axes)
    else: raise TypeError(f'Parameter axes of operation {op.name} misunderstood, '
                          f'expect int value of list of int, while {type(axes)} was given.')
    return unsqueezing_tensor

def Squeeze_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Remove single-dimensional entries from the shape of a tensor. Takes an
    input axes with a list of axes to squeeze. If axes is not provided, all the
    single dimensions will be removed from the shape. If an axis is selected
    with shape entry not equal to one, an error is raised.

    Inputs (1 - 2)
        data (differentiable) : T
        Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
        List of integers indicating the dimensions to squeeze.
        Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

    Outputs
        squeezed (differentiable) : T
        Reshaped tensor with same data as input.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        squeezing_tensor = values[0]
        axes = [axis for axis in range(squeezing_tensor.ndim) if squeezing_tensor.shape[axis] == 1]

        if len(values) > 1:
            axes = values[1]
            ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[axes])
            axes = axes.tolist()
    else:
        ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        [squeezing_tensor], axes = values, GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=False, default=None)
    
    # common part
    if axes is None:
        axes = []
        shape = squeezing_tensor.shape
        for dim, s in enumerate(shape):
            if s == 1: axes.append(dim)
    if isinstance(axes, list):
        for squeezing_dim in sorted(axes, reverse=True):
            squeezing_tensor = torch.squeeze(squeezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        squeezing_tensor = torch.squeeze(squeezing_tensor, axes)
    else: raise TypeError(f'Parameter axes of operation {op.name} misunderstood, '
                          f'expect int value of list of int, while {type(axes)} was given.')
    return squeezing_tensor

def Gather_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Given data tensor of rank r >= 1, and indices tensor of rank q, gather
    entries of the axis dimension of data (by default outer-most one as axis=0)
    indexed by indices,

    and concatenates them in an output tensor of rank q + (r - 1).

    Attributes
        axis : int (default is 0)
        Which axis to gather on. Negative value means counting dimensions from the back.
        Accepted range is [-r, r-1] where r = rank(data).

    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q.
            All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
        Tensor of rank q + (r - 1).

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        Union[torch.Tensor, int, float]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=0, compulsive=False)
    input_data, indices = values
    array_idx = [indices if axis == i else slice(dim) for i, dim in enumerate(input_data.shape)]
    output = input_data[array_idx]
    return output

def Range_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Generate a tensor containing a sequence of numbers that begin at start
    and extends by increments of delta up to limit (exclusive).

    The number of elements in the output of range is computed as below-

    number_of_elements = max( ceil( (limit - start) / delta ) , 0 )

    The pseudocode determining the contents of the output is shown below-

    for(int i=0; i<number_of_elements; ++i)

        output[i] = start + (i * delta);

    Example 1 Inputs: start = 3, limit = 9, delta = 3 Output: [3, 6]

    Example 2 Inputs: start = 10, limit = 4, delta = -2 Output: [10, 8, 6]

    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Inputs
        start : T
            Scalar. First entry for the range of output values.

        limit : T
            Scalar. Exclusive upper limit for the range of output values.

        delta : T
            Scalar. Value to step by.

    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    start, limit, delta = values
    output = torch.arange(start, limit, delta)
    return output

def ConstantOfShape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Generate a tensor with given value and shape.

    Attributes
    value : tensor
    (Optional) The value of the output elements.Should be a one-element tensor.
        If not specified, it defaults to a tensor of value 0 and datatype float32

    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor.
            If empty tensor is given, the output would be a scalar. All values must be >= 0.

    Outputs
        output : T2
        Output tensor of shape specified by 'input'.If attribute 'value' is specified,
            the value and datatype of the output tensor is taken from 'value'. If attribute 'value' is not specified,
            the value in the output defaults to 0, and the datatype defaults to float32.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    value = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='value', compulsive=False, default=0.0)
    [shape], fill_value = values, convert_any_to_python_primary_type(value)
    output = torch.Tensor().new_full(
        size=shape.tolist(), fill_value=fill_value)
    if isinstance(fill_value, int): output = output.long()
    elif isinstance(fill_value, float): output = output.float()
    else: raise TypeError(f'Can not parse value type{type(value)}.')
    return output

def Expand_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Broadcast the input tensor following the given shape and the broadcast
    rule.

    The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
        Dimensions are right alignment;

    Two corresponding dimension must have the same value, or one of them is equal to 1.
    Also, this operator is similar to numpy.broadcast_to(input, shape),
        but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().

    It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
        or the shape.ndim < input.shape.ndim.

    Version
    This version of the operator has been available since version 8 of the default ONNX operator set.

    Inputs
        input : T
            Input tensor

        shape : tensor(int64)
            A 1-D tensor indicates the shape you want to expand to, following the broadcast rule

    Outputs
        output : T
            Output tensor

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    tensor, repeats = values
    return tensor * torch.ones(tuple(repeats.type(torch.int64).tolist())).type(torch.int64)

def Tile_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Constructs a tensor by tiling a given tensor. This is the same as
    function tile in Numpy, but no broadcast.

    For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

    Version
        This version of the operator has been available since version 6 of the default ONNX operator set.

    Inputs
        input : T
            Input tensor of any shape.

        repeats : T1
            1D int64 tensor of the same length as input's dimension number,
            includes numbers of repeated copies along input's dimensions.

    Outputs
        output : T
            Output tensor of the same dimension and type as tensor input.
            output_dim[i] = input_dim[i] * repeats[i]

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    input, repeats = values
    # the repeats parameter is a 1D tensor in onnx,
    # the tiles parameter is a scalar in caffe

    # caffe op attributes
    axis = op.attributes.get('axis', None)
    tiles = op.attributes.get('tiles', None)

    if axis is not None:
        repeats = [1 for _ in range(input.ndim)]
        repeats[axis] = tiles
    else:
        repeats = convert_any_to_python_primary_type(values[-1])
        if not isinstance(repeats, list): repeats = [repeats]
    assert input.ndim == len(repeats)
    output = input.repeat(tuple(repeats))
    return output

def Reshape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Reshape the input tensor similar to numpy.reshape. First input is the
    data tensor, second input is a shape tensor which specifies the output
    shape. It outputs the reshaped tensor.

    At most one dimension of the new shape can be -1.
    In this case, the value is inferred from the size of the tensor and the remaining dimensions.
    A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
    If 'allowzero' is set, and the new shape includes 0,
        the dimension will be set explicitly to zero (i.e. not taken from input tensor)

    Attributes
        allowzero : int (default is 0) (Not implemented)
        (Optional) By default, when any value in the 'shape' input is equal to zero
            the corresponding dimension value is copied from the input tensor dynamically.

        allowzero=1 indicates that if any value in the 'shape' input is set to zero,
            the zero value is honored, similar to NumPy.
    Inputs
        data (differentiable) : T
            An input tensor.

        shape (non-differentiable) : tensor(int64)
            Specified shape for output.

    Outputs
        reshaped (differentiable) : T
            Reshaped data.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    if 'allowzero' in op.attributes: raise NotImplemented('Not implemented yet.')
    data, shape = values
    return data.reshape(shape.tolist())

def Less_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Returns the tensor resulted from performing the less logical operation
    elementwise on the input tensors A and B (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    Version
    This version of the operator has been available since version 9 of the default ONNX operator set.

    Inputs
        A : T
            First input operand for the logical operator.

        B : T
            Second input operand for the logical operator.

    Outputs
        C : T1
            Result tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    operand_a, operand_b = values
    if operand_a.dim() >= operand_b.dim() or operand_a.shape > operand_b.shape:
        output = torch.lt(operand_a, operand_b)
    else: output = torch.gt(operand_a, operand_b)
    return output

def Where_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Return elements, either from X or Y, depending on condition. Where
    behaves like numpy.where with three parameters.

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    Inputs
        condition (non-differentiable) : B
        When True (nonzero), yield X, otherwise yield Y

        X (differentiable) : T
        values selected at indices where condition is True

        Y (differentiable) : T
        values selected at indices where condition is False

    Outputs
        output (differentiable) : T
        Tensor of shape equal to the broadcasted shape of condition, X, and Y.

    Args:
        op (Operation): [description]
        input_values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    condition, x, y = values
    output = torch.where(condition, x, y)
    return output

def Transpose_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Transpose the input tensor similar to numpy.transpose. For example, when
    perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
    will be (2, 1, 3).

    Attributes
        perm : list of ints
            A list of integers. By default, reverse the dimensions,
            otherwise permute the axes according to the values given.

    Inputs
        data (differentiable) : T
            An input tensor.

    Outputs
        transposed (differentiable) : T
            Transposed output.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    perm = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='perm', compulsive=True)
    [data] = values
    output = data.permute(perm)
    return output

def Clip_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Clip operator limits the given input within an interval. The interval is
    specified by the inputs 'min' and 'max'. They default to
    numeric_limits::lowest() and numeric_limits::max(), respectively.

    Inputs (1 - 3)
        input (differentiable) : T
            Input tensor whose elements to be clipped

        min (optional, non-differentiable) : T
            Minimum value, under which element is replaced by min.
            It must be a scalar(tensor of empty shape).

        max (optional, non-differentiable) : T
            Maximum value, above which element is replaced by max.
            It must be a scalar(tensor of empty shape).

    Outputs
        output (differentiable) : T
            Output tensor with clipped input elements

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    output = torch.clamp(*values)
    return output

def Flatten_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Flatten.

    Flattens the input tensor into a 2D matrix.
        If input tensor has shape (d_0, d_1, ... d_n)
        then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).

    Inputs
        input (differentiable) : T
        A tensor of rank >= axis.

    Outputs
        output (differentiable) : T
        A 2D tensor with the contents of the input tensor,
        with input dimensions up to axis flattened to the outer dimension of the output
        and remaining input dimensions flattened into the inner dimension of the output.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value], dim = values, GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', compulsive=False, default=1)

    shape = list(value.shape)
    new_shape = [1, -1] if dim == 0 else [reduce(operator.mul, shape[:dim], 1), -1]
    output = value.reshape(new_shape)
    return output

def NMS_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    from .default import _NMS_forward
    """
    Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
    Bounding boxes with score less than score_threshold are removed.
    Bounding box format is indicated by attribute center_point_box.

    Note that this algorithm is agnostic to where the origin is in the coordinate system and
    more generally is invariant to orthogonal transformations and translations of the coordinate system;

    thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm.
    The selected_indices output is a set of integers indexing into the input collection of
        bounding boxes representing the selected boxes.
    The bounding box coordinates corresponding to the selected indices
        can then be obtained using the Gather or GatherND operation.

    Attributes
        center_point_box : int (default is 0)
        Integer indicate the format of the box data.
        The default is 0. 0 - the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2)
            are the coordinates of any diagonal pair of box corners and the coordinates can be provided as normalized
            (i.e., lying in the interval [0, 1]) or absolute.
            Mostly used for TF models. 1 - the box data is supplied as
                [x_center, y_center, width, height]. Mostly used for Pytorch models.

    Inputs (2 - 5)
        boxes : tensor(float)
        An input tensor with shape [num_batches, spatial_dimension, 4].
        The single box data format is indicated by center_point_box.

        scores : tensor(float)
        An input tensor with shape [num_batches, num_classes, spatial_dimension]

        max_output_boxes_per_class (optional) : tensor(int64)
        Integer representing the maximum number of boxes to be selected per batch per class.
        It is a scalar. Default to 0, which means no output.

        iou_threshold (optional) : tensor(float)
        Float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
            It is scalar. Value range [0, 1]. Default to 0.

        score_threshold (optional) : tensor(float)
        Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
        selected indices from the boxes tensor. [num_selected_indices, 3],
            the selected index format is [batch_index, class_index, box_index].

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values, force_convert=True)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=5)
    bboxs, scores = values[: 2]
    bboxs = bboxs.to(ctx.executing_device)
    scores = scores.to(ctx.executing_device)
    return _NMS_forward(op=op, values=[bboxs, scores] + values[2: ]).to('cpu')

def Equal_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Returns the tensor resulted from performing the equal logical operation
    elementwise on the input tensors A and B (with Numpy-style broadcasting
    support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    Inputs
        A (non-differentiable) : T
        First input operand for the logical operator.

        B (non-differentiable) : T
        Second input operand for the logical operator.

    Outputs
        C (non-differentiable) : T1
        Result tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return torch.eq(a, b)

def ScatterND_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """OPSET 11: ScatterND takes three inputs data tensor of rank r >= 1,
    indices tensor of rank q >= 1,

        and updates tensor of rank q + r - indices.shape[-1] - 1.

    The output of the operation is produced by creating a copy of the input data,
    and then updating its value to values specified by updates at specific index positions specified by indices.
    Its output shape is the same as the shape of data. Note that indices should not have duplicate entries.
    That is, two or more updates for the same index-location is not supported.

    indices is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of indices.
    indices is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into data.
    Hence, k can be a value at most the rank of data. When k equals rank(data),
    each update entry specifies an update to a single element of the tensor.
    When k is less than rank(data) each update entry specifies an update to a slice of the tensor.

    updates is treated as a (q-1)-dimensional tensor of replacement-slice-values.
    Thus, the first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
    The remaining dimensions of updates correspond to the dimensions of the replacement-slice-values.
    Each replacement-slice-value is a (r-k) dimensional tensor, corresponding to the trailing (r-k) dimensions of data.
    Thus, the shape of updates must equal indices.shape[0:q-1] ++ data.shape[k:r-1],
    where ++ denotes the concatenation of shapes.

    Inputs

        data : T
            Tensor of rank r >= 1.

        indices : tensor(int64)
            Tensor of rank q >= 1.

        updates : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs

        output : T
            Tensor of rank r >= 1.

    Args:
        op ([type]): [description]
        values ([type]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    data, indices, updates = values
    # output = data.clone()
    # convert ot numpy for accelerating.
    output  = data.numpy()
    updates = updates.numpy()
    indices = indices.long().numpy()

    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[tuple(indices[idx])] = updates[idx]
    return convert_any_to_torch_tensor(output)

def TopK_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Retrieve the top-K largest or smallest elements along a specified axis.
    Given an input tensor of shape [a_1, a_2, ..., a_n, r] and integer argument
    k, return two outputs: -Value tensor of shape [a_1, a_2, ..., a_{axis-1},
    k, a_{axis+1}, ... a_n] which contains the values of the top k elements.

    along the specified axis - Index tensor of shape [a_1, a_2, ...,
    a_{axis-1}, k, a_{axis+1}, ... a_n] which contains the indices of the top k
    elements (original indices from the input tensor).

    If "largest" is 1 (the default value) then the k largest elements are returned.
    If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
    If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

    Given two equivalent values, this operator uses the indices along the axis as a tiebreaker.
    That is, the element with the lower index will appear first.

    Attributes
        axis : int (default is -1)
            Dimension on which to do the sort. Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(input).

        largest : int (default is 1)
            Whether to return the top-K largest or smallest elements.

        sorted : int (default is 1)
            Whether to return the elements in sorted order.

    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]

        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve

    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing top K values from the input tensor

        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
            containing the corresponding input tensor indices for the top K values.

    Args:
        op (Operation): [description]
        input_value ([type]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    largest = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='largest', default=1)
    sorted = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sorted', default=1)
    largest, sorted = bool(largest), bool(sorted)

    x, k = values
    k = convert_any_to_python_primary_type(k)
    values, indices = torch.topk(input=x, k=k, dim=axis, largest=largest, sorted=sorted)
    return values, indices

def ReduceMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Computes the max of the input tensor's element along the provided axes.
    The resulted tensor has the same rank as the input if keepdims equal 1. If
    keepdims equal 0, then the resulted tensor have the reduced dimension
    pruned.

    The above behavior is similar to numpy,
    with the exception that numpy default keepdims to False instead of True.

    Attributes
        axes : list of ints
            A list of integers, along which to reduce.
            The default is to reduce over all the dimensions of the input tensor.
            Accepted range is [-r, r-1] where r = rank(data).

        keepdims : int (default is 1)
            Keep the reduced dimension or not, default 1 mean keep reduced dimension.

    Inputs
        data : T
            An input tensor.

    Outputs
        reduced : T
            Reduced output tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    axes = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', default=None)
    keepdims = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='keepdim', default=1)
    [data], keepdims = values, bool(keepdims)

    if data.numel() == 0: return data
    if axes is None:
        # The default is to reduce over all the dimensions of the input tensor
        reduced = data.max()
        if keepdims: reduced = reduced.reshape([1] * data.dim())
        return reduced

    for axis in axes:
        data = torch.max(data, keepdim=True, dim=axis)
    if keepdims == False: return torch.squeeze(data)
    return data

def Sqrt_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Square root takes one input data (Tensor) and produces one output data (Tensor)
        where the square root is, y = x^0.5, is applied to the tensor elementwise.
        If x is negative, then it will return NaN.

    Inputs
        X (differentiable) : T
            Input tensor

    Outputs
        Y (differentiable) : T
            Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.sqrt(x)

def Log_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Calculates the natural log of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor

    Outputs
        output (differentiable) : T
            The natural log of the input tensor computed element-wise

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    return torch.log(input)

def Floor_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Floor takes one input data (Tensor) and produces one output data (Tensor)
        where the floor is, y = floor(x), is applied to the tensor elementwise.

    Inputs
        X (non-differentiable) : T
            Input tensor

    Outputs
        Y (non-differentiable) : T
            Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return x.floor()

def Ceil_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Ceil takes one input data (Tensor) and produces one output data (Tensor)
    where the ceil is, y = ceil(x), is applied to the tensor elementwise.

    Inputs
        X (non-differentiable) : T
            Input tensor

    Outputs
        Y (non-differentiable) : T
            Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return x.ceil()

def RoiAlign_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Region of Interest (RoI) align operation described in the Mask R-CNN
    paper. RoiAlign consumes an input tensor X and region of interests (rois)
    to apply pooling across each RoI; it produces a 4-D tensor of shape
    (num_rois, C, output_height, output_width).

    RoiAlign is proposed to avoid the misalignment by removing quantizations while
        converting from original image into feature map and from feature map into RoI feature;

    in each ROI bin, the value of the sampled locations are computed directly through bilinear interpolation.

    Attributes
        coordinate_transformation_mode : string (default is half_pixel)
            Allowed values are 'half_pixel' and 'output_half_pixel'.
            Use the value 'half_pixel' to pixel shift the input coordinates by -0.5 (the recommended behavior).
            Use the value 'output_half_pixel' to omit the pixel shift for the input
            (use this for a backward-compatible behavior).

        mode : string (default is avg)
            The pooling method. Two modes are supported: 'avg' and 'max'. Default is 'avg'.

        output_height : int (default is 1)
            default 1; Pooled output Y's height.

        output_width : int (default is 1)
            default 1; Pooled output Y's width.

        sampling_ratio : int (default is 0)
            Number of sampling points in the interpolation grid used to compute
                the output value of each pooled output bin. If > 0,

            then exactly sampling_ratio x sampling_ratio grid points are used.
            If == 0, then an adaptive number of grid points are used
            (computed as ceil(roi_width / output_width), and likewise for height). Default is 0.

        spatial_scale : float (default is 1.0)
            Multiplicative spatial scale factor to translate ROI coordinates from their
                input spatial scale to the scale used when pooling, i.e.,
            spatial scale of the input feature map X relative to the input image. E.g.; default is 1.0f.

    Inputs
        X : T1
            Input data tensor from the previous operator;
            4-D feature map of shape (N, C, H, W),
            where N is the batch size, C is the number of channels,
            and H and W are the height and the width of the data.

        rois : T1
            RoIs (Regions of Interest) to pool over;
            rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...].
            The RoIs' coordinates are in the coordinate system of the input image.
            Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.

        batch_indices : T2
            1-D tensor of shape (num_rois,) with each element denoting
                the index of the corresponding image in the batch.

    Outputs
        Y : T1
            RoI pooled output, 4-D tensor of shape
            (num_rois, C, output_height, output_width).
            The r-th batch element Y[r-1] is a pooled feature map
            corresponding to the r-th RoI X[r-1].

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    from torchvision.ops import roi_align as torch_roi_align
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    x, rois, batch_indices = values

    output_height  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_height', default=1)
    output_width   = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_width', default=1)
    sampling_ratio = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sampling_ratio', default=0)
    spatial_scale  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='spatial_scale', default=1.0)

    if rois.shape[1] == 5: boxes = rois
    elif rois.shape[1] == 4: boxes = [rois]

    output_size = (output_height, output_width)
    output = torch_roi_align(
        x, boxes, output_size, spatial_scale, sampling_ratio)
    return output

def PPQDeviceSwitch_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return value.to('cpu')

def Exp_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Calculates the exponential of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor

    Outputs
        output (differentiable) : T
            The exponential of the input tensor computed element-wise

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    return torch.exp(input)

def Softmax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """The operator computes the normalized exponential values for the given
    input:

    Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

    The "axis" attribute indicates the dimension along which Softmax will be performed.
    The output tensor has the same shape and contains the Softmax values of the corresponding input.

    Attributes
        axis : int (default is -1)
            Describes the dimension Softmax will be performed on.
            Negative value means counting dimensions from the back.
            Accepted range is [-r, r-1] where r = rank(input).

    Inputs
        input (differentiable) : T
        The input tensor of rank >= axis.

    Outputs
        output (differentiable) : T
        The output values with the same shape as the input tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    output = F.softmax(input, axis)
    return output

def Sigmoid_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Sigmoid takes one input data (Tensor) and produces one output data (Tensor) where the sigmoid function,
        y = 1 / (1 + exp(-x)), is applied to the tensor elementwise.


    Inputs
        X (differentiable) : T
        Input tensor

    Outputs
        Y (differentiable) : T
        Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    return torch.sigmoid(values[0])

def Greater_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Returns the tensor resulted from performing the greater logical
    operation elementwise on the input tensors A and B (with Numpy-style
    broadcasting support).

    This operator supports multidirectional (i.e., Numpy-style) broadcasting; for more details please check the doc.

    Inputs
        A (non-differentiable) : T
        First input operand for the logical operator.

        B (non-differentiable) : T
        Second input operand for the logical operator.

    Outputs
        C (non-differentiable) : T1
        Result tensor.

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    if a.dim() >= b.dim() or a.shape > b.shape:
        c = torch.gt(a, b)
    else:
        c = torch.lt(a, b)
    return c

def Not_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Returns the negation of the input tensor element-wise.

    Inputs
        X (non-differentiable) : T
        Input tensor

    Outputs
        Y (non-differentiable) : T
        Output tensor

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    output = ~ values[0]
    return output

def MMCVRoiAlign_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    from mmcv.ops import roi_align as mmcv_roi_align

    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)

    data, rois = values[: 2]
    mode = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='mode', default='avg')
    aligned = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='aligned', default=True)
    output_height = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_height', default=1)
    output_width = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_width', default=1)
    sampling_ratio = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sampling_ratio', default=0)
    spatial_scale = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='spatial_scale', default=1.0)

    output_size = (output_height, output_width)
    if rois.shape[0] == 0:
        # TODO ??? WHY here got a 14
        output = torch.empty([0, data.shape[1], 14, 14])
    else:
        output = mmcv_roi_align(
            data, rois, output_size, spatial_scale, sampling_ratio, mode, aligned)
    return output

def Split_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Split a tensor into a list of tensors, along the specified 'axis'.
    Lengths of the parts can be specified using argument 'split'. Otherwise,
    the tensor is split to equal sized parts.

    Attributes
        axis : int (default is 0)
        Which axis to split on. A negative value means counting dimensions from the back.
        Accepted range is [-rank, rank-1] where r = rank(input).

        split : list of ints
        length of each output. Values should be >= 0.

    Inputs
        input : T
        The tensor to split

    Outputs (1 - ∞)
        outputs (variadic) : T
        One or more outputs forming list of tensors after splitting

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=0)
    split = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='split', compulsive=True)

    [input] = values
    output = torch.split(input, split, axis)
    return output

def GreaterOrEqual_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a >= b

def LessOrEqual_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a <= b

def ReduceSum_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if dim is None:
        #  The default is to reduce over all the dimensions of the input tensor
        output = torch.sum(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output = torch.sum(input_value, dim=dim[0], keepdim=keepdim)
    return output


def ScatterElements_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data, indices, updates = values
    dim = op.attributes.get('axis', 0)
    # Negative indices
    indices[indices < 0] += input_data.shape[dim]
    output = input_data.scatter(dim, indices, updates)
    return output


def Onehot_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Produces a one-hot tensor based on inputs. The locations represented by
    the index values in the 'indices' input tensor will have 'on_value' and the
    other locations will have 'off_value' in the output tensor,

    where 'on_value' and 'off_value' are specified as part of required input argument 'values',
    which is a two-element tensor of format [off_value, on_value].

    The rank of the output tensor will be one greater than the rank of the input tensor.
    The additional dimension is for one-hot representation. The additional dimension will be inserted at the position specified by 'axis'.
    If 'axis' is not specified then then additional dimension will be inserted as the innermost dimension,
    i.e. axis=-1. The size of the additional dimension is specified by required scalar input 'depth'.

    The type of the output tensor is the same as the type of the 'values' input. Any entries in the 'indices'
    input tensor with values outside the range [-depth, depth-1] will result in one-hot representation
    with all 'off_value' values in the output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
    Version
    This version of the operator has been available since version 11 of the default ONNX operator set.

    Attributes
    axis : int (default is -1)
    (Optional) Axis along which one-hot representation in added. Default: axis=-1. axis=-1 means that
        the additional dimension will be inserted as the innermost/last dimension in the output tensor.
    Negative value means counting dimensions from the back. Accepted range is [-r-1, r] where r = rank(indices).

    Inputs
    indices (non-differentiable) : T1
        Input tensor containing indices. Any entries in the 'indices' input tensor with values outside the range [-depth, depth-1]
            will result in one-hot representation with all 'off_value' values in the output tensor.In case 'indices' is of non-integer type,
            the values will be casted to int64 before use.

    depth (non-differentiable) : T2
        Scalar specifying the number of classes in one-hot tensor.
        This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor.
            The values in the 'indices' input tensor are expected to be in the range [-depth, depth-1].
            In case 'depth' is of non-integer type, it will be casted to int64 before use.

    values (non-differentiable) : T3
        Rank 1 tensor containing exactly two elements,
        in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor,
        and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor.

    Outputs
    output (non-differentiable) : T3
        Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1.
        The data type for the elements of the output tensor is the same as the type of input 'values' is used.
    """
    # implementation from https://github.com/ToriML/onnx2pytorch/blob/master/onnx2pytorch/operations/onehot.py
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    indices, depth, values = values

    off_value, on_value = values
    out = F.one_hot(indices.to(int), depth.to(int).item())
    out = out * (on_value - off_value) + off_value

    rank = len(indices.shape)
    if axis < 0:
        axis += rank + 1
    if not rank == axis:  # permute only if dim not last dimension
        order = list(range(len(indices.shape)))
        order.insert(axis, -1)
        out = out.permute(order)
    return out


def Identity_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    return values[0]

from .default import Tan_forward, Tanh_forward
SOI_BACKEND_TABLE = {
    'Shape': Shape_forward,
    'Div': Div_forward,
    'Add': Add_forward,
    'And': And_forward,
    'Mul': Mul_forward,
    'Sub': Sub_forward,
    'Cast': Cast_forward,
    'Concat': Concat_forward,
    'Slice': Slice_forward,
    'Constant': Constant_forward,
    'Unsqueeze': Unsqueeze_forward,
    'Squeeze': Squeeze_forward,
    'Gather': Gather_forward,
    'Range' : Range_forward,
    'ConstantOfShape': ConstantOfShape_forward,
    'Expand': Expand_forward,
    'Tile': Tile_forward,
    'Reshape': Reshape_forward,
    'Less': Less_forward,
    'Where': Where_forward,
    'Transpose': Transpose_forward,
    'Clip': Clip_forward,
    'Flatten': Flatten_forward,
    'NonMaxSuppression': NMS_forward,
    'Equal': Equal_forward,
    'ScatterND': ScatterND_forward,
    'TopK': TopK_forward,
    'ReduceMax': ReduceMax_forward,
    'Sqrt': Sqrt_forward,
    'Log': Log_forward,
    'Floor': Floor_forward,
    'Ceil': Ceil_forward,
    'RoiAlign': RoiAlign_forward,
    'PPQDeviceSwitch': PPQDeviceSwitch_forward,
    'Exp': Exp_forward,
    'Softmax': Softmax_forward,
    'Sigmoid': Sigmoid_forward,
    'Greater': Greater_forward,
    'Not': Not_forward,
    'MMCVRoiAlign': MMCVRoiAlign_forward,
    'Split': Split_forward,
    'GreaterOrEqual': GreaterOrEqual_forward,
    'LessOrEqual': LessOrEqual_forward,
    'ReduceSum': ReduceSum_forward,
    'ScatterElements': ScatterElements_forward,
    'OneHot': Onehot_forward,
    'Identity': Identity_forward,
    'Tanh': Tanh_forward,
    'Tan': Tan_forward
}
