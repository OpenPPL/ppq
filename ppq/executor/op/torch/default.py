import logging
import operator
from functools import reduce
from typing import List

import numpy as np
from ppq.core import DataType, convert_any_to_python_primary_type
from ppq.IR import Operation
from ppq.utils import process_attribute

import torch
import torch.nn.functional as F

from .base import *

# Reference:
# onnx op: https://github.com/onnx/onnx/blob/master/docs/Operators.md
# torch func: https://pytorch.org/docs/stable/nn.functional.html

# TODO:
# Resize only support pytorch cases
__all__ = [
    'BatchNormalization_forward', 'Cast_forward', 'Clip_forward', 'Concat_forward',
    'Constant_forward', 'ConstantOfShape_forward', 'Conv_forward', 'Eltwise_forward', 'Equal_forward',
    'UnaryEltwise_forward', 'Expand_forward', 'Flatten_forward', 'Gather_forward', 'GatherND_forward', 'Gemm_forward',
    'Grid_sampler_forward', 'AveragePool_forward', 'Greater_forward', 'Less_forward', 'MatMul_forward',
    'MaxPool2d_forward', '_NMS_forward', 'NonZero_forward', 'Not_forward', 'Range_forward',
    'ReduceL2_forward', 'ReduceMax_forward', 'Reshape_forward', 'Resize_forward', 'ScatterElements_forward',
    'ScatterND_forward', 'Shape_forward', 'Slice_forward', 'Softmax_forward', 'Squeeze_forward', 'Tile_forward',
    'TopK_forward', 'Transpose_forward', 'Unsqueeze_forward', 'Where_forward', 'ReduceSum_forward', 'ArgMax_forward',
    'Split_forward', 'ReduceMean_forward', 'PRelu_forward', 'Pad_forward', 'LeakyRelu_forward', 'ConvTranspose_forward',
    'Sqrt_forward', 'Log_forward', 'Floor_forward', 'RoiAlign_forward', 'MMCVRoiAlign_forward', 'SpaceToDepth_forward',
    'DepthToSpace_forward', 'Tanh_forward', 'Pow_forward', 'Crop_forward', 'ChannelShuffle_forward',
    'InstanceNormalization_forward', 'Parameter_forward', 'Interp_forward', 'CaffeArgMax_forward',
    'DEFAULT_BACKEND_TABLE',
]
logger = logging.getLogger('PPQ')

def Conv_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    The convolution operator consumes an input tensor and a filter, and computes the output.

    Attributes
        auto_pad : string (default is NOTSET)
            auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, 
            which means explicit padding is used. 
            
            SAME_UPPER or SAME_LOWER mean pad the input so that 
                `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
            The padding is split between the two sides equally or almost equally 
                (depending on whether it is even or odd).
            In case the padding is an odd number, 
                the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.
    
        dilations : list of ints
            dilation value along each spatial axis of the filter. 
            If not present, the dilation defaults is 1 along each spatial axis.
    
        group : int (default is 1)
            number of groups input channels and output channels are divided into.
    
        kernel_shape : list of ints
            The shape of the convolution kernel. If not present, should be inferred from input W.
    
        pads : list of ints
            Padding for the beginning and ending along each spatial axis, 
            it can take any value greater than or equal to 0. 
            
            The value represent the number of pixels added to the beginning and end part of the corresponding axis.
            `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
            where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
            the number of pixels added at the end of axis `i`.
            
            This attribute cannot be used simultaneously with auto_pad attribute.
            If not present, the padding defaults to 0 along start and end of each spatial axis.
    
        strides : list of ints
            Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.
    
    Inputs (2 - 3)
        X (differentiable) : T
            Input data tensor from previous layer;
            has size (N x C x H x W), where N is the batch size,
            C is the number of channels, and H and W are the height and width.
            Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn).
            
            Optionally, if dimension denotation is in effect, 
            the operation expects input data tensor to arrive 
                with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
        
        W (differentiable) : T
            The weight tensor that will be used in the convolutions; 
            has size (M x C/group x kH x kW), where C is the number of channels,
            and kH and kW are the height and width of the kernel, 
            and M is the number of feature maps. For more than 2 dimensions, 
            the kernel shape will be (M x C/group x k1 x k2 x ... x kn),
                where (k1 x k2 x ... kn) is the dimension of the kernel.
            Optionally, if dimension denotation is in effect,
                the operation expects the weight tensor to arrive with the dimension
                denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...].
            
            Assuming zero based indices for the shape array,
            X.shape[1] == (W.shape[1] * group) == C and W.shape[0] mod G == 0.
            
            Or in other words FILTER_IN_CHANNEL multiplied by the number of groups should be 
            equal to DATA_CHANNEL and the number of feature maps M should be a multiple of the number of groups G.
        
        B (optional, differentiable) : T
            Optional 1D bias to be added to the convolution, has size of M.
    
    Outputs
        Y (differentiable) : T
            Output data tensor that contains the result of the convolution.
            The output dimensions are functions of the kernel size, stride size, and pad lengths.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:])
    
    groups    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    padding   = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    
    x, w = values[: 2]
    b = values[2] if len(values) > 2 else None
    
    # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
    if isinstance(padding, list) and len(padding) == 4:
        p_left, p_right, p_top, p_bottom = padding[1], padding[3], padding[0], padding[2]
        # torch does not support padding contains 4 value, there is a fix of it.
        if p_left == p_right and p_top == p_bottom:
            padding = [p_top, p_left]
        else:
            x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
            padding = 0
    
    output = F.conv2d(input=x, weight=w, bias=b, groups=groups, padding=padding, 
                      dilation=dilation, stride=stride)
    return output


def ConvTranspose_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    
    The convolution transpose operator consumes an input tensor and a filter, and computes the output.

    If the pads parameter is provided the shape of the output is calculated via the following equation:

        output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
        output_shape can also be explicitly specified in which case pads values are 
            auto generated using these equations:

    total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]

    If (auto_pads == SAME_UPPER): 
        pads[start_i] = total_padding[i]/2; 
        pads[end_i] = total_padding[i] - (total_padding[i]/2)
    Else: 
        pads[start_i] = total_padding[i] - (total_padding[i]/2); 
        pads[end_i] = (total_padding[i]/2).

    Attributes
        auto_pad : string (default is NOTSET)
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET,
            which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that 
            `output_shape[i] = input_shape[i] * strides[i]` for each axis `i`. 
        The padding is split between the two sides equally or almost equally 
            (depending on whether it is even or odd). 
        In case the padding is an odd number, 
            the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

        dilations : list of ints
        dilation value along each spatial axis of the filter. 
            If not present, the dilation defaults to 1 along each spatial axis.

        group : int (default is 1)
        number of groups input channels and output channels are divided into.

        kernel_shape : list of ints
        The shape of the convolution kernel. If not present, should be inferred from input W.

        output_padding : list of ints
        Additional elements added to the side with higher coordinate indices in the output. 
            Each padding value in "output_padding" must be less than the corresponding stride/dilation dimension. 
            By default, this attribute is a zero vector. 
        Note that this attribute doesn't directly affect the computed output values. 
            It only controls the selection of the computed values, 
            so changing this attribute only adds or removes output elements. 
        If "output_shape" is explicitly provided, 
            "output_padding" does not contribute additional size to "output_shape" 
            but participates in the computation of the needed padding amount. 
            This is also called adjs or adjustment in some frameworks.

        output_shape : list of ints
        The shape of the output can be explicitly set which will cause pads values to be auto generated. 
        If output_shape is specified pads values are ignored. See doc for details for equations to generate pads

        pads : list of ints
        Padding for the beginning and ending along each spatial axis, 
            it can take any value greater than or equal to 0. 
        The value represent the number of pixels added to the beginning and end part of the corresponding axis. 
        `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], 
            where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, 
            the number of pixels added at the end of axis `i`. 
        This attribute cannot be used simultaneously with auto_pad attribute. 
        If not present, the padding defaults to 0 along start and end of each spatial axis.

        strides : list of ints
        Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

    Inputs (2 - 3)
        X (differentiable) : T
        Input data tensor from previous layer; has size (N x C x H x W), 
            where N is the batch size, C is the number of channels, and H and W are the height and width. 
            Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)
        
        W (differentiable) : T
        The weight tensor that will be used in the convolutions; has size (C x M/group x kH x kW), 
        where C is the number of channels, and kH and kW are the height and width of the kernel, 
            and M is the number of feature maps. 
        For more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn), 
            where (k1 x k2 x ... x kn) is the dimension of the kernel. 
        The number of channels in the output should be equal to 
            W.shape[1] * group (assuming zero based indices of the shape array)
        
        B (optional, differentiable) : T
        Optional 1D bias to be added to the convolution, has size of M.

    Outputs
        Y (differentiable) : T
        Output data tensor that contains the result of the convolution. 
        The output dimensions are functions of the kernel size, stride size, pad lengths and group count. 
        The number of channels in the output should be equal to 
            W.shape[1] * group (assuming zero based indices of the shape array)
        
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double)
        Constrain input and output types to float tensors.
    """
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:], 'ConvTranspose')
    
    groups    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    padding   = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    output_padding = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_padding', default=0)

    x, w = values[:2]
    b = values[2] if len(values) > 2 else None

    # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
    if isinstance(padding, list) and len(padding) == 4:
        p_left, p_right, p_top, p_bottom = padding[1], padding[3], padding[0], padding[2]
        # torch does not support padding contains 4 value, there is a fix of it.
        if p_left == p_right and p_top == p_bottom:
            padding = [p_top, p_left]
        else:
            x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
            padding = 0

    output = F.conv_transpose2d(
        input=x, weight=w, bias=b, groups=groups, padding=padding, 
        dilation=dilation, stride=stride, output_padding=output_padding)
    return output


def MaxPool2d_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])
    
    [input_value] = values
    padding   = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))

    # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
    if isinstance(padding, list) and len(padding) == 4:
        p_left, p_right, p_top, p_bottom = padding[1], padding[3], padding[0], padding[2]
        # torch does not support padding contains 4 value, there is a fix of it.
        if p_left == p_right and p_top == p_bottom:
            padding = [p_top, p_left]
        else:
            input_value = F.pad(input_value, pad=[p_left, p_right, p_top, p_bottom])
            padding = 0
    
    if op.type == 'GlobalMaxPool': kernel_size = input_value.size()[-2:]
    else: kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    
    output = F.max_pool2d(input_value, kernel_size=kernel_size,
                          padding=padding, dilation=dilation, 
                          stride=stride, ceil_mode=ceil_mode)
    return output


# noinspection PyCallingNonCallable
def BatchNormalization_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=5, max_num_of_input=5)
    input_data, weight, bias, running_mean, running_var = values

    # Default momentum in pytorch is 0.1 while onnx is 0.9, caffe is 0.999
    op_attr = {'eps': op.attributes.get(
        'epsilon', 1e-05), 'momentum': 1 - op.attributes.get('momentum', 0.9)}
    output = F.batch_norm(input_data, running_mean,
                          running_var, weight=weight, bias=bias, **op_attr)
    return output


def Mul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Performs element-wise binary multiplication (with Numpy-style broadcasting support).

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    multiplicand, multiplier = values
    return multiplicand * multiplier


def Add_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Performs element-wise binary addition (with Numpy-style broadcasting support).

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a + b


def Eltwise_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    if op.type == 'Add':
        assert len(values) == 2
        output = torch.add(*values).float()
    elif op.type == 'Sub':
        assert len(values) == 2
        output = torch.sub(*values).float()
    elif op.type == 'Mul':
        assert len(values) == 2
        output = torch.mul(*values).float()
    elif op.type == 'Div':
        assert len(values) == 2
        version = torch.__version__
        if version < '1.5.0' or version >= '1.7.0':
            output = torch.div(*values)
        else:
            if values[0].dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                output = torch.floor_divide(*values)
            else:
                output = torch.div(*values)
    elif op.type == 'Max':
        output = torch.max(*values)
    elif op.type == 'Min':
        output = torch.min(*values)
    else:
        logger.warning('Not Eltwise op, return input as output')
        output = values
    return output


# TODO: shape might contain 0, needs better solution
def Reshape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Reshape the input tensor similar to numpy.reshape. 
    First input is the data tensor, second input is a shape tensor which specifies the output shape. 
    It outputs the reshaped tensor. 
    
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
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[None, values[-1]])
    if 'allowzero' in op.attributes: raise NotImplemented('Not implemented yet.')
    data, shape = values
    shape = [shape[i] if shape[i] != 0 else data.shape[i] for i in range(len(shape))]
    return data.reshape(shape)


def AveragePool_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])
    
    [input_value] = values
    padding   = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    
    # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
    if isinstance(padding, list) and len(padding) == 4:
        p_left, p_right, p_top, p_bottom = padding[1], padding[3], padding[0], padding[2]
        # torch does not support padding contains 4 value, there is a fix of it.
        if p_left == p_right and p_top == p_bottom:
            padding = [p_top, p_left]
        else:
            input_value = F.pad(input_value, pad=[p_left, p_right, p_top, p_bottom])
            padding = 0

    if op.type == 'GlobalAveragePool': kernel_size = input_value.size()[-2:]
    else: kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)
    
    output = F.avg_pool2d(input_value, kernel_size=kernel_size,
                          padding=padding, stride=stride, ceil_mode=ceil_mode)
    return output


def ArgMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axis', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.argmax(input_value, dim=dim, keepdim=keepdim)
    return output


def Transpose_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Transpose the input tensor similar to numpy.transpose. 
    For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    perm = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='perm', compulsive=True)
    [data] = values
    output = data.permute(perm)
    return output


def Concat_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Concatenate a list of tensors into a single tensor. 
    All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
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


def Constant_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    A constant tensor. Exactly one of the two attributes, 
        either value or sparse_value, must be specified.

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


def Tile_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Constructs a tensor by tiling a given tensor. 
    This is the same as function tile in Numpy, but no broadcast.
    
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


def Squeeze_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Remove single-dimensional entries from the shape of a tensor. 
    Takes an input axes with a list of axes to squeeze. 
    If axes is not provided, all the single dimensions will be removed from the shape. 
    If an axis is selected with shape entry not equal to one, an error is raised.

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
    [squeezing_tensor], axes = values, GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axes', compulsive=True)
    if isinstance(axes, list): 
        for squeezing_dim in sorted(axes, reverse=True):
            squeezing_tensor = torch.squeeze(squeezing_tensor, squeezing_dim)
    elif isinstance(axes, int):
        squeezing_tensor = torch.squeeze(squeezing_tensor, axes)
    else: raise TypeError(f'Parameter axes of operation {op.name} misunderstood, '
                          f'expect int value of list of int, while {type(axes)} was given.')
    return squeezing_tensor


def Unsqueeze_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Insert single-dimensional entries to the shape of an input tensor (data). 
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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values)
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


def Gather_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_data, indices = values
    axis = op.attributes.get('axis', 0)
    if op.type == 'Gather':
        array_idx = [indices if axis == i else slice(
            dim) for i, dim in enumerate(input_data.shape)]
        output = input_data[array_idx]
    elif op.type == 'GatherElements':
        output = torch.gather(input_data, axis, indices)
    else:
        logger.warning('Not Gather op, return input as output')
        output = values
    return output


def GatherND_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    # y[i_1, ..., i_b, j_1, ..., j_n, k_1, ..., k_m] =
    #        x[i_1, ...,i_b, *idx[i_1, ..., i_b, j_1, ...,j_n, :], k_1,...,k_m]
    input_data, indices = values
    batch_dims = op.attributes.get('batch_dims', 0)
    data_rank = len(input_data.shape)
    assert indices.shape[-1] <= data_rank

    num_i = batch_dims
    # num_j = len(indices.shape) - num_i - 1
    num_k = len(input_data.shape) - num_i - indices.shape[-1]
    num_idx = indices.shape[-1]

    shape_i = indices.shape[:num_i]
    shape_j = indices.shape[num_i:-1]
    shape_k = input_data.shape[num_i + num_idx:]
    shape_idx = input_data.shape[num_i:num_i + num_idx]

    # indices reshape
    reshaped_indices = indices.reshape(
        *shape_i, -1, num_idx)  # shape [i_1, ..., i_b, J, 1]
    # indices tensordot, expand the last dim in indices
    strides = torch.tensor(
        [reduce(operator.mul, shape_idx[i + 1:], 1) for i in range(num_idx)], device=input_data.device,
        dtype=torch.float)
    merged_indices = torch.tensordot(
        reshaped_indices.float(), strides, 1)  # shape [i_1, ..., i_b, J]

    # indices expand
    expanded_indices = merged_indices.reshape(*merged_indices.shape, *([1] * num_k)).expand(
        *merged_indices.shape, *shape_k).long()

    # reshape input
    reshaped_input = input_data.reshape(*shape_i, -1, *shape_k)
    output = reshaped_input.gather(batch_dims, expanded_indices)

    # reshaped output
    reshaped_output = output.reshape(*shape_i, *shape_j, *shape_k)
    return output


def Greater_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_a, input_b = values
    if input_a.dim() >= input_b.dim() or input_a.shape > input_b.shape:
        output = torch.gt(input_a, input_b)
    else:
        output = torch.lt(input_b, input_a)
    return output


def Less_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_a, input_b = values
    if input_a.dim() >= input_b.dim() or input_a.shape > input_b.shape:
        output = torch.lt(input_a, input_b)
    else:
        output = torch.gt(input_b, input_a)
    return output


def Cast_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    new_data_type = DataType.to_torch(op.attributes['to'])
    output = input_value.to(dtype=new_data_type)
    return output


def ConstantOfShape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Generate a tensor with given value and shape.

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    value = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='value', compulsive=False, default=0.0)
    [shape], fill_value = values, convert_any_to_python_primary_type(value)
    output = torch.Tensor().new_full(
        size=shape.tolist(), fill_value=fill_value)
    if isinstance(fill_value, int): output = output.long()
    elif isinstance(fill_value, float): output = output.float()
    else: raise TypeError(f'Can not parse value type{type(value)}.')
    return output.to(ctx.executing_device)


def UnaryEltwise_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    if op.type == 'Exp':
        output = torch.exp(input_value)
    elif op.type == 'Sigmoid':
        output = torch.sigmoid(input_value)
    elif op.type == 'Relu':
        output = F.relu(input_value)
    else:
        logger.warning('Not UnaryEltwise op, return input as output')
        output = input_value
    return output


def NonZero_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    output = torch.nonzero(input_value, as_tuple=True)
    output = torch.stack(output)
    return output


def Clip_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    if len(values) == 1:
        values.append(op.attributes.get('min', float('-inf')))
        values.append(op.attributes.get('max', float('+inf')))
    output = torch.clip(*values)
    return output


def Slice_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Produces a slice of the input tensor along multiple axes. 
    Similar to numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html 
    Slices uses starts, ends, axes and steps inputs to specify the start 
        and end dimension and step for each axis in the list of axes, 
    
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
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[None] + values[1: ])
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


def Interp_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_data = values[0]
    mode = op.attributes.get('mode', 'nearest')
    # onnx resize 'linear' model include N-linear interpolate for N-D tensor
    linear_mode_map = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}

    # Caffe Interp preprocess
    # caffe resize by shrink_factor/zoom_factor
    input_shape = input_data.shape
    align_corners = False if not op.attributes.get('align_corners') else True
    zoom_factor = op.attributes.get('zoom_factor', 1)
    shrink_factor = op.attributes.get('shrink_factor', 1)
    pad_beg = op.attributes.get('pad_beg', 0)
    pad_end = op.attributes.get('pad_end', 0)

    height_in_eff = input_shape[-2] + pad_beg + pad_end
    width_in_eff = input_shape[-1] + pad_beg + pad_end
    height_out = height_in_eff
    width_out = width_in_eff
    if zoom_factor != 1:
        height_out = height_in_eff + (height_in_eff - 1) * (zoom_factor - 1)
        width_out = width_in_eff + (width_in_eff - 1) * (zoom_factor - 1)
    if shrink_factor != 1:
        height_out = (height_in_eff - 1) // shrink_factor + 1
        width_out = (width_in_eff - 1) // shrink_factor + 1
    if bool(op.attributes.get('height', None)):
        height_out = op.attributes.get('height')
        width_out = op.attributes.get('width')
    # PPL3 use second input define the shape
    if len(values) == 2:
        height_out, width_out = values[1].shape[-2:]

    sizes = list(input_shape[:2]) + [height_out, width_out]
    # the sizes in onnx is 4-D while in pytorch is 2-D
    # check the dim.0 & dim.1 is equal, then remain dim.2 and dim.3
    scales = None
    assert (sizes[:2] == list(input_data.shape[:2]))
    sizes = sizes[2:]
    mode = linear_mode_map[len(sizes)] if mode == 'linear' else mode

    output = F.interpolate(input_data, sizes, scales, mode, align_corners)
    return output


def Resize_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_data = values[0]
    # Not used roi
    # roi  = input_value[1] if len(input_value) > 1 else None
    scales = values[2] if len(values) > 2 else None
    sizes = values[-1].tolist() if len(values) == 4 else None
    mode = op.attributes.get('mode', 'nearest')
    if mode == 'cubic':
        mode = 'bicubic'
    # onnx resize 'linear' model include N-linear interpolate for N-D tensor
    linear_mode_map = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}

    # If 'size' is specified, then set scales to empty data (zero shape) in this operator's input list.
    if sizes is None or len(sizes) == 0:
        sizes = None
        if scales.numel() == 1:
            scales = scales.item()
        else:
            assert scales.numel() % 2 == 0
            scales = scales[-2].cpu().numpy().tolist()
    else:
        # the sizes in onnx is 4-D while in pytorch is 2-D
        # check the dim.0 & dim.1 is equal, then remain dim.2 and dim.3
        scales = None
        assert (sizes[:2] == list(input_data.shape[:2]))
        sizes = sizes[2:]
        mode = linear_mode_map[len(sizes)] if mode == 'linear' else mode

        if mode == 'cubic':
            logger.warning('Only support bicubic now')
            assert (len(sizes[2:]) == 2)
            mode = 'bicubic'

    trans_mode = op.attributes.get(
        'coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        output = F.interpolate(input_data, sizes, scales,
                               mode, align_corners=True)
    else:
        output = F.interpolate(input_data, sizes, scales, mode)
    return output


def _NMS_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
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
    import mmcv.ops

    boxes, scores = values[:2]
    max_output_boxes_per_class = values[2].item() if len(values) > 2 else 0
    iou_threshold = values[3].item() if len(values) > 3 else 0
    score_threshold = values[4].item() if len(values) > 4 else 0
    center_point_box = op.attributes.get('center_point_box', 0)

    batch, num_classes = boxes.shape[0], scores.shape[1]
    output = []
    for i in range(batch):
        sub_boxes = boxes[i]
        sub_scores = scores[i]
        if center_point_box:
            sub_boxes = torch.stack((
                sub_boxes[:, 0] - sub_boxes[:, 2] / 2,
                sub_boxes[:, 1] - sub_boxes[:, 3] / 2,
                sub_boxes[:, 0] + sub_boxes[:, 2] / 2,
                sub_boxes[:, 1] + sub_boxes[:, 3] / 2,
            ), dim=1)
        for j in range(num_classes):
            # If Given retinanet has 39w2k boxes, GPU run out of memory, move to cpu
            # sub_boxes.cpu(), sub_scores[j].cpu(), sub_scores[j][keep].cpu()
            # output.append(keep.to(device)) move back to gpu

            # In mmcv.ops.nms, boxes should given in the order of (x_min, y_min, x_max, y_max)
            '''
            # Strange speed, Revert back to original implementation.
            # May lead to error if boxes are not in the order given above

            sorted_boxes = torch.tensor([[min(item[0], item[2]), min(item[1], item[3]), max(item[0], item[2]),
                                          max(item[1], item[3])] for item in sub_boxes], device=device)
            keep = mmcv.ops.nms(sorted_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            '''
            keep = mmcv.ops.nms(sub_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            # keep = nms(sub_boxes, sub_scores[j], iou_threshold)[1]
            keep = keep[sub_scores[j][keep] > score_threshold]  # TODO: check GT or GE
            keep = keep[:max_output_boxes_per_class]
            keep = torch.stack((torch.full_like(keep, i), torch.full_like(keep, j), keep), dim=1)
            output.append(keep)
    output = torch.cat(output)
    return output.to('cpu')


def ReduceMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    else:
        if dim is None:
            #  The default is to reduce over all the dimensions of the input tensor
            output = torch.max(input_value)
            if keepdim:
                output = output.reshape([1] * input_value.dim())
        else:
            output, _ = torch.max(input_value, dim=dim[0], keepdim=keepdim)
    return output


def ReduceMean_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if len(input_value) == 0:
        output = input_value
    else:
        if dim is None:
            #  The default is to reduce over all the dimensions of the input tensor
            output = torch.mean(input_value)
            if keepdim:
                output = output.reshape([1] * input_value.dim())
        else:
            output = torch.mean(input_value, dim=dim, keepdim=keepdim)
    return output


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


def Shape_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.

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


def TopK_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Retrieve the top-K largest or smallest elements along a specified axis. 
    Given an input tensor of shape [a_1, a_2, ..., a_n, r] and integer argument k, 
    return two outputs: -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis -
    Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
    which contains the indices of the top k elements (original indices from the input tensor).

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
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[None, values[-1]])
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    largest = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='largest', default=1)
    sorted = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='sorted', default=1)
    largest, sorted = bool(largest), bool(sorted)

    x, k = values
    k = convert_any_to_python_primary_type(k)
    values, indices = torch.topk(input=x, k=k, dim=axis, largest=largest, sorted=sorted)
    return values.to('cpu'), indices.to('cpu')


def Expand_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    """
    Broadcast the input tensor following the given shape and the broadcast rule. 
    The broadcast rule is similar to numpy.array(input) * numpy.ones(shape): 
        Dimensions are right alignment; 
        Two corresponding dimension must have the same value, or one of them is equal to 1. 
    
    Also, this operator is similar to numpy.broadcast_to(input, shape), 
    but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size(). 
    It is possible that the output.shape is not equal to shape, 
    when some dimensions in shape is equal to 1, or the shape.ndim < input.shape.ndim.

    Inputs
        input (differentiable) : T
        Input tensor

        shape (non-differentiable) : tensor(int64)
        A 1-D tensor indicates the shape you want to expand to, following the broadcast rule

    Outputs
        output (differentiable) : T
        Output tensor

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    tensor, shape = values
    # TODO torch.ones 将会引起设备间同步
    output = tensor * torch.ones(tuple(shape.int().tolist()), dtype=tensor.dtype, device=tensor.device)
    return output


def Equal_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_a, input_b = values
    output = torch.eq(input_a, input_b)
    return output


def Flatten_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    [input_value] = values
    dim = op.attributes.get('axis', 1)
    shape = list(input_value.shape)
    new_shape = [
        1, -1] if dim == 0 else [reduce(operator.mul, shape[:dim], 1), -1]
    output = input_value.reshape(new_shape)
    return output


def Range_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    start, limit, delta = values
    output = torch.arange(start, limit, delta, device=ctx.executing_device)
    return output


def Where_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    condition, x, y = values
    output = torch.where(condition, x, y)
    return output


def ScatterElements_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data, indices, updates = values
    dim = op.attributes.get('axis', 0)
    # Negative indices
    indices[indices < 0] += input_data.shape[dim]
    output = input_data.scatter(dim, indices, updates)
    return output


def ScatterND_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    OPSET 11:
    ScatterND takes three inputs data tensor of rank r >= 1, indices tensor of rank q >= 1, 
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
    # device = kwargs['ctx'].executing_device
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=[values[0], None, values[-1]])
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[None, values[1], None])

    data, indices, updates = values
    output = data.clone()
    indices = indices.long()
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        output[tuple(indices[idx])] = updates[idx]
    return output


def Split_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    axis = op.attributes.get('axis', 0)
    split = op.attributes.get('split', 0)
    output_num = len(op.outputs)
    [input_value] = values
    output = torch.split(input_value, split, axis)
    return output


def Gemm_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)
    
    A, B = values[:2]
    if len(A.shape) != 2:
        # reshape A from [n, c, h, w] to [n, chw]
        A = A.reshape(A.shape[0], -1)
    C = values[2] if len(values) > 2 else 0
    alpha = op.attributes.get('alpha', 1.0)
    beta = op.attributes.get('beta', 1.0)
    transA = op.attributes.get('transA', 0)
    transB = op.attributes.get('transB', 0)
    A = A.transpose(0, 1) if transA else A
    B = B.transpose(0, 1) if transB else B

    output = alpha * torch.matmul(A, B) + beta * C

    return output

def MatMul_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    output = torch.matmul(values[0], values[1])
    return output

def Softmax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    The operator computes the normalized exponential values for the given input:

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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input] = values
    axis = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='axis', default=-1)
    output = F.softmax(input, axis)
    return output


def ReduceL2_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    [input_value] = values
    axis = op.attributes['axes']
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.norm(input_value, dim=axis, keepdim=keepdim)
    if axis is None and keepdim:
        output = output.reshape([1] * input_value.dim())
    return output


def PRelu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data, weight = values
    output = F.prelu(input_data, weight)
    return output


def LeakyRelu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    [input_data] = values
    alpha = op.attributes.get('alpha', 0.01)
    output = F.leaky_relu(input_data, alpha)
    return output


def Pad_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    mode = op.attributes.get('mode', 'constant')
    input_data = values[0]
    pads = values[1] if len(values) > 1 else op.attributes['pads']
    if isinstance(pads, torch.Tensor):
        assert pads.device.type == 'cpu', 'Oops'
        pads = pads.tolist()

    if len(pads) == 2:
        pads = [0, 0, pads[0], pads[1]]
    elif len(pads) == 4:
        pads = [pads[1], pads[3], pads[0], pads[2]]
    elif len(pads) == 8:  # inception v3, i don't kown if the order is correct.
        pads = [pads[2], pads[3], pads[6], pads[7]]

    if mode == 'constant':
        constant_value = values[-1] if len(values) == 3 else 0
        output = F.pad(input_data, pads, mode, constant_value)
    elif mode == 'reflect':
        output = F.pad(input_data, pads, mode)
    elif mode == 'edge':
        output = F.pad(input_data, pads, 'replicate')
    else:
        raise TypeError(f'Unsupported mode {mode} in Pad op')
    return output


def Sqrt_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    output = torch.sqrt(input_data)
    return output


def Log_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    output = torch.log(input_data)
    return output


def Floor_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    output = torch.floor(input_data)
    return output


def RoiAlign_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    from torchvision.ops import roi_align as torch_roi_align

    input_data = values[0]
    rois = values[1]

    # TODO: batch_indices and mode may be used in the future
    # batch_indices = input_value[2]
    # Attention: for op.attributes.mode=max, mmcv is different from onnx
    # mode = op.attributes.get('mode', 'avg')
    output_height = op.attributes.get('output_height', 1)
    output_width = op.attributes.get('output_width', 1)
    sampling_ratio = op.attributes.get('sampling_ratio', 0)
    spatial_scale = op.attributes.get('spatial_scale', 1.0)

    if isinstance(rois, torch.Tensor):
        if rois.shape[1] == 5:
            boxes = rois
        elif rois.shape[1] == 4:
            boxes = [rois]
        else:
            raise ValueError(f'Unsupported rois shape {rois.shape}')
    else:
        raise TypeError('Unsupported rois type')

    output_size = (output_height, output_width)
    output = torch_roi_align(
        input_data, boxes, output_size, spatial_scale, sampling_ratio)
    return output


def MMCVRoiAlign_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    from mmcv.ops import roi_align as mmcv_roi_align
    # ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)

    data, rois = values
    rois = FORCE_CONVERT_DEVICE(rois, device=data.device)

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


def SpaceToDepth_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    # SubpixelDown in caffe
    input_data = values[0]
    downsample = op.attributes.get('blocksize', 1)

    # F.pixel_unshuffle needs torch >= 1.8.0
    # TODO: Only aligned with pytorch, Caffe has yet to be aligned
    output = F.pixel_unshuffle(input_data, downsample)
    return output


def DepthToSpace_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    # SubpixelUp in caffe
    input_data = values[0]
    upsample = op.attributes.get('blocksize', 1)
    mode = op.attributes.get('mode', 'DCR')
    if mode == 'DCR':
        output = F.pixel_shuffle(input_data, upsample)
    else:  # mode == 'CRD'
        output = None
        raise ValueError
    return output


def Scale_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    # if just one bottom is given, scale is a learned parameter
    assert len(values) >= 2
    input_data = values[0]
    scale = values[1]

    bias_term = op.attributes.get('bias_term', False)
    axis = op.attributes.get('axis', 1)
    # num_axes is ignored unless just one bottom is given
    # num_axes is determined by the number of axes by the second bottom
    # num_axes = scale.dim()

    scale_shape = list(scale.shape)
    # get scale shape [1, c, 1, 1]
    for i in range(axis):
        scale_shape.insert(0, 1)
    for i in range(input_data.dim() - scale.dim() - axis):
        scale_shape.append(1)

    scale = scale.reshape(scale_shape)
    scale = scale.expand_as(input_data)

    if bias_term:
        bias = values[2]
        bias = bias.reshape(scale_shape).expand_as(input_data)
        output = input_data * scale + bias
    else:
        output = input_data * scale

    return output


def Tanh_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    output = torch.tanh(input_data)
    return output


def Pow_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    power = op.attributes.get('power', 1)
    scale = op.attributes.get('scale', 1)
    shift = op.attributes.get('shift', 0)

    if len(values) == 2:  # In onnx op, power is the second input
        power = values[1]
        scale, shift = 1.0, 0.0

    output = torch.pow(input_data * scale + shift, power)
    return output


def Crop_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    logger.error('Not support Crop op yet.')
    exit(-1)
    pass
    # TODO
    '''
    ! Not implemented yet
    input_data = input_value[0]
    shape = input_value[1].shape

    crop_param = op.attributes.get('crop_param', None)
    axis = crop_param.axis

    offset = crop_param.offset

    assert axis == len(shape) - len(offset)

    output =
    '''


def ChannelShuffle_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    group = op.attributes.get('group', 1)
    assert input_data.shape[1] % group == 0

    n, c, h, w = input_data.shape
    input_data = input_data.view(n, group, c // group, h, w)
    input_data = input_data.permute(0, 2, 1, 3, 4)
    output = input_data.contiguous().view(n, c, h, w)
    return output


def InstanceNormalization_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    num_features = op.attributes.get('num_features', 1)
    eps = op.attributes.get('eps', 1e-5)
    affine = op.attributes.get('affine', False)

    # TODO:
    # For len(input_value)==5 not support yet, inputs = [data weight bais mean and var].

    if len(values) == 3:  # caffe op and onnx op
        input_data, weight, bias = values
        running_mean, running_var = None, None

    elif len(values) == 1:
        input_data = values[0]
        running_mean, running_var, weight, bias = None, None, None, None
    else:
        raise ValueError(
            f'The number of input data in InstanceNom is {len(values)}')

    if affine:  # caffe op check
        assert num_features == input_data.shape[1]

    output = F.instance_norm(input_data, running_mean,
                             running_var, weight, bias, eps=eps)
    return output


def Parameter_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    m = op.attributes.get('m', -1)
    n = op.attributes.get('n', -1)

    # TODO
    # other parameters of paramter op are not added yet
    output = input_data.reshape(m, n)
    return output


def CaffeArgMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    assert len(values) == 1
    input_data = values[0]
    dim = op.attributes.get('axis', None)
    output = input_data.topk(op.attributes.get('top_k', 1), dim=dim)
    return output  # only return maxval now

    # TODO
    '''
    # There are some gaps between ppl-argmax and standard argmax
    # If out_max_val is true, produce pairs (argmax, maxval)
    output = (output[1], output[0])
    if op.attributes.get('out_max_val', False):
        _update_output(op, output, 1)
    else:
        _update_output(op, output[0], 1)
    '''


def Grid_sampler_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    # ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    # domain is mmcv
    value, grid = values
    output = F.grid_sample(value, grid, align_corners=False)
    return output


def Not_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return ~value


def HardSigmoid_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return F.hardsigmoid(value)


def HardSwish_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [value] = values
    return F.hardswish(value)


def Neg_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Neg takes one input data (Tensor) and produces one output data (Tensor) 
    where each element flipped sign, y = -x, is applied to the tensor elementwise.

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
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return -x


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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    return torch.sigmoid(values[0])


def PPQDeviceSwitch_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [value] = values
    return value.to(ctx.executing_device)



DEFAULT_BACKEND_TABLE = {
    'Add': Add_forward,
    'ArgMax': ArgMax_forward,
    'AveragePool': AveragePool_forward,
    'BatchNormalization': BatchNormalization_forward,
    'Cast': Cast_forward,
    'Clip': Clip_forward,
    'Concat': Concat_forward,
    'Constant': Constant_forward,
    'ConstantOfShape': ConstantOfShape_forward,
    'Conv': Conv_forward,
    'ConvTranspose': ConvTranspose_forward,
    'Div': Eltwise_forward,
    'Equal': Equal_forward,
    'Exp': UnaryEltwise_forward,
    'Expand': Expand_forward,
    'Flatten': Flatten_forward,
    'Gather': Gather_forward,
    'GatherElements': Gather_forward,
    'GatherND': GatherND_forward,
    'Gemm': Gemm_forward,
    'grid_sampler': Grid_sampler_forward,
    'GlobalAveragePool': AveragePool_forward,
    'GlobalMaxPool': MaxPool2d_forward,
    'Greater': Greater_forward,
    'LeakyRelu': LeakyRelu_forward,
    'Less': Less_forward,
    'MatMul': MatMul_forward,
    'Max': Eltwise_forward,
    'MaxPool': MaxPool2d_forward,
    'Min': Eltwise_forward,
    'Mul': Mul_forward,
    'NonMaxSuppression': _NMS_forward,
    'NonZero': NonZero_forward,
    'Not': Not_forward,
    'Pad': Pad_forward,
    'PRelu': PRelu_forward,
    'Range': Range_forward,
    'ReduceL2': ReduceL2_forward,
    'ReduceMax': ReduceMax_forward,
    'ReduceMean': ReduceMean_forward,
    'ReduceSum': ReduceSum_forward,
    'Relu': UnaryEltwise_forward,
    'Reshape': Reshape_forward,
    'Resize': Resize_forward,
    'ScatterElements': ScatterElements_forward,
    'ScatterND': ScatterND_forward,
    'Shape': Shape_forward,
    'Sigmoid': UnaryEltwise_forward,
    'Slice': Slice_forward,
    'Softmax': Softmax_forward,
    'Split': Split_forward,
    'Squeeze': Squeeze_forward,
    'Sub': Eltwise_forward,
    'Tile': Tile_forward,
    'TopK': TopK_forward,
    'Transpose': Transpose_forward,
    'Unsqueeze': Unsqueeze_forward,
    'Where': Where_forward,
    'Sqrt': Sqrt_forward,
    'Log': Log_forward,
    'Floor': Floor_forward,
    'RoiAlign': RoiAlign_forward,
    'MMCVRoiAlign': MMCVRoiAlign_forward,
    'SpaceToDepth': SpaceToDepth_forward,
    'DepthToSpace': DepthToSpace_forward,
    'Scale': Scale_forward,  # caffe op
    'Tanh': Tanh_forward,
    'Pow': Pow_forward,
    'Crop': Crop_forward,  # caffe op
    'ChannelShuffle': ChannelShuffle_forward,  # caffe op
    'InstanceNormalization': InstanceNormalization_forward,
    'Parameter': Parameter_forward,  # caffe op
    'Interp': Interp_forward,  # caffe op
    'CaffeArgMax': CaffeArgMax_forward,  # caffe op
    'HardSigmoid': HardSigmoid_forward,
    'HardSwish': HardSwish_forward,
    'Neg': Neg_forward,
    'PPQDeviceSwitch': PPQDeviceSwitch_forward
}
