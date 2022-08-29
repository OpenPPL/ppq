import operator
from functools import reduce
from typing import List, Tuple

import numpy as np
from ppq.core import DataType, convert_any_to_python_primary_type
from ppq.IR import Operation
from ppq.core.common import GRU_FLATTEN_WEIGHT_ATTRIB, LSTM_FLATTEN_WEIGHT_ATTRIB
from ppq.log import NaiveLogger
from ppq.utils import process_attribute

import torch
import torch.nn.functional as F
from torch import _VF

from .base import *

# Reference:
# onnx op: https://github.com/onnx/onnx/blob/master/docs/Operators.md
# torch func: https://pytorch.org/docs/stable/nn.functional.html

logger = NaiveLogger.get_logger('PPQ')


def convert_onnx_pads_to_torch(onnx_pads: List[int], mode: str=None) -> List[int]:
    # Convert padding from onnx format to torch format
    # onnx format: [x1_begin, x2_begin, ... , x1_end, x2_end, ...]
    # torch format [xn_begin, xn_end, ... , x2_begin, x2_end, x1_begin, x1_end]
    if onnx_pads is None: return 0
    if isinstance(onnx_pads, int): return onnx_pads

    # check pads dimension
    if mode is not None:
        if mode == '1d': assert len(onnx_pads) == 2, (
            f'1d Operation needs 2-d padding value, while your padding value is {onnx_pads}')
        elif mode == '2d': assert len(onnx_pads) == 4, (
            f'2d Operation needs 4-d padding value, while your padding value is {onnx_pads}')
        elif mode == '3d': assert len(onnx_pads) == 6, (
            f'3d Operation needs 6-d padding value, while your padding value is {onnx_pads}')

    middle = len(onnx_pads) // 2
    onnx_pad_begin, onnx_pad_end = onnx_pads[:middle], onnx_pads[middle:]
    onnx_pad_begin, onnx_pad_end = onnx_pad_begin[::-1], onnx_pad_end[::-1]
    
    torch_pads = []
    for begin, end in zip(onnx_pad_begin, onnx_pad_end):
        torch_pads.extend([begin, end])

    if mode is None: return torch_pads
    # check if we can merge torch pads
    if len(torch_pads) == 2:
        p1, p2 = torch_pads
        if p1 == p2: 
            torch_pads = [p1]
    if len(torch_pads) == 4:
        p1, p2, p3, p4 = torch_pads
        if p1==p2 and p3==p4: 
            torch_pads=[p1, p3]
    if len(torch_pads) == 6:
        p1, p2, p3, p4, p5, p6 = torch_pads
        if p1==p2 and p3==p4 and p5==p6: 
            torch_pads=[p1, p3, p5]
    return torch_pads


def Abs_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Absolute takes one input data (Tensor) and produces one output data (Tensor) 
    where the absolute is, y = abs(x), is applied to the tensor elementwise.

    Inputs
        X (differentiable) : T
            Input tensor
    
    Outputs
        Y (differentiable) : T
            Output tensor

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return x.abs()


def Conv_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """The convolution operator consumes an input tensor and a filter, and
    computes the output.

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

    groups    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='group', default=1)
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    auto_pad  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='auto_pad', default='NOTSET')

    x, w = values[: 2]
    b = values[2] if len(values) > 2 else None

    ndim = w.ndim

    # conv - 1d
    if ndim in {2, 3}:
        if auto_pad != 'NOTSET': raise NotImplementedError(f'auto_pad must be "NOTSET" with 1-d conv {op.name}')
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) == 2:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.conv1d(
            input=x, weight=w, bias=b, groups=groups, 
            padding=torch_pads, dilation=dilation, stride=stride)

    # conv - 2d
    elif ndim == 4:
        process_attribute(op.attributes, values[0].shape[2:], values[1].shape[2:])
        # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            # torch does not support padding contains 4 value, there is a fix of it.
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0

        output = F.conv2d(
            input=x, weight=w, bias=b, groups=groups, padding=onnx_pads,
            dilation=dilation, stride=stride)

    # conv - 3d
    elif ndim == 5:
        if auto_pad != 'NOTSET': raise NotImplementedError(f'auto_pad must be "NOTSET" with 3-d conv {op.name}')
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) == 6:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.conv3d(
            input=x, weight=w, bias=b, groups=groups, 
            padding=torch_pads, dilation=dilation, stride=stride)
    
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def ConvTranspose_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """The convolution transpose operator consumes an input tensor and a
    filter, and computes the output.

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
    onnx_pads = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=1)
    output_padding = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='output_padding', default=0)

    x, w = values[:2]
    b = values[2] if len(values) > 2 else None
    ndim = x.ndim

    # 2d conv transpose
    if ndim == 4:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='2d')
        if isinstance(torch_pads, list) and len(torch_pads) == 2:
            output = F.conv_transpose2d(
                input=x, weight=w, bias=b, groups=groups, padding=torch_pads,
                dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose2d(
                input=x, weight=w, bias=b, groups=groups, padding=0,
                dilation=dilation, stride=stride, output_padding=output_padding)

            p1, p2, p3, p4 = torch_pads
            _, _, h, w = output.shape
            output = output[:, :, 0 + p1: h - p2, 0 + p3: w - p4]

    # 1d conv transpose
    elif ndim in {2, 3}:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) == 1:
            output = F.conv_transpose1d(
                input=x, weight=w, bias=b, groups=groups, padding=torch_pads,
                dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose1d(
                input=x, weight=w, bias=b, groups=groups, padding=0,
                dilation=dilation, stride=stride, output_padding=output_padding)

            p1, p2 = torch_pads
            _, _, h = output.shape
            output = output[:, :, 0 + p1: h - p2]

    # 3d conv transpose
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) == 3:
            output = F.conv_transpose3d(
                input=x, weight=w, bias=b, groups=groups, padding=torch_pads,
                dilation=dilation, stride=stride, output_padding=output_padding)
        else:
            output = F.conv_transpose3d(
                input=x, weight=w, bias=b, groups=groups, padding=0,
                dilation=dilation, stride=stride, output_padding=output_padding)

            p1, p2, p3, p4, p5, p6 = torch_pads
            _, _, d, h, w = output.shape
            output = output[:, :, 0 + p1: d - p2, 0 + p3: h - p4, 0 + p5: w - p6]
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


def MaxPool2d_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    process_attribute(op.attributes, values[0].shape[2:])

    [x] = values
    onnx_pads   = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    dilation  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='dilations', default=1)
    stride    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    if op.type == 'GlobalMaxPool': kernel_size = x.size()[2:]
    else: kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)

    ndim = x.ndim
    # pool - 3d
    if ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.max_pool3d(
            x, kernel_size=kernel_size, padding=torch_pads,
            dilation=dilation, stride=stride, ceil_mode=ceil_mode)

    elif ndim == 4:
        # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            # torch does not support padding contains 4 value, there is a fix of it.
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=convert_onnx_pads_to_torch(onnx_pads), value=float("-inf"))
                onnx_pads = 0

        output = F.max_pool2d(
            x, kernel_size=kernel_size,
            padding=onnx_pads, dilation=dilation,
            stride=stride, ceil_mode=ceil_mode)

    elif ndim in {2, 3}:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.max_pool1d(
            x, kernel_size=kernel_size, padding=torch_pads,
            dilation=dilation, stride=stride, ceil_mode=ceil_mode)
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')
    return output


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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    multiplicand, multiplier = values
    return multiplicand * multiplier


def MultiHeadAttention_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Perform MultiHeadAttetion opr forward.

    Args:
        op (Operation): MultiHeadAttention
        values (List[torch.Tensor]): opr inputs
        ctx (TorchBackendContext, optional): Context. Defaults to None.

    Raises:
        NotImplementedError: In [Vit Paper](https://arxiv.org/abs/2010.11929), MultiHeadAttention inputs are actually the same tensor, we suppose that this would **not** be simplified.
        ValueError: MultiHeadAttention contains `embed_dim` and `num_heads`.

    Returns:
        list: opr output and internal result for quantization.
    """
    if len(values) != 11:
        raise NotImplementedError('Not implement simplified MultiHeadAttention')

    q_in,k_in,v_in,q_w,q_b,k_w,k_b,v_w,v_b,o_w,o_b = values
    embed_dim = op.attributes.get('embed_dim')
    num_heads = op.attributes.get('num_heads')

    if embed_dim is None or num_heads is None:
        raise ValueError('Cannot fetch embed_dim or num_heads')

    # setup parameters
    batch_size = q_in.shape[0]
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    xq = F.linear(q_in, q_w, q_b)
    xk = F.linear(k_in, k_w, k_b)
    xv = F.linear(v_in, v_w, v_b)
    
    B, N, _ = xq.shape
    
    q = xq.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    k = xk.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    v = xv.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

    energy = (q @ k.transpose(-2, -1)) * scale
    attn = energy.softmax(dim=-1)

    feat = (attn @ v).transpose(1, 2).reshape(batch_size, -1, embed_dim)
    out = F.linear(feat, o_w, o_b)
    
    return out


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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a + b


def And_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    a, b = values
    return a & b


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

    [x] = values
    onnx_pads    = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='pads', default=0)
    stride       = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='strides', default=None)
    ceil_mode    = bool(GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='ceil_mode', default=False))
    if op.type   == 'GlobalAveragePool': kernel_size = x.size()[2:]
    else: kernel_size = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='kernel_shape', compulsive=True)

    ndim = x.ndim
    # pool 1d
    if ndim == 3:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='1d')
        if isinstance(torch_pads, list) and len(torch_pads) != 1:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool1d(
            x, kernel_size=kernel_size, padding=torch_pads,
            stride=stride, ceil_mode=ceil_mode)
        return output
    # pool 2d
    if ndim == 4:
        # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
        if isinstance(onnx_pads, list) and len(onnx_pads) == 4:
            p_left, p_right, p_top, p_bottom = onnx_pads[1], onnx_pads[3], onnx_pads[0], onnx_pads[2]
            # torch does not support padding contains 4 value, there is a fix of it.
            if p_left == p_right and p_top == p_bottom:
                onnx_pads = [p_top, p_left]
            else:
                x = F.pad(x, pad=[p_left, p_right, p_top, p_bottom])
                onnx_pads = 0

        output = F.avg_pool2d(
            x, kernel_size=kernel_size,
            padding=onnx_pads, stride=stride, ceil_mode=ceil_mode)
        return output
    # pool 3d
    elif ndim == 5:
        torch_pads = convert_onnx_pads_to_torch(onnx_pads=onnx_pads, mode='3d')
        if isinstance(torch_pads, list) and len(torch_pads) != 3:
            x = F.pad(x, torch_pads)
            torch_pads = 0
        output = F.avg_pool3d(
            x, kernel_size=kernel_size, padding=torch_pads,
            stride=stride, ceil_mode=ceil_mode)
        return output
    else:
        raise ValueError(f'Operation {op.name} is invalid, {ndim}-d input is not supported.')

    return output


def AdaptiveAvgPool2d_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_value, output_size = values
    output = F.adaptive_avg_pool2d(input_value, output_size)
    return output


def ArgMax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    dim = op.attributes.get('axis', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    output = torch.argmax(input_value, dim=dim, keepdim=keepdim)
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
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    perm = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='perm', compulsive=True)
    [data] = values
    output = data.permute(perm)
    return output


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
    # we acquire axes in different ways according to opset
    # only opset=11 or opset=13 supported
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


def Gather_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    input_data, indices = values
    indices = indices.long()
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

def Gelu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    [input_value] = values
    return F.gelu(input_value)

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
    output = torch.clamp(*values)
    return output


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
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[None] + values[1: ])
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=5)
    data, starts, ends = values[: 3]
    axes  = values[3] if len(values) > 3 else None
    steps = values[4] if len(values) > 4 else torch.ones_like(starts)
    if axes is not None: axes = axes.tolist()
    starts, ends, steps = starts.tolist(), ends.tolist(), steps.tolist()

    slices, flip_dims = {}, []
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step
        slices[axis] = slice(start, end, step)

    pos_axes_slices = list(slices.get(a, slice(None, None)) for a in range(max(axes) + 1))
    neg_axes_slices = list(slices.get(a, slice(None, None)) for a in range(min(axes), 0))
    if neg_axes_slices: neg_axes_slices = [Ellipsis] + neg_axes_slices

    if flip_dims: data = torch.flip(data, dims=flip_dims)
    if pos_axes_slices: data = data[pos_axes_slices]
    if neg_axes_slices: data = data[neg_axes_slices]
    return data

    ''' Legacy implementation
    data, starts, ends = values[: 3]
    axes  = values[3] if len(values) > 3 else None
    steps = values[4] if len(values) > 4 else torch.ones_like(starts)
    if axes is not None: axes = axes.tolist()
    starts, ends, steps = starts.tolist(), ends.tolist(), steps.tolist()

    slice_args = list(zip(starts, ends, steps))
    if axes is not None and all([_ != 0 for _ in axes]):
        assert len(axes) == len(slice_args)
        new_axes = [data if data >= 0 else data.dim() + data for data in axes]
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
    '''


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
            scales = scales.cpu().tolist()
            if len(scales) == 2:
                # 大家相安无事，和平共处
                pass
            elif len(scales) == 4:
                if scales[:2] != [1, 1]:
                    raise NotImplementedError(
                        'Can not resize your image with current op, '
                        'cause 4-dimension resize is not implemented with pytorch.')
                scales = scales[2:]
            else:
                raise NotImplementedError(
                    'Can not resize your image with current op, '
                    f'cause {len(scales)}-dimension resize is not implemented with pytorch.')
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

    # PATCH 2022.04.22
    # ONNX DO NOT HAVE BILINEAR MODE, FOR 4D INPUT, WE OVERRIDE MODE TO BILINEAR
    if len(input_data.shape) == 4 and mode == 'linear':
        mode = 'bilinear'

    trans_mode = op.attributes.get(
        'coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        output = F.interpolate(input_data, sizes, scales,
                               mode, align_corners=True)
    else:
        output = F.interpolate(input_data, sizes, scales, mode)
    return output


def _NMS_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Filter out boxes that have high intersection-over-union (IOU) overlap
    with previously selected boxes. Bounding boxes with score less than
    score_threshold are removed. Bounding box format is indicated by attribute
    center_point_box.

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
            """# Strange speed, Revert back to original implementation.

            # May lead to error if boxes are not in the order given above

            sorted_boxes = torch.tensor([[min(item[0], item[2]), min(item[1], item[3]), max(item[0], item[2]),
                                          max(item[1], item[3])] for item in sub_boxes], device=device)
            keep = mmcv.ops.nms(sorted_boxes, sub_scores[j].contiguous(), iou_threshold)[1]
            """
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
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        input_value, dim = values[0], None
        if len(values) > 1:
            dim = values[1]
            ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[dim])
        keepdim, noop_with_empty_axes = bool(op.attributes.get('keepdims', 1)), op.attributes.get('noop_with_empty_axes', 0)

        if dim is None:
            if noop_with_empty_axes:
                return input_value
            else:
                output = torch.sum(input_value)
                if keepdim:
                    output = output.reshape([1] * input_value.dim())
                return output
        else:
            dim = dim.tolist()
            if isinstance(dim, int):
                dim = [dim]
            output = torch.sum(input_value, dim=dim, keepdim=keepdim)
            return output

    [input_value] = values
    dim = op.attributes.get('axes', None)
    keepdim = bool(op.attributes.get('keepdims', 1))
    if dim is None:
        #  The default is to reduce over all the dimensions of the input tensor
        output = torch.sum(input_value)
        if keepdim:
            output = output.reshape([1] * input_value.dim())
    else:
        output = torch.sum(input_value, dim=dim, keepdim=keepdim)
    return output


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
    """Broadcast the input tensor following the given shape and the broadcast
    rule.

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
    # device = kwargs['ctx'].executing_device
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=3)
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=[values[0], None, values[-1]])
    ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[None, values[1], None])

    data, indices, updates = values

    output = data.clone()
    ind_dim = indices.dim()
    # last dimension is a partial-index into data
    indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
    # update.shape = indices.shape[0:ind_dim-1] ++ data.shape[indices.shape[-1]:data.dim()-1]
    updates = updates.reshape((-1, *updates.shape[ind_dim - 1 :]))
    output[indices] = updates
    return output


def Split_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    if op.opset.onnx_opset_version() >= 13:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=2)
        input_value = values[0]
        axis = op.attributes.get('axis', 0)
        split = input_value.shape[axis] // len(op.outputs)
        if len(values) > 1:
            split = values[1]
            ASSERT_ALL_TENSORS_AT_CPU(op=op, values=[split])
            split = split.tolist()
    else:
        ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
        axis = op.attributes.get('axis', 0)
        split = op.attributes.get('split', 0)
        [input_value] = values
        if 'split' not in op.attributes:
            split = input_value.shape[axis] // len(op.outputs)
    outputs = torch.split(input_value, split, axis)
    return outputs


def Gemm_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=3)

    A, B = values[:2]

    # PATCH for ommited reshape before inner product in caffe
    if op.opset.is_caffe() and A.ndim > 2:
        axis = op.attributes.get('axis', 1)
        A = A.flatten(start_dim=axis)

    C = values[2] if len(values) > 2 else 0
    alpha  = op.attributes.get('alpha', 1.0)
    beta   = op.attributes.get('beta', 1.0)
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


def LayerNorm_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    if len(values) != 3:
        raise ValueError('Unsupported LayerNorm without affine')

    input_data, weight, bias = values
    eps = op.attributes.get('epsilon', 1e-5)
    normalized_shape = weight.shape

    output = F.layer_norm(input_data, normalized_shape, weight, bias, eps)
    return output


def Pad_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    mode = op.attributes.get('mode', 'constant')
    input_data = values[0]
    pads = values[1] if len(values) > 1 else op.attributes['pads']
    if isinstance(pads, torch.Tensor):
        assert pads.device.type == 'cpu', 'Oops'
        pads = pads.tolist()
    pads = convert_onnx_pads_to_torch(pads)

    if mode == 'constant':
        constant_value = values[-1] if len(values) == 3 else 0
        output = F.pad(input_data, pads, mode, constant_value)
    elif mode == 'reflect':
        output = input_data
        while len(pads) > 4:
            output = F.pad(input_data, pads[-4:], mode)
            pads   = pads[: -4]
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


def Mod_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=2, max_num_of_input=2)
    fmod = op.attributes.get('fmod', 0)
    if values[0].dtype in {torch.float, torch.float16, torch.float32, torch.float64}:
        assert fmod, 'fmod must equals to 1 when operands are floats'
    if fmod:
        output = torch.fmod(values[0], values[1])
    else:
        output = torch.remainder(values[0], values[1])
    return output


def Softplus_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    input_data = values[0]
    output = torch.log(torch.exp(input_data) + 1)
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
        # ??? i do kown why following is correct.
        output = F.pixel_shuffle(input_data, upsample)
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

def Tan_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs):
    input_data = values[0]
    output = torch.tan(input_data)
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
    # other parameters of parameter op are not added yet
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


def GRU_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Computes an one-layer GRU. This operator is usually supported via some
    custom implementation such as CuDNN.
    只支持 pytorch 导出来的 GRU 啊亲; 必须要 6 个输入 Variable
    Notations:
        X - input tensor
        z - update gate
        r - reset gate
        h - hidden gate
        t - time step (t-1 means previous time step)
        W[zrh] - W parameter weight matrix for update, reset, and hidden gates
        R[zrh] - R recurrence weight matrix for update, reset, and hidden gates
        Wb[zrh] - W bias vectors for update, reset, and hidden gates
        Rb[zrh] - R bias vectors for update, reset, and hidden gates
        WB[zrh] - W parameter weight matrix for backward update, reset, and hidden gates
        RB[zrh] - R recurrence weight matrix for backward update, reset, and hidden gates
        WBb[zrh] - W bias vectors for backward update, reset, and hidden gates
        RBb[zrh] - R bias vectors for backward update, reset, and hidden gates
        H - Hidden state
        num_directions - 2 if direction == bidirectional else 1
    Activation functions:
        Relu(x)                - max(0, x)
        Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
        Sigmoid(x)             - 1/(1 + e^{-x})
    (NOTE: Below are optional)
        Affine(x)              - alpha*x + beta
        LeakyRelu(x)           - x if x >= 0 else alpha * x
        ThresholdedRelu(x)     - x if x >= alpha else 0
        ScaledTanh(x)          - alpha*Tanh(beta*x)
        HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
        Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
        Softsign(x)            - x/(1 + |x|)
        Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh):
        - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
        - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
        - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
        - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
        - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    This operator has optional inputs/outputs. See the doc for more details about the representation of optional arguments.
    An empty string may be used in the place of an actual argument's name to indicate a missing argument.
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
    Version
    This version of the operator has been available since version 14 of the default ONNX operator set.
    Other versions of this operator: 1, 3, 7
    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions.
            The values are consumed in the order of activation functions,
            for example (f, g, h) in LSTM.
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu,
            the default alpha is 0.01.
        activation_beta : list of floats
            Optional scaling values used by some activation functions.
            The values are consumed in the order of activation functions,
            for example (f, g, h) in LSTM.
            Default values are the same as of corresponding ONNX operators.
        activations : list of strings
            A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates.
            The activation functions must be one of the activation functions specified above.
            Optional: See the equations for default if not specified.
        clip : float
            Cell clip threshold.
            Clipping bounds the elements of a tensor in the range of [-threshold, +threshold]
            and is applied to the input of activations. No clip if not specified.
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional.
            Must be one of forward (default), reverse, or bidirectional.
        hidden_size : int
            Number of neurons in the hidden layer
        layout : int (default is 0)
            The shape format of inputs X, initial_h and outputs Y, Y_h.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].
        linear_before_reset : int (default is 0)
            When computing the output of the hidden gate,
            apply the linear transformation before multiplying by the output of the reset gate.
    Inputs (3 - 6)
        X (differentiable) : T
            The input sequences packed (and potentially padded) into one 3-D tensor with the shape of
            `[seq_length, batch_size, input_size]`.
        W (differentiable) : T
            The weight tensor for the gates.
            Concatenation of `W[zrh]` and `WB[zrh]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 3*hidden_size, input_size]`.
        R (differentiable) : T
            The recurrence weight tensor.
            Concatenation of `R[zrh]` and `RB[zrh]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 3*hidden_size, hidden_size]`.
        B (optional, differentiable) : T
            The bias tensor for the gates.
            Concatenation of `[Wb[zrh], Rb[zrh]]` and `[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0.
            This tensor has shape `[num_directions, 6*hidden_size]`. Optional: If not specified - assumed to be 0
        sequence_lens (optional, non-differentiable) : T1
            Optional tensor specifying lengths of the sequences in a batch.
            If not specified - assumed all sequences in the batch to have length `seq_length`.
            It has shape `[batch_size]`.
        initial_h (optional, non-differentiable) : T
            Optional initial value of the hidden.
            If not specified - assumed to be 0.
            It has shape `[num_directions, batch_size, hidden_size]`.
    Outputs (0 - 2)
        Y (optional, differentiable) : T
            A tensor that concats all the intermediate output values of the hidden.
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
        Y_h (optional, differentiable) : T
            The last output value of the hidden.
            It has shape `[num_directions, batch_size, hidden_size]`.
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
        T1 : tensor(int32)
    Constrain seq_lens to integer tensor.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=6)
    # first 3 are mandatory input
    x, w, r   = values[: 3]
    b         = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len   = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    
    # sequence length will be dropped without warrning.
    # if seq_len is not None: raise NotImplementedError('PPQ do not support LSTM with explicite length.')
    
    # check attributes
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip             = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction        = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    linear_before_reset = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if linear_before_reset != 0: raise NotImplementedError('PPQ do not support LSTM with linear_before_reset != 1.')
    if activation_alpha is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')

    # flag
    bidirectional = (direction == 'bidirectional')
    has_bias      = (b is not None)

    # create flatten weights:
    if GRU_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([
            w[0][hidden_size * 1: hidden_size * 2],
            w[0][hidden_size * 0: hidden_size * 1],
            w[0][hidden_size * 2: hidden_size * 3]], dim=0).contiguous()
        forward_r = torch.cat([
            r[0][hidden_size * 1: hidden_size * 2],
            r[0][hidden_size * 0: hidden_size * 1],
            r[0][hidden_size * 2: hidden_size * 3]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([
                b[0, hidden_size * 1: hidden_size * 2],
                b[0, hidden_size * 0: hidden_size * 1],
                b[0, hidden_size * 2: hidden_size * 3]]).contiguous()
            forward_bias_2 = torch.cat([
                b[0, hidden_size * 4: hidden_size * 5],
                b[0, hidden_size * 3: hidden_size * 4],
                b[0, hidden_size * 5: hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([
                w[1][hidden_size * 1: hidden_size * 2],
                w[1][hidden_size * 0: hidden_size * 1],
                w[1][hidden_size * 2: hidden_size * 3]], dim=0).contiguous()
            reverse_r = torch.cat([
                r[1][hidden_size * 1: hidden_size * 2],
                r[1][hidden_size * 0: hidden_size * 1],
                r[1][hidden_size * 2: hidden_size * 3]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([
                    b[1, hidden_size * 1: hidden_size * 2],
                    b[1, hidden_size * 0: hidden_size * 1],
                    b[1, hidden_size * 2: hidden_size * 3]]).contiguous()
                reverse_bias_2 = torch.cat([
                    b[1, hidden_size * 4: hidden_size * 5],
                    b[1, hidden_size * 3: hidden_size * 4],
                    b[1, hidden_size * 5: hidden_size * 6]]).contiguous()

        flatten_weight = [forward_w, forward_r]
        if has_bias:                   flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:              flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias: flatten_weight = [
            forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(GRU_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(
            size=[s, x.shape[1], x.shape[2]], 
            device=x.device, dtype=torch.float32)

    result = _VF.gru(
        x,                                       # x
        initial_h,                               # initial hidden state
        op._detail[GRU_FLATTEN_WEIGHT_ATTRIB],   # flatten weights
        has_bias,                                # has bias
        1,                                       # num of layer
        0.0,                                     # dropout
        False,                                   # training flag
        bidirectional,                           # bidirectional
        False)                                   # batch first

    hidden_vector, last_state = result
    return hidden_vector.unsqueeze(1), last_state


def LSTM_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Computes an one-layer LSTM. This operator is usually supported via some
    custom implementation such as CuDNN.

    只支持 pytorch 导出来的 LSTM 啊亲; 必须要 7 个输入 Variable

    Computes an one-layer LSTM. This operator is usually supported via some custom implementation such as CuDNN.

    Notations:

    X - input tensor

    i - input gate

    o - output gate

    f - forget gate

    c - cell gate

    t - time step (t-1 means previous time step)

    W[iofc] - W parameter weight matrix for input, output, forget, and cell gates

    R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates

    Wb[iofc] - W bias vectors for input, output, forget, and cell gates

    Rb[iofc] - R bias vectors for input, output, forget, and cell gates

    P[iof] - P peephole weight vector for input, output, and forget gates

    WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates

    RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates

    WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates

    RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates

    PB[iof] - P peephole weight vector for backward input, output, and forget gates

    H - Hidden state

    num_directions - 2 if direction == bidirectional else 1

    Activation functions:

    Relu(x)                - max(0, x)

    Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

    Sigmoid(x)             - 1/(1 + e^{-x})

    (NOTE: Below are optional)

    Affine(x)              - alpha*x + beta

    LeakyRelu(x)           - x if x >= 0 else alpha * x

    ThresholdedRelu(x)     - x if x >= alpha else 0

    ScaledTanh(x)          - alpha*Tanh(beta*x)

    HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

    Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

    Softsign(x)            - x/(1 + |x|)

    Softplus(x)            - log(1 + e^x)
    Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

    - Ct = ft (.) Ct-1 + it (.) ct

    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

    - Ht = ot (.) h(Ct)
    This operator has optional inputs/outputs. 
    See the doc for more details about the representation of optional arguments. 
    An empty string may be used in the place of an actual argument's name to indicate a missing argument. 
    Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

    Version
    This version of the operator has been available since version 7 of the default ONNX operator set.

    Attributes
        activation_alpha : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
                for example (f, g, h) in LSTM. 
    
            Default values are the same as of corresponding ONNX operators.For example with LeakyRelu, the default alpha is 0.01.
    
        activation_beta : list of floats
            Optional scaling values used by some activation functions. 
            The values are consumed in the order of activation functions, 
            for example (f, g, h) in LSTM. 
            
            Default values are the same as of corresponding ONNX operators.
    
        activations : list of strings
            A list of 3 (or 6 if bidirectional) activation functions for input, 
            output, forget, cell, and hidden. 
            
            The activation functions must be one of the activation functions specified above. 
            Optional: See the equations for default if not specified.
    
        clip : float
            Cell clip threshold. Clipping bounds the elements of a tensor in the range of 
            [-threshold, +threshold] and is applied to the input of activations.
            No clip if not specified.
        
        direction : string (default is forward)
            Specify if the RNN is forward, reverse, or bidirectional. 
            Must be one of forward (default), reverse, or bidirectional.

        hidden_size : int
            Number of neurons in the hidden layer
    
        input_forget : int (default is 0)
            Couple the input and forget gates if 1.
    
    Inputs (3 - 8)
        X : T
            The input sequences packed (and potentially padded) into one 3-D tensor 
                with the shape of `[seq_length, batch_size, input_size]`.
   
        W : T
            The weight tensor for the gates. Concatenation of `W[iofc]` and `WB[iofc]` 
            (if bidirectional) along dimension 0. The tensor has shape `[num_directions, 4*hidden_size, input_size]`.
    
        R : T
            The recurrence weight tensor. Concatenation of `R[iofc]` and `RB[iofc]` (if bidirectional) along dimension 0. 
            This tensor has shape `[num_directions, 4*hidden_size, hidden_size]`.
    
        B (optional) : T
            The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, 
            and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. 
            
            This tensor has shape `[num_directions, 8*hidden_size]`. 
            Optional: If not specified - assumed to be 0.
    
        sequence_lens (optional) : T1
            Optional tensor specifying lengths of the sequences in a batch. 
            If not specified - assumed all sequences in the batch to have length `seq_length`. 
            It has shape `[batch_size]`.
        
        initial_h (optional) : T
            Optional initial value of the hidden. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        initial_c (optional) : T
            Optional initial value of the cell. 
            If not specified - assumed to be 0. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        P (optional) : T
            The weight tensor for peepholes.
            Concatenation of `P[iof]` and `PB[iof]` (if bidirectional) along dimension 0. 
            It has shape `[num_directions, 3*hidde_size]`. Optional: If not specified - assumed to be 0.
    
    Outputs (0 - 3)
        Y (optional) : T
            A tensor that concats all the intermediate output values of the hidden. 
            It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
    
        Y_h (optional) : T
            The last output value of the hidden. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    
        Y_c (optional) : T
            The last output value of the cell. 
            It has shape `[num_directions, batch_size, hidden_size]`.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=8)
    # first 3 are mandatory input
    x, w, r   = values[: 3]
    b         = GET_VALUE_FROM_INPUTS(values, 3)
    seq_len   = GET_VALUE_FROM_INPUTS(values, 4)
    initial_h = GET_VALUE_FROM_INPUTS(values, 5)
    initial_c = GET_VALUE_FROM_INPUTS(values, 6)
    p         = GET_VALUE_FROM_INPUTS(values, 7)
    if p is not None: raise NotImplementedError('PPQ do not support LSTM with peepholes.')
    
    # sequence length will be dropped without warrning.
    # if seq_len is not None: raise NotImplementedError('PPQ do not support LSTM with explicite length.')

    # check attributes
    activation_alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_alpha', default=None)
    activation_beta  = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activation_beta', default=None)
    activations      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='activations', default=None)
    clip             = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='clip', default=None)
    direction        = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='direction', default='forward')
    hidden_size      = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='hidden_size', compulsive=True)
    input_forget     = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='input_forget', default=0)
    layout           = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='layout', default=0)
    if layout != 0: raise NotImplementedError('PPQ do not support LSTM with layout != 1.')
    if activation_alpha is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activation_beta is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')
    if activations is not None: raise NotImplementedError('PPQ do not support LSTM with cutimized activation.')

    # flag
    bidirectional = (direction == 'bidirectional')
    has_bias      = (b is not None)

    if direction == 'reverse': raise NotImplementedError('GRU do not support reverse mode now.')

    # create flatten weights:
    if LSTM_FLATTEN_WEIGHT_ATTRIB not in op.attributes:
        forward_w = torch.cat([
            w[0][hidden_size * 0: hidden_size * 1],
            w[0][hidden_size * 2: hidden_size * 3],
            w[0][hidden_size * 3: hidden_size * 4],
            w[0][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
        forward_r = torch.cat([
            r[0][hidden_size * 0: hidden_size * 1],
            r[0][hidden_size * 2: hidden_size * 3],
            r[0][hidden_size * 3: hidden_size * 4],
            r[0][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
        if has_bias:
            forward_bias_1 = torch.cat([
                b[0, hidden_size * 0: hidden_size * 1],
                b[0, hidden_size * 2: hidden_size * 3],
                b[0, hidden_size * 3: hidden_size * 4],
                b[0, hidden_size * 1: hidden_size * 2]]).contiguous()
            forward_bias_2 = torch.cat([
                b[0, hidden_size * 4: hidden_size * 5],
                b[0, hidden_size * 6: hidden_size * 7],
                b[0, hidden_size * 7: hidden_size * 8],
                b[0, hidden_size * 5: hidden_size * 6]]).contiguous()
        if bidirectional == True:
            reverse_w = torch.cat([
                w[1][hidden_size * 0: hidden_size * 1],
                w[1][hidden_size * 2: hidden_size * 3],
                w[1][hidden_size * 3: hidden_size * 4],
                w[1][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
            reverse_r = torch.cat([
                r[1][hidden_size * 0: hidden_size * 1],
                r[1][hidden_size * 2: hidden_size * 3],
                r[1][hidden_size * 3: hidden_size * 4],
                r[1][hidden_size * 1: hidden_size * 2]], dim=0).contiguous()
            if has_bias:
                reverse_bias_1 = torch.cat([
                    b[1, hidden_size * 0: hidden_size * 1],
                    b[1, hidden_size * 2: hidden_size * 3],
                    b[1, hidden_size * 3: hidden_size * 4],
                    b[1, hidden_size * 1: hidden_size * 2]]).contiguous()
                reverse_bias_2 = torch.cat([
                    b[1, hidden_size * 4: hidden_size * 5],
                    b[1, hidden_size * 6: hidden_size * 7],
                    b[1, hidden_size * 7: hidden_size * 8],
                    b[1, hidden_size * 5: hidden_size * 6]]).contiguous()
        
        flatten_weight = [forward_w, forward_r]
        if has_bias:                   flatten_weight = [forward_w, forward_r, forward_bias_1, forward_bias_2]
        if bidirectional:              flatten_weight = [forward_w, forward_r, reverse_w, reverse_r]
        if bidirectional and has_bias: flatten_weight = [
            forward_w, forward_r, forward_bias_1, forward_bias_2, reverse_w, reverse_r, reverse_bias_1, reverse_bias_2]
        op.set_extension_attrib(LSTM_FLATTEN_WEIGHT_ATTRIB, flatten_weight)
    # end if
    
    s = 2 if bidirectional else 1
    if initial_h is None:
        initial_h = torch.zeros(
            size=[s, x.shape[1], hidden_size], 
            device=x.device, dtype=torch.float32)

    if initial_c is None:
        initial_c = torch.zeros(
            size=[s, x.shape[1], hidden_size], 
            device=x.device, dtype=torch.float32)

    result = _VF.lstm(
        x,                                       # x
        (initial_h, initial_c),                  # initial hidden state
        op._detail[LSTM_FLATTEN_WEIGHT_ATTRIB],  # flatten weights
        has_bias,                                # has bias
        1,                                       # num of layer
        0.0,                                     # dropout
        False,                                   # training flag
        bidirectional,                           # bidirectional
        False)                                   # batch first

    hs, h, c = result
    if bidirectional:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 2, hs.shape[-1] // 2))
        hs = hs.permute((0, 2, 1, 3))
    else:
        hs = hs.reshape((hs.shape[0], hs.shape[1], 1, hs.shape[-1]))
        hs = hs.permute((0, 2, 1, 3))
    return hs, h, c


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


def Identity_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    return values[0]


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


def Reciprocal_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Reciprocal takes one input data (Tensor) and produces one output data (Tensor) where the reciprocal is,
        y = 1/x, is applied to the tensor elementwise.

    Version
        This version of the operator has been available since version 13 of the default ONNX operator set.

    Inputs
        X (differentiable) : T Input tensor
    Outputs
        Y (differentiable) : T Output tensor

    Constrain input and output types to float tensors.
    """
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return 1 / x


def LogSoftmax_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
 
    x = Softmax_forward(op=op, values=values, ctx=ctx, kwargs=kwargs)
    x = Log_forward(op=op, values=[x], ctx=ctx, kwargs=kwargs)
    return x


def Sin_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Calculates the sine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The sine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.sin(x)


def Cos_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Calculates the cosine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The cosine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.cos(x)


def Cos_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """Calculates the cosine of the given input tensor, element-wise.

    Inputs
        input (differentiable) : T
            Input tensor
    
    Outputs
        output (differentiable) : T
            The cosine of the input tensor computed element-wise
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.cos(x)


def Sum_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Element-wise sum of each of the input tensors (with Numpy-style broadcasting support). 
    All inputs and outputs must have the same data type. 
    This operator supports multidirectional (i.e., Numpy-style) broadcasting; 
    for more details please check the doc.

    Version
    This version of the operator has been available since version 13 of the default ONNX operator set.

    Other versions of this operator: 1, 6, 8

    Inputs (1 - ∞)
        data_0 (variadic, differentiable) : T
            List of tensors for sum.
    Outputs
        sum (differentiable) : Tq
            Output tensor.
    
    Type Constraints
        T : tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
    Constrain input and output types to float tensors.

    Args:
        op (Operation): _description_
        values (List[torch.Tensor]): _description_
        ctx (TorchBackendContext, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    output = torch.zeros_like(values[0])
    for value in values:
        output += value
    return output


def Elu_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Elu takes one input data (Tensor) and produces one output data (Tensor) 
    where the function f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0., 
    is applied to the tensor elementwise.

    Version
    This version of the operator has been available since version 6 of the default ONNX operator set.

    Other versions of this operator: 1

    Attributes
        alpha : float (default is 1.0)
            Coefficient of ELU.
    
    Inputs
        X (differentiable) : T
            1D input tensor
    
    Outputs
        Y (differentiable) : T
            1D output tensor
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    alpha = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='alpha', default=1.0)
    return F.elu(x, alpha=alpha)


def Erf_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """
    Elu takes one input data (Tensor) and produces one output data (Tensor) 
    where the function f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0., 
    is applied to the tensor elementwise.

    Version
    This version of the operator has been available since version 6 of the default ONNX operator set.

    Other versions of this operator: 1

    Attributes
        alpha : float (default is 1.0)
            Coefficient of ELU.
    
    Inputs
        X (differentiable) : T
            1D input tensor
    
    Outputs
        Y (differentiable) : T
            1D output tensor
    
    Type Constraints
    T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.
    """
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values
    return torch.erf(x) # may require a higher version pytorch


DEFAULT_BACKEND_TABLE = {
    'Abs': Abs_forward,
    'AdaptiveAvgPool2d': AdaptiveAvgPool2d_forward,
    'And':And_forward,
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
    'Cos': Cos_forward,
    'Div': Eltwise_forward,
    'Equal': Equal_forward,
    'Exp': UnaryEltwise_forward,
    'Expand': Expand_forward,
    'Flatten': Flatten_forward,
    'Gather': Gather_forward,
    'GatherElements': Gather_forward,
    'GatherND': GatherND_forward,
    'Gelu': Gelu_forward,
    'Gemm': Gemm_forward,
    'grid_sampler': Grid_sampler_forward,
    'GlobalAveragePool': AveragePool_forward,
    'GlobalMaxPool': MaxPool2d_forward,
    'Greater': Greater_forward,
    'LayerNorm': LayerNorm_forward,
    'LeakyRelu': LeakyRelu_forward,
    'Less': Less_forward,
    'LogSoftmax': LogSoftmax_forward,
    'MatMul': MatMul_forward,
    'Max': Eltwise_forward,
    'MaxPool': MaxPool2d_forward,
    'Min': Eltwise_forward,
    'Mul': Mul_forward,
    'MultiHeadAttention': MultiHeadAttention_forward,
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
    'Sin': Sin_forward,
    'Slice': Slice_forward,
    'Softmax': Softmax_forward,
    'Softplus': Softplus_forward,
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
    'Tan': Tan_forward,
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
    'GRU': GRU_forward,
    'PPQDeviceSwitch': PPQDeviceSwitch_forward,
    'Identity': Identity_forward,
    'OneHot': Onehot_forward,
    'Reciprocal': Reciprocal_forward,
    'LSTM': LSTM_forward,
    'Sum': Sum_forward,
    'Elu': Elu_forward,
    'Erf': Erf_forward,
}
