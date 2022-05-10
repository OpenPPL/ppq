from typing import List

from ppq.IR import Operation

import torch

from .base import ASSERT_ALL_TENSORS_AT_SAME_DEVICE, ASSERT_NUM_OF_INPUT, TorchBackendContext
from .default import DEFAULT_BACKEND_TABLE

PPL_GPU_BACKEND_TABLE = DEFAULT_BACKEND_TABLE.copy()

# When you trying to implement a custimized function for ppl_gpu platform
# Be aware that you can just overwrite part of DEFAULT_DISPATCHING_TABLE
# rather than rewrite all dispatching table.
# here an example was given: Sample_Forward
def Sample_Forward():
    return None

def AveragePool_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
    """AveragePool-11 AveragePool consumes an input tensor X and applies
    average pooling across the tensor according to kernel sizes, stride sizes,
    and pad lengths. average pooling consisting of computing the average on all
    values of a subset of the input tensor according to the kernel size and
    downsampling the data into the output tensor Y for further processing. The
    output spatial shape will be following:

    Attributes
        auto_pad : string (default is NOTSET)
        auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.

        Where default value is NOTSET, which means explicit padding is used.
        SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])`
            for each axis `i`. The padding is split between the two sides equally or almost equally
            (depending on whether it is even or odd).

        In case the padding is an odd number,
            the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

    ceil_mode : int (default is 0)
        Whether to use ceil or floor (default) to compute the output shape.

    count_include_pad : int (default is 0)
        Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad.

    kernel_shape : list of ints (required)
        The size of the kernel along each axis.

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
        Stride along each spatial axis.
        If not present, the stride defaults to 1 along each spatial axis.

    Inputs
        X (differentiable) : T
        Input data tensor from the previous operator;
            dimensions for image case are (N x C x H x W),
            where N is the batch size, C is the number of channels,
            and H and W are the height and the width of the data.

        For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn),
            where N is the batch size. Optionally, if dimension denotation is in effect,
            the operation expects the input data tensor to arrive with
            the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

    Outputs
        Y (differentiable) : T
        Output data tensor from average or max pooling across the input tensor.
            Dimensions will vary based on various kernel, stride, and pad sizes.
            Floor value of the dimension is used

    Args:
        op (Operation): [description]
        values (List[torch.Tensor]): [description]
        ctx (TorchBackendContext, optional): [description]. Defaults to None.

    Returns:
        torch.Tensor: [description]
    """
    ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op=op, values=values)
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    # checker(op.attributes, input_value[0].shape[2:])
    # op_attr = preprocess_attr(op.attributes, 'Pooling')
    # [input_value] = input_value
    # if op.type == 'GlobalAveragePool':
    #     image_shape = input_value.size()[-2:]
    #     op_attr['kernel_size'] = image_shape
    #
    # output = F.avg_pool2d(input_value, **op_attr)
    # return output

PPL_GPU_BACKEND_TABLE['Sample_Function'] = Sample_Forward
