"""
    PPQ Core Foreign Function Interface
    PPQ 核心编程语言接口

    You are not allowed to modify this
    请勿修改此文件
"""

import os
from typing import List

import torch
from torch.utils.cpp_extension import load

from .defs import ppq_warning

ppq_warning('Ninja is compling CUDA Backends now, Please wait...')
PPQ_CUDA = load(
    name='PPQ_Cuda_Impls',
    sources=[
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/export.cc'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/linear.cu'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/sort.cu'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/sieve.cu'),
    ],
    build_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/build/'),
    with_cuda=True,
    extra_cflags=['-O3']
)

# helper class for calling cuda methods.
class CUDA:
    """
    CUDA is a helper class for invoking highly-effcient custimized cuda kernel.
        PPQ developer team has implemented a series of quantization related cuda kernel,
        They are 5-100x faster than torch kernels, with 50% less gpu memory cost.

    You can easily extend your cuda kernel via this class:
        Firstly, implement your kernel within ppq/csrc/cuda, write your own .cu file and .h file.
        Secondly, add your functions to ppq/csrc/cuda/export.cc, add them to export table.
        Finally, add a interface with this python class(ppq.core.ffi.CUDA), following the signature as same as others.
    """
    @ staticmethod
    def TensorwiseLinearQuantize(
        tensor: torch.Tensor,
        scale: float,
        offset: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
        inplace: bool = False,
    ) -> torch.Tensor:
        # we can quantize all element inplace via custimized CUDA kernel to save memory.
        if not inplace: output = tensor.clone()
        else: output = tensor

        # if scale is too small, quantization might cause fp32 underflow.
        # if scale < 1e-7: raise ValueError('scale is too small.')

        PPQ_CUDA.TensorwiseLinearQuantize(
            tensor, output, scale, offset, minimum, maximum, rounding)

        return output
    
    @ staticmethod
    def ChannelwiseLinearQuantize(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
        inplace: bool = False,
    ) -> torch.Tensor:
        # we can quantize all element inplace via custimized CUDA kernel to save memory.
        if not inplace: output = tensor.clone()
        else: output = tensor

        PPQ_CUDA.ChannelwiseLinearQuantize(tensor, scales, offsets, output, 
                                           channel_axis, minimum, maximum, rounding)

        return output

    @ staticmethod
    def Histogram(
        tensor: torch.Tensor,
        histogram: torch.Tensor,
        scale: float,
        offset: int = 0,
        abs_mode: bool = True,
        rounding: int = 0
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        PPQ_CUDA.TensorwiseHistogram(tensor, histogram, scale, offset, abs_mode, rounding)
        return histogram

    @ staticmethod
    def Quantile(
        tensor: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        return PPQ_CUDA.Quantile(tensor, q)

    @ staticmethod
    def TensorwiseLinearQuantSieve(
        tensor: torch.Tensor,
        fp_offset: torch.Tensor,
        scale: float,
        offset: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
        limit: float = 2.0,
        threshold: float = 0.95
    ) -> List[torch.Tensor]:
        return PPQ_CUDA.TensorwiseLinearQuantSieve(
            tensor, fp_offset, scale, offset, 
            minimum, maximum, rounding, limit, threshold)
    
    @ staticmethod
    def ChannelwiseLinearQuantSieve(
        tensor: torch.Tensor,
        fp_offset: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
        limit: float = 2.0,
        threshold: float = 0.95
    ) -> List[torch.Tensor]:
        return PPQ_CUDA.ChannelwiseLinearQuantSieve(
            tensor, fp_offset, scales, offsets, 
            channel_axis, minimum, maximum, rounding, 
            limit, threshold)
