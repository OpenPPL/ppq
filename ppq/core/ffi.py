"""
    PPQ Core Foreign Function Interface
    PPQ 核心编程语言接口

    You are not allowed to modify this
    请勿修改此文件
"""

import os

import torch
from torch.utils.cpp_extension import load
from torch.cuda import synchronize

from .defs import ppq_warning

ppq_warning('Compling CUDA Kernels. Please wait...')
__CUDA_EXTENTION__ = load(
    name='PPQ_Cuda_Impls',
    sources=[
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/export.cc'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/linear.cu'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/sort.cu'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/train.cu'),
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
        They are 5-100x faster than torch kernels, with less gpu memory cost.

    You can easily extend your cuda kernel via this class:
        Firstly, implement your kernel within ppq/csrc/cuda, write your own .cu file and .h file.
        Secondly, add your functions to ppq/csrc/cuda/export.cc, add them to export table.
        Finally, add a interface with this python class(ppq.core.ffi.CUDA), 
        following the signature as same as others.

    PPQ CUDA Extention 命名规则:
        我们使用函数名+后缀名的形式命名 CUDA Extension 函数:
        
        后缀名 _T 表示 Tensorwise 函数
        后缀名 _C 表示 Channelwise 函数
        后缀名 _B 表示 导函数

    例如函数 LinearQuantize_T_B 表示线性量化函数的 Tensorwise 版本，并且是导函数。
    """
    @ staticmethod
    def LinearQuantize_T(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
        dropout: float = 0
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        # if scale is too small, quantization might cause fp32 underflow.
        # if scale < 1e-7: raise ValueError('scale is too small.')
        return __CUDA_EXTENTION__.QuantizeTensor_LT(
            tensor, scales, offsets, minimum, maximum, rounding, dropout)

    @ staticmethod
    def LinearQuantize_C(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
        dropout: float = 0,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.QuantizeTensor_LC(
            tensor, scales, offsets, minimum, maximum, channel_axis, rounding, dropout)
        
    @ staticmethod
    def LinearQuantize_T_B(
        tensor: torch.Tensor,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        minimum: int,
        maximum: int
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.QuantizeTensor_LT_B(
            tensor, quantized, scales, offsets, 
            dy, minimum, maximum
        )

    @ staticmethod
    def LinearQuantize_C_B(
        tensor: torch.Tensor,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        minimum: int,
        maximum: int,
        channel_axis: int,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.QuantizeTensor_LC_B(
            tensor, quantized, scales, offsets, 
            dy, minimum, maximum, channel_axis
        )

    @ staticmethod
    def Histogram_T(
        tensor: torch.Tensor,
        histogram: torch.Tensor,
        scale: float,
        clip_outliers: bool = True
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        __CUDA_EXTENTION__.Histogram_T(tensor, scale, clip_outliers, histogram)
        return histogram

    @ staticmethod
    def Histogram_C(
        tensor: torch.Tensor,
        channel_axis: int,
        histogram: torch.Tensor,
        scale: float,
        clip_outliers: bool = True
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        __CUDA_EXTENTION__.Histogram_C(tensor, channel_axis, scale, clip_outliers, histogram)
        return histogram

    @ staticmethod
    def Quantile(
        tensor: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        return __CUDA_EXTENTION__.Quantile_T(tensor, q)

    @ staticmethod
    def TensorClip_T(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        limit: torch.Tensor,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        if not reference.is_contiguous(): tensor = reference.contiguous()
        return __CUDA_EXTENTION__.TensorClip_T(tensor, reference, limit)

    @ staticmethod
    def TensorClip_C(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        limit: torch.Tensor,
        channel_axis: int,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        if not reference.is_contiguous(): tensor = reference.contiguous()
        return __CUDA_EXTENTION__.TensorClip_C(
            tensor, reference, limit, channel_axis)
        
    @ staticmethod
    def RoundingLoss_LT(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.RoundingLoss_LT(
            tensor, scales, offsets, minimum, maximum, rounding)

    @ staticmethod
    def RoundingLoss_LT_B(
        tensor: torch.Tensor,
        dy    : torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.RoundingLoss_LT_B(
            tensor, dy, scales, offsets, minimum, maximum, rounding)
        
    @ staticmethod
    def RoundingLoss_LC(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.RoundingLoss_LC(
            tensor, scales, offsets, minimum, maximum, channel_axis, rounding)

    @ staticmethod
    def RoundingLoss_LC_B(
        tensor: torch.Tensor,
        dy    : torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return __CUDA_EXTENTION__.RoundingLoss_LC_B(
            tensor, dy, scales, offsets, minimum, maximum, channel_axis, rounding)

    @ staticmethod
    def Sync():
        """
        Synchronize device.
        """
        synchronize()