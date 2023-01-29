"""PPQ Core Foreign Function Interface PPQ 核心编程语言接口.

You are not allowed to modify this 请勿修改此文件
"""

import os
from typing import List

import torch
from torch.cuda import synchronize
from torch.utils.cpp_extension import load
from .config import PPQ_CONFIG
from .defs import ppq_warning, SingletonMeta


class ComplieHelper(metaclass=SingletonMeta):
    """ PPQ-Torch Compile Wrapper. """
    def __init__(self) -> None:
        self.__CUDA_EXTENTION__ = None

    def complie(self):
        ppq_warning('Compling Kernels... Please wait (It will take a few minutes).')
        lock_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/build/lock')
        if os.path.exists(lock_file): 
            try: os.remove(lock_file)
            except Exception as e:
                raise PermissionError(f'Can not delete lock file at {lock_file}, delete it first!')

        self.__CUDA_EXTENTION__ = load(
            name='PPQ_Cuda_Impls',
            sources=[
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/export.cc'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/linear.cu'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/sort.cu'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/train.cu'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cuda/floating.cu'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/cpu/hist_mse.cc'),
            ],
            build_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc/build/'),
            with_cuda=True,
            extra_cflags=['-O3'])
    
    @ property
    def CUDA_EXTENSION(self):
        if self.__CUDA_EXTENTION__ is None:
            raise Exception(
                'Cuda Extension has not been compiled, '
                'invoke ppq.core.ffi.ComplieHelper.complie() First.')
        return self.__CUDA_EXTENTION__

CUDA_COMPLIER = ComplieHelper()
if PPQ_CONFIG.USING_CUDA_KERNEL:
    CUDA_COMPLIER.complie()

# helper class for calling cuda methods.
class CUDA:
    """CUDA is a helper class for invoking highly-effcient custimized cuda
    kernel. PPQ developer team has implemented a series of quantization related
    cuda kernel, They are 5-100x faster than torch kernels, with less gpu
    memory cost.

    You can easily extend your cuda kernel via this class:
        Firstly, implement your kernel within ppq/csrc/cuda, write your own .cu file and .h file.
        Secondly, add your functions to ppq/csrc/cuda/export.cc, add them to export table.
        Finally, add a interface with this python class(ppq.core.ffi.CUDA),
        following the signature as same as others.

    PPQ CUDA EXTENSION 命名规则:
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
        rounding: int = 0
    ) -> torch.Tensor:
        # if scale is too small, quantization might cause fp32 underflow.
        # if scale < 1e-7: raise ValueError('scale is too small.')
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LT(
            tensor, scales, offsets, minimum, maximum, rounding)

    @ staticmethod
    def LinearQuantize_C(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0
    ) -> torch.Tensor:
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LC(
            tensor, scales, offsets, minimum, maximum, channel_axis, rounding)

    @ staticmethod
    def LinearQuantize_T_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        minimum: int,
        maximum: int,
        rounding: int,
    ) -> List[torch.Tensor]:
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LT_B(
            tensor, scales, offsets,
            dy, minimum, maximum, rounding
        )

    @ staticmethod
    def LinearQuantize_C_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        minimum: int,
        maximum: int,
        channel_axis: int,
        rounding: int,
    ) -> List[torch.Tensor]:
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_LC_B(
            tensor, scales, offsets,
            dy, minimum, maximum, rounding, channel_axis
        )

    @ staticmethod
    def Histogram_T(
        tensor: torch.Tensor,
        histogram: torch.Tensor,
        scale: float,
        clip_outliers: bool = True
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        CUDA_COMPLIER.CUDA_EXTENSION.Histogram_T(tensor, scale, clip_outliers, histogram)
        return histogram

    @ staticmethod
    def Histogram_Asymmetric_T(
        min_value: float,
        max_value: float,
        tensor: torch.Tensor,
        histogram: torch.Tensor,
        clip_outliers: bool = True
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        CUDA_COMPLIER.CUDA_EXTENSION.Histogram_Asymmetric_T(min_value, max_value, tensor, clip_outliers, histogram)
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
        CUDA_COMPLIER.CUDA_EXTENSION.Histogram_C(tensor, channel_axis, scale, clip_outliers, histogram)
        return histogram

    @ staticmethod
    def Quantile(
        tensor: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        return CUDA_COMPLIER.CUDA_EXTENSION.Quantile_T(tensor, q)

    @ staticmethod
    def TensorClip_T(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        limit: torch.Tensor,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        if not reference.is_contiguous(): tensor = reference.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.TensorClip_T(tensor, reference, limit)

    @ staticmethod
    def TensorClip_C(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        limit: torch.Tensor,
        channel_axis: int,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        if not reference.is_contiguous(): tensor = reference.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.TensorClip_C(
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
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LT(
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
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LT_B(
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
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LC(
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
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LC_B(
            tensor, dy, scales, offsets, minimum, maximum, channel_axis, rounding)

    @ staticmethod
    def OrderPreservingObserve(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.RoundingLoss_LC_B(tensor)

    @ staticmethod
    def compute_mse_loss(
        histogram: list,
        start: int,
        step: int,
        end: int
    ) -> float:
        return CUDA_COMPLIER.CUDA_EXTENSION.compute_mse_loss(histogram, start, step, end)

    @ staticmethod
    def FloatingQuantize_T(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        exponent: int = 4, 
        mantissa: int = 3,
        minimum: float = - 448, # FP8 E4M3
        maximum: float = + 448,
        rounding: int = 0
    ) -> torch.Tensor:
        if exponent <= 0: raise ValueError('Floating Quantization requires exponent > 0')
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        # if scale is too small, quantization might cause fp32 underflow.
        # if scale < 1e-7: raise ValueError('scale is too small.')
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FT(
            tensor, scales, offsets, exponent, mantissa, minimum, maximum, rounding)

    @ staticmethod
    def FloatingQuantize_C(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        exponent: int = 4, 
        mantissa: int = 3,
        minimum: float = - 448, # FP8 E4M3
        maximum: float = + 448,
        rounding: int = 0
    ) -> torch.Tensor:
        if exponent <= 0: raise ValueError('Floating Quantization requires exponent > 0')
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FC(
            tensor, scales, offsets, exponent, mantissa, 
            minimum, maximum, channel_axis, rounding)

    @ staticmethod
    def FloatingQuantize_T_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        exponent: int, 
        mantissa: int,
        minimum: float,
        maximum: float,
        rounding: int,
    ) -> List[torch.Tensor]:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FT_B(
            tensor, scales, offsets,
            dy, exponent, mantissa, minimum, maximum, rounding
        )

    @ staticmethod
    def FloatingQuantize_C_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        exponent: int, 
        mantissa: int,
        minimum: float,
        maximum: float,
        channel_axis: int,
        rounding: int,
    ) -> List[torch.Tensor]:
        if not tensor.is_contiguous(): tensor = tensor.contiguous()
        return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FC_B(
            tensor, scales, offsets,
            dy, exponent, mantissa, minimum, maximum, 
            rounding, channel_axis
        )


    @ staticmethod
    def Sync():
        """Synchronize device."""
        synchronize()
