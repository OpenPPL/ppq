"""PPQ Core Foreign Function Interface PPQ 核心编程语言接口.

You are not allowed to modify this 请勿修改此文件
"""

# pylint: disable=invalid-name

import json
import os
import shutil
import subprocess as sp
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import pooch
import torch
from torch.cuda import synchronize
from torch.utils.cpp_extension import load

from mppq.defs import SingletonMeta
from mppq.logger import warning

_URLS = [
    (
        "https://github.com/microsoft/vswhere/releases/download/3.1.7/vswhere.exe",
        "c54f3b7c9164ea9a0db8641e81ecdda80c2664ef5a47c4191406f848cc07c662",
    ),
]


class MSVCLocator:
    """Download vswhere to locate the installed Microsoft Visual Studio instances.
    And set CXX environment variable to compiler path.

    Set CXX to None if no Visual Studio is installed.
    Raise FileNotFoundError if Visual Studio is installed without C++ components.
    """

    def __init__(self, arch: str = "x64"):
        self._old_cxx: Optional[str] = None
        if (vswhere := shutil.which("vswhere")) is None:
            for url, sha256 in _URLS:
                vswhere = pooch.retrieve(url, known_hash=sha256, progressbar=True)
        assert vswhere is not None
        self.vswhere = Path(vswhere)
        assert self.vswhere.exists() and self.vswhere.is_file()
        if arch.lower() in ("amd64", "x64"):
            self.x86 = False
        elif arch.lower() in ("x86", "i386"):
            self.x86 = True
        else:
            raise NotImplementedError(f"Unrecoganized arch: {arch}")

    def __enter__(self) -> Path:
        stdout = sp.check_output(
            ["", "-latest", "-utf8", "-format", "json"], executable=self.vswhere
        )
        results = json.loads(stdout)
        for cl in filter(
            lambda i: i is not None, (self._find_cl(search) for search in results)
        ):
            assert cl is not None
            self._old_cxx = os.environ.get("CXX")
            os.environ["CXX"] = str(cl.resolve())
            # expand path
            os.environ["PATH"] += f";{cl.parent.resolve()}"
            print(f"set CXX={cl}")
            break
        return Path(os.environ["CXX"])

    def __exit__(self, *args):
        if self._old_cxx is None and "CXX" in os.environ:
            os.environ.pop("CXX")
        elif self._old_cxx:
            os.environ["CXX"] = str(self._old_cxx)

    def _find_cl(self, search: dict) -> Optional[Path]:
        msvc_path = Path(search["installationPath"]) / "VC/Tools/MSVC"
        if not msvc_path.exists():
            raise FileNotFoundError(
                f"Visual Studio is installed without C++ components: {msvc_path}"
            )
        for instance_dir in filter(lambda p: p.is_dir(), msvc_path.glob("*")):
            if self.x86:
                cl = instance_dir / "bin/Hostx86/x86/cl.exe"
            else:
                cl = instance_dir / "bin/Hostx64/x64/cl.exe"
            if cl.exists() and cl.is_file():
                return cl
        return None


class CompilerHelper(metaclass=SingletonMeta):
    """PPQ-Torch Compile Wrapper."""

    def __init__(self) -> None:
        self.__cuda_extention__ = None

    def compile(self):
        """Compile CUDA kernels."""

        warning("Compiling Kernels... Please wait (It will take a few minutes).")
        build_directory = Path(__file__).parent / "csrc/build/"
        lock_file = build_directory / "lock"

        if not build_directory.exists():
            build_directory.mkdir(parents=True)
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception as e:
                raise PermissionError(
                    f"Can not delete lock file at {lock_file}, delete it first!"
                ) from e

        self.__cuda_extention__ = load(
            name="PPQ_Cuda_Impls",
            sources=[
                str(Path(__file__).parent / "csrc/export.cc"),
                str(Path(__file__).parent / "csrc/cuda/linear.cu"),
                str(Path(__file__).parent / "csrc/cuda/sort.cu"),
                str(Path(__file__).parent / "csrc/cuda/train.cu"),
                str(Path(__file__).parent / "csrc/cuda/floating.cu"),
                str(Path(__file__).parent / "csrc/cpu/hist_mse.cc"),
            ],
            build_directory=build_directory,
            with_cuda=True,
            extra_cflags=["-O3"],
        )

    @property
    def extension(self):
        """Return the loaded CUDA library."""
        if self.__cuda_extention__ is None:
            raise RuntimeError(
                "Cuda Extension has not been compiled, "
                "invoke ppq.core.ffi.ComplieHelper.compile() First."
            )
        return self.__cuda_extention__


CUDA_COMPLIER = CompilerHelper()


@contextmanager
def dummy_locator():
    """Linux compiler locator, do nothing for now"""
    yield


class ENABLE_CUDA_KERNEL:
    """Auto config compiler path before entering compiling CUDA context"""

    USING_CUDA_KERNEL = True

    def __init__(self) -> None:
        self._state = True
        if os.name == "nt":
            with MSVCLocator():
                self._compile()
        else:
            with dummy_locator():
                self._compile()

    def _compile(self):
        if CUDA_COMPLIER.__cuda_extention__ is None:
            CUDA_COMPLIER.compile()
        self._state = False

    def __enter__(self):
        self._state = ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL
        ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL = True

    def __exit__(self, *args):
        ENABLE_CUDA_KERNEL.USING_CUDA_KERNEL = self._state


# pylint: disable=missing-function-docstring, invalid-name
# helper class for calling cuda methods.
class CUDA:
    """CUDA is a helper class for invoking highly-effcient custimized cuda
    kernel. PPQ developer team has implemented a series of quantization related
    cuda kernel, They are 5-100x faster than torch kernels, with less gpu
    memory cost.

    You can easily extend your cuda kernel via this class:
        Firstly, implement your kernel within ppq/csrc/cuda, write your own .cu file
        and .h file.

        Secondly, add your functions to ppq/csrc/cuda/export.cc, add them to export
        table.

        Finally, add a interface with this python class(ppq.core.ffi.CUDA), following
        the signature as same as others.

    PPQ CUDA EXTENSION 命名规则:
        我们使用函数名+后缀名的形式命名 CUDA Extension 函数:

        后缀名 _T 表示 Tensorwise 函数
        后缀名 _C 表示 Channelwise 函数
        后缀名 _B 表示 导函数

    例如函数 LinearQuantize_T_B 表示线性量化函数的 Tensorwise 版本，并且是导函数。
    """

    @staticmethod
    def LinearQuantize_T(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
    ) -> torch.Tensor:
        # if scale is too small, quantization might cause fp32 underflow.
        # if scale < 1e-7: raise ValueError('scale is too small.')
        return CUDA_COMPLIER.extension.QuantizeTensor_LT(
            tensor, scales, offsets, minimum, maximum, rounding
        )

    @staticmethod
    def LinearQuantize_C(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
    ) -> torch.Tensor:
        return CUDA_COMPLIER.extension.QuantizeTensor_LC(
            tensor, scales, offsets, minimum, maximum, channel_axis, rounding
        )

    @staticmethod
    def LinearQuantize_T_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        minimum: int,
        maximum: int,
        rounding: int,
    ) -> List[torch.Tensor]:
        return CUDA_COMPLIER.extension.QuantizeTensor_LT_B(
            tensor, scales, offsets, dy, minimum, maximum, rounding
        )

    @staticmethod
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
        return CUDA_COMPLIER.extension.QuantizeTensor_LC_B(
            tensor, scales, offsets, dy, minimum, maximum, rounding, channel_axis
        )

    @staticmethod
    def Histogram_T(
        tensor: torch.Tensor,
        histogram: torch.Tensor,
        scale: float,
        clip_outliers: bool = True,
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        CUDA_COMPLIER.extension.Histogram_T(tensor, scale, clip_outliers, histogram)
        return histogram

    @staticmethod
    def Histogram_Asymmetric_T(
        min_value: float,
        max_value: float,
        tensor: torch.Tensor,
        histogram: torch.Tensor,
        clip_outliers: bool = True,
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        CUDA_COMPLIER.extension.Histogram_Asymmetric_T(
            min_value, max_value, tensor, clip_outliers, histogram
        )
        return histogram

    @staticmethod
    def Histogram_C(
        tensor: torch.Tensor,
        channel_axis: int,
        histogram: torch.Tensor,
        scale: float,
        clip_outliers: bool = True,
    ) -> torch.Tensor:
        # if scale < 1e-7: raise ValueError('scale is too small.')
        CUDA_COMPLIER.extension.Histogram_C(
            tensor, channel_axis, scale, clip_outliers, histogram
        )
        return histogram

    @staticmethod
    def Quantile(
        tensor: torch.Tensor,
        q: float,
    ) -> torch.Tensor:
        return CUDA_COMPLIER.extension.Quantile_T(tensor, q)

    @staticmethod
    def TensorClip_T(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        limit: torch.Tensor,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not reference.is_contiguous():
            tensor = reference.contiguous()
        return CUDA_COMPLIER.extension.TensorClip_T(tensor, reference, limit)

    @staticmethod
    def TensorClip_C(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        limit: torch.Tensor,
        channel_axis: int,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not reference.is_contiguous():
            tensor = reference.contiguous()
        return CUDA_COMPLIER.extension.TensorClip_C(
            tensor, reference, limit, channel_axis
        )

    @staticmethod
    def RoundingLoss_LT(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.RoundingLoss_LT(
            tensor, scales, offsets, minimum, maximum, rounding
        )

    @staticmethod
    def RoundingLoss_LT_B(
        tensor: torch.Tensor,
        dy: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.RoundingLoss_LT_B(
            tensor, dy, scales, offsets, minimum, maximum, rounding
        )

    @staticmethod
    def RoundingLoss_LC(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.RoundingLoss_LC(
            tensor, scales, offsets, minimum, maximum, channel_axis, rounding
        )

    @staticmethod
    def RoundingLoss_LC_B(
        tensor: torch.Tensor,
        dy: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        minimum: int = -128,
        maximum: int = 127,
        rounding: int = 0,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.RoundingLoss_LC_B(
            tensor, dy, scales, offsets, minimum, maximum, channel_axis, rounding
        )

    @staticmethod
    def OrderPreservingObserve(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.RoundingLoss_LC_B(tensor)

    @staticmethod
    def compute_mse_loss(
        histogram: List[int], start: int, step: int, end: int
    ) -> float:
        return CUDA_COMPLIER.extension.compute_mse_loss(histogram, start, step, end)

    @staticmethod
    def FloatingQuantize_T(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        exponent: int = 4,
        mantissa: int = 3,
        minimum: float = -448,  # FP8 E4M3
        maximum: float = +448,
        rounding: int = 0,
    ) -> torch.Tensor:
        if exponent <= 0:
            raise ValueError("Floating Quantization requires exponent > 0")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # if scale is too small, quantization might cause fp32 underflow.
        # if scale < 1e-7: raise ValueError('scale is too small.')
        return CUDA_COMPLIER.extension.QuantizeTensor_FT(
            tensor, scales, offsets, exponent, mantissa, minimum, maximum, rounding
        )

    @staticmethod
    def FloatingQuantize_C(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        channel_axis: int,
        exponent: int = 4,
        mantissa: int = 3,
        minimum: float = -448,  # FP8 E4M3
        maximum: float = +448,
        rounding: int = 0,
    ) -> torch.Tensor:
        if exponent <= 0:
            raise ValueError("Floating Quantization requires exponent > 0")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.QuantizeTensor_FC(
            tensor,
            scales,
            offsets,
            exponent,
            mantissa,
            minimum,
            maximum,
            channel_axis,
            rounding,
        )

    @staticmethod
    def FloatingQuantize_T_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        exponent: int = 4,
        mantissa: int = 3,
        minimum: float = -448,
        maximum: float = 448,
        rounding: int = 0,
    ) -> List[torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.QuantizeTensor_FT_B(
            tensor, scales, offsets, dy, exponent, mantissa, minimum, maximum, rounding
        )

    @staticmethod
    def FloatingQuantize_C_B(
        tensor: torch.Tensor,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        dy: torch.Tensor,
        channel_axis: int,
        exponent: int = 4,
        mantissa: int = 3,
        minimum: float = -448,
        maximum: float = 448,
        rounding: int = 0,
    ) -> List[torch.Tensor]:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return CUDA_COMPLIER.extension.QuantizeTensor_FC_B(
            tensor,
            scales,
            offsets,
            dy,
            exponent,
            mantissa,
            minimum,
            maximum,
            rounding,
            channel_axis,
        )

    @staticmethod
    def Sync():
        """Synchronize device."""
        synchronize()
