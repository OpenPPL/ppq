"""PPQ Core Data Structure Abstraction PPQ 核心数据结构抽象.

You are not allowed to modify this 请勿修改此文件
"""

from enum import IntEnum
from numbers import Number
from typing import Any, Literal, Optional, Sequence, overload

import numpy as np
import torch
from numpy import dtype as np_type
from onnx import TensorProto
from torch import dtype as torch_type


class DataType(IntEnum):
    """
    DataType defines all PPQ internal data type and its enumeration value.
        ATTENTION: PPQ shares same data type enumeration value with Onnx.

    System maintainers and modifier are supposed to keep this corresponding.
    Cause OnnxExporter directly use this value to export PPQ graph towards Onnx.
    """

    INT4 = TensorProto.INT4
    UINT4 = TensorProto.UINT4
    INT8 = TensorProto.INT8
    UINT8 = TensorProto.UINT8
    INT16 = TensorProto.INT16
    UINT16 = TensorProto.UINT16
    INT32 = TensorProto.INT32
    UINT32 = TensorProto.UINT32
    INT64 = TensorProto.INT64
    UINT64 = TensorProto.UINT64

    FP8_E4M3FN = TensorProto.FLOAT8E4M3FN
    FP8_E4M3FNUZ = TensorProto.FLOAT8E4M3FNUZ
    FP8_E5M2 = TensorProto.FLOAT8E5M2
    FP8_E5M2FNUZ = TensorProto.FLOAT8E5M2FNUZ
    BF16 = TensorProto.BFLOAT16
    FP16 = TensorProto.FLOAT16
    FP32 = TensorProto.FLOAT
    FP64 = TensorProto.DOUBLE

    BOOL = TensorProto.BOOL
    COMPLEX128 = TensorProto.COMPLEX128
    COMPLEX64 = TensorProto.COMPLEX64
    NONETYPE = TensorProto.UNDEFINED

    @classmethod
    def from_numpy(cls, dtype: np_type):
        """Construct DataType from numpy dtype."""
        numpy_converting_dict = {
            np_type("bool"): DataType.BOOL,
            np_type("uint8"): DataType.UINT8,
            np_type("int8"): DataType.INT8,
            np_type("int16"): DataType.INT16,
            np_type("int32"): DataType.INT32,
            np_type("int64"): DataType.INT64,
            np_type("float16"): DataType.FP16,
            np_type("float32"): DataType.FP32,
            np_type("float64"): DataType.FP64,
            np.bool: DataType.BOOL,
            np.uint8: DataType.UINT8,
            np.int8: DataType.INT8,
            np.int16: DataType.INT16,
            np.int32: DataType.INT32,
            np.int64: DataType.INT64,
            np.float16: DataType.FP16,
            np.float32: DataType.FP32,
            np.float64: DataType.FP64,
        }
        if dtype not in numpy_converting_dict:
            raise TypeError(
                f"Numpy type {dtype} is not included in ppq now. "
                "please contact with system developer."
            )
        else:
            return numpy_converting_dict[dtype]

    @classmethod
    def from_torch(cls, dtype: torch_type):
        """Construct DataType from torch dtype."""
        torch_converting_dict = {
            torch.bool: DataType.BOOL,
            torch.uint8: DataType.UINT8,
            torch.int8: DataType.INT8,
            torch.int16: DataType.INT16,
            torch.int32: DataType.INT32,
            torch.int64: DataType.INT64,
            torch.float16: DataType.FP16,
            torch.float32: DataType.FP32,
            torch.float64: DataType.FP64,
            torch.float8_e4m3fn: DataType.FP8_E4M3FN,
            torch.float8_e4m3fnuz: DataType.FP8_E4M3FNUZ,
            torch.float8_e5m2: DataType.FP8_E5M2,
            torch.float8_e5m2fnuz: DataType.FP8_E5M2FNUZ,
        }
        if dtype not in torch_converting_dict:
            raise TypeError(
                f"Torch dtype {dtype} is not included in ppq now. "
                "please contact with system developer."
            )
        else:
            return torch_converting_dict[dtype]

    @classmethod
    def to_numpy(cls, dtype) -> np_type:
        """Convert DataType to numpy dtype."""
        numpy_converting_dict = {
            DataType.BOOL: np_type("bool"),
            DataType.UINT8: np_type("uint8"),
            DataType.INT8: np_type("int8"),
            DataType.INT16: np_type("int16"),
            DataType.INT32: np_type("int32"),
            DataType.INT64: np_type("int64"),
            DataType.FP16: np_type("float16"),
            DataType.FP32: np_type("float32"),
            DataType.FP64: np_type("float64"),
        }
        assert isinstance(dtype, int)
        return numpy_converting_dict[DataType(dtype)]

    @classmethod
    def to_torch(cls, dtype) -> torch_type:
        """Convert DataType to torch dtype."""
        torch_converting_dict = {
            DataType.BOOL: torch.bool,
            DataType.UINT8: torch.uint8,
            DataType.INT8: torch.int8,
            DataType.INT16: torch.int16,
            DataType.INT32: torch.int32,
            DataType.INT64: torch.int64,
            DataType.FP16: torch.float16,
            DataType.FP32: torch.float32,
            DataType.FP64: torch.float64,
            DataType.FP8_E4M3FN: torch.float8_e4m3fn,
            DataType.FP8_E4M3FNUZ: torch.float8_e4m3fnuz,
            DataType.FP8_E5M2: torch.float8_e5m2,
            DataType.FP8_E5M2FNUZ: torch.float8_e5m2fnuz,
        }
        assert isinstance(dtype, int)
        return torch_converting_dict[DataType(dtype)]


@overload
def convert_any_to_python_primary_type(
    x: Any, accept_none: Literal[True] = True
) -> Number | str | bool | list | None:
    """Return contains None type"""


@overload
def convert_any_to_python_primary_type(
    x: Any, accept_none: Literal[False]
) -> Number | str | bool | list:
    """Return does not contain None type"""


def convert_any_to_python_primary_type(x: Any, accept_none: bool = True):
    """Try to convert an object to python POD."""

    if x is None and accept_none:
        return None
    if x is None and not accept_none:
        raise ValueError("Trying to convert an empty value.")
    if isinstance(x, (Number, str, bool)):
        return x
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none:
            return None
        if x.numel() == 0 and not accept_none:
            raise ValueError("Trying to convert an empty value.")
        if str(x.device) != "cpu":
            x = x.cpu()
        if x.numel() == 1:
            return x.item()
        if x.numel() > 1:
            return x.tolist()
    elif isinstance(x, np.ndarray):
        if x.size == 0 and accept_none:
            return None
        if x.size == 0 and not accept_none:
            raise ValueError("Trying to convert an empty value.")
        if x.size == 1:
            return x.flatten().tolist()[0]
        if x.size > 1:
            return x.tolist()
    elif isinstance(x, Sequence):
        return list(x)
    else:
        raise TypeError(
            f"input value {x}({type(x)}) can not be converted as python primary type."
        )


@overload
def convert_any_to_numpy(
    x: Any, accept_none: Literal[True] = True
) -> None | np.ndarray:
    """Return contains None type"""


@overload
def convert_any_to_numpy(x: Any, accept_none: Literal[False]) -> np.ndarray:
    """Return does not contain None type"""


def convert_any_to_numpy(x: Any, accept_none: bool = True) -> None | np.ndarray:
    """Try to convert an object to numpy array."""
    if x is None and accept_none:
        return None
    if x is None and not accept_none:
        raise ValueError("Trying to convert an empty value.")
    if isinstance(x, np.ndarray):
        if x.size == 0 and accept_none:
            return None
        if x.size == 0 and not accept_none:
            raise ValueError("Trying to convert an empty value.")
        return x
    elif isinstance(x, torch.Tensor):
        if "float8" in str(x.dtype):
            return convert_any_to_numpy(
                x.cpu().to(torch.float32).numpy(), accept_none=accept_none
            )
        return convert_any_to_numpy(x.cpu().numpy(), accept_none=accept_none)
    elif isinstance(x, Number):
        return np.array([x])
    elif isinstance(x, (Sequence)):
        return np.array(x)
    else:
        raise TypeError(
            f"input value {x}({type(x)}) can not be converted as numpy type."
        )


def convert_any_to_tensor(
    x: Any,
    dtype: Optional[torch.dtype] = None,
    device: str | torch.device | None = "cpu",
) -> torch.Tensor:
    """Try to convert an object to torch tensor."""

    if isinstance(x, int):
        if dtype is None:
            dtype = torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, float):
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    elif isinstance(x, list) or isinstance(x, tuple):
        if all([isinstance(element, int) for element in x]):
            if dtype is None:
                dtype = torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        raise TypeError(
            f"input value {x}({type(x)}) can not be converted as torch tensor."
        )
