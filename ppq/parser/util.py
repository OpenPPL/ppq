from typing import Union

import numpy as np
import torch
from ppq.core import DataType, convert_any_to_numpy


def convert_value(
    value: Union[int, float, np.ndarray, torch.Tensor],
    export_as_float: bool, dtype: DataType = DataType.FP32) -> Union[float, list]:
    """Converting value from any to python native data dtype, ready for export.

    Args:
        value (Union[int, float, np.ndarray, torch.Tensor]): exporting value.
        export_as_list (bool): export value as a list.
        dtype (DataType, optional): exporting dtype.

    Returns:
        Union[float, list]: Converted value
    """
    if dtype not in {DataType.FP32, DataType.INT32}:
        raise ValueError(f'Can Only export dtype fp32 and int32, '
                         f'while you are requiring to dump a {dtype.name} value')
    value = convert_any_to_numpy(value, accept_none=False)
    value = value.astype(dtype=DataType.to_numpy(dtype))
    if export_as_float:
        value = value[0].item()
        assert type(value) in {int, float}, (
            f'Trying to dump a tensorwise quantization value {value}. '
            f'It is Expected to be a int or float value, while {type(value)} was given')
        return value
    else:
        value = convert_any_to_numpy(value, accept_none=False)
        return value.tolist()
