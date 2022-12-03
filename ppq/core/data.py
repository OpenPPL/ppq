"""PPQ Core Data Structure Abstraction PPQ 核心数据结构抽象.

You are not allowed to modify this 请勿修改此文件
"""
from enum import Enum
from typing import Any, List, Union

import numpy as np
import torch
from numpy import dtype as np_type
from numpy import ndarray
from torch import Tensor
from torch import dtype as torch_type


class DataType(Enum):
    """
        DataType defines all PPQ internal data type and its enumeration value.
            ATTENTION: PPQ shares same data type enumeration value with Onnx.

        System maintainers and modifier are supposed to keep this corresponding.
        Cause OnnxExporter directly use this value to export PPQ graph towards Onnx.
    """
    INT4   = -1 # Onnx doesn't have this definition
    UINT4  = -2 # Onnx doesn't have this definition
    INT8   = 3  # Onnx.TensorProto.DataType.INT8
    UINT8  = 2  # Onnx.TensorProto.DataType.UINT8
    INT16  = 5  # Onnx.TensorProto.DataType.INT16
    UINT16 = 4  # Onnx.TensorProto.DataType.UINT16
    INT32  = 6  # Onnx.TensorProto.DataType.INT32
    UINT32 = 12 # Onnx.TensorProto.DataType.UINT32
    INT64  = 7  # Onnx.TensorProto.DataType.INT64
    UINT64 = 13 # Onnx.TensorProto.DataType.UINT64

    FP16 = 10 # Onnx.TensorProto.DataType.FLOAT16
    FP32 = 1  # Onnx.TensorProto.DataType.FLOAT
    FP64 = 11 # Onnx.TensorProto.DataType.DOUBLE

    BOOL = 9  # Onnx.TensorProto.DataType.BOOL
    COMPLEX128 = 15 # Onnx.TensorProto.DataType.COMPLEX128
    COMPLEX64 = 14 # Onnx.TensorProto.DataType.COMPLEX64
    NONETYPE  = 0 # Onnx.TensorProto.DataType.UNSPECIFIED

    @ classmethod
    def convert_from_numpy(cls, dtype: np_type):
        numpy_converting_dict = {
            np_type('bool'):    DataType.BOOL,
            np_type('uint8'):   DataType.UINT8,
            np_type('int8'):    DataType.INT8,
            np_type('int16'):   DataType.INT16,
            np_type('int32'):   DataType.INT32,
            np_type('int64'):   DataType.INT64,
            np_type('float16'): DataType.FP16,
            np_type('float32'): DataType.FP32,
            np_type('float64'): DataType.FP64,
        }
        if dtype not in numpy_converting_dict:
            raise TypeError(f'Numpy type {dtype} is not included in ppq now. '
                'please contact with system developer.')
        else:
            return numpy_converting_dict[dtype]

    @ classmethod
    def convert_from_torch(cls, dtype: torch_type):
        torch_converting_dict = {
            torch.bool:    DataType.BOOL,
            torch.uint8:   DataType.UINT8,
            torch.int8:    DataType.INT8,
            torch.int16:   DataType.INT16,
            torch.int32:   DataType.INT32,
            torch.int64:   DataType.INT64,
            torch.float16: DataType.FP16,
            torch.float32: DataType.FP32,
            torch.float64: DataType.FP64,
        }
        if dtype not in torch_converting_dict:
            raise TypeError(f'Torch dtype {dtype} is not included in ppq now. '
                'please contact with system developer.')
        else:
            return torch_converting_dict[dtype]

    @ classmethod
    def to_numpy(cls, dtype) -> np_type:
        numpy_converting_dict = {
            DataType.BOOL:  np_type('bool'),
            DataType.UINT8: np_type('uint8'),
            DataType.INT8:  np_type('int8'),
            DataType.INT16: np_type('int16'),
            DataType.INT32: np_type('int32'),
            DataType.INT64: np_type('int64'),
            DataType.FP16:  np_type('float16'),
            DataType.FP32:  np_type('float32'),
            DataType.FP64:  np_type('float64'),
        }
        assert isinstance(dtype, DataType)
        return numpy_converting_dict[dtype]

    @ classmethod
    def to_torch(cls, dtype) -> torch_type:
        torch_converting_dict = {
            DataType.BOOL:  torch.bool,
            DataType.UINT8: torch.uint8,
            DataType.INT8:  torch.int8,
            DataType.INT16: torch.int16,
            DataType.INT32: torch.int32,
            DataType.INT64: torch.int64,
            DataType.FP16:  torch.float16,
            DataType.FP32:  torch.float32,
            DataType.FP64:  torch.float64,
        }
        assert isinstance(dtype, DataType)
        return torch_converting_dict[dtype]


class TensorMeta:
    def __init__(
        self, dtype: DataType, shape: List[int],
        tensor_name: str = None) -> None:
        """TensorMeta structure described metadata of a tensor.

        Which includes tensor's data type and shape.
        TensorMeta is necessary to initialize quantization configuration and hooks,
        and is needed to compute the number of input channels.
        Args:
            dtype (DataType):
                A DataType enumeration described tensor type.
            shape (List[int]):
                A int list contains size of each dimensions.
            tensor_name (str, optional): Not yet used.
        """
        if not isinstance(dtype, DataType):
            raise TypeError(f'Can not create Tensor Meta with dtype {type(dtype)}, '
                            'only ppq.core.DataType instance is acceptable here.')
        self.dtype = dtype
        self.shape = shape
        self.name  = tensor_name

    @ classmethod
    def parsing_from_numpy_ndarray(cls, numpy_array: ndarray, name: str = None):
        shape = list(numpy_array.shape)
        dtype = DataType.convert_from_numpy(numpy_array.dtype)
        return TensorMeta(dtype=dtype, shape=shape,tensor_name=name)

    @ classmethod
    def parsing_from_torch_tensor(cls, torch_tensor: Tensor, name: str = None):
        if not isinstance(torch_tensor, Tensor):
            raise TypeError(f'Can not parse meta data for {type(torch_tensor)} instance, '
                'it should be torch.Tensor object.')
        shape = list(torch_tensor.shape)

        # for tensor scalar, which do not have an valid shape
        # just mannully give a empty list to them.
        if not shape: shape = []

        dtype = DataType.convert_from_torch(torch_tensor.dtype)
        return TensorMeta(dtype=dtype, shape=shape, tensor_name=name)

    def create_tensor(self, device: str, fill_value: Any = 0):
        return torch.Tensor(size=self.shape, device='cpu').fill_(
            fill_value).type(dtype=DataType.to_torch(self.dtype)).to(device)

    def create_ndarray(self, fill_value: Any = 0):
        return ndarray(shape=self.shape,
            dtype=DataType.to_numpy(self.dtype)).fill(fill_value)

    def __str__(self) -> str:
        return f'Tensor({self.name}) meta: dtype({self.dtype}), shape({self.shape})'
    
    def copy(self):
        if self.shape is not None:
            return TensorMeta(dtype=self.dtype, shape=self.shape.copy(), tensor_name=self.name)
        else: return TensorMeta(dtype=self.dtype, shape=None, tensor_name=self.name)


class OperationMeta:
    def __init__(self,
        input_metas: List[TensorMeta], output_metas: List[TensorMeta],
        operation_name: str, operation_type: str, executing_order: int) -> None:
        """OperationMeta structure describes all related tensor metadata of an
        operation.

        It naturally is a collection of TensorMeta.
        Take a look at TensorMeta to get more information.
        Args:
            input_metas (List[TensorMeta]):
                A collection contains all input tensors' metadata.
                ATTENTION: All parameters are considered as input in PPQ.
            output_metas (List[TensorMeta]):
                A collection contains all output tensors' metadata.
            operation_name (str): Not yet used.
            operation_type (str): Not yet used.
            executing_order (int): a int value represents the executing order of this operation.
                (order 0 means this operation is the first operation to be executed)
        """
        assert isinstance(input_metas, list), 'can only accept list object here.'
        assert isinstance(output_metas, list), 'can only accept list object here.'

        self.input_metas  = input_metas
        self.output_metas = output_metas

        self.operation_name = operation_name
        self.operation_type = operation_type
        self.executing_order = executing_order

    def __str__(self) -> str:
        return 'Inputs: '.join(str([_ for _ in self.input_metas])) + \
            'Outputs: '.join(str([_ for _ in self.input_metas]))

    @ property
    def num_of_input(self):
        return len(self.input_metas)

    @ property
    def num_of_output(self):
        return len(self.output_metas)

    def copy(self):
        return OperationMeta(
            input_metas = [meta.copy() for meta in self.input_metas],
            output_metas = [meta.copy() for meta in self.output_metas],
            operation_name=self.operation_name, operation_type=self.operation_type, 
            executing_order=self.executing_order)


def convert_any_to_python_primary_type(
    x: Union[torch.Tensor, np.ndarray, int, float, list, str],
    accept_none: bool=True) -> Union[int, float, list, str]:
    if x is None and accept_none: return None
    if x is None and not accept_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple): return list(x)
    elif isinstance(x, int) or isinstance(x, float): return x
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none: return None
        if x.numel() == 0 and not accept_none: raise ValueError('Trying to convert an empty value.')
        if str(x.device) != 'cpu': x = x.cpu()
        if x.numel() == 1: return x.item()
        if x.numel()  > 1: return x.tolist()
    elif isinstance(x, np.ndarray):
        if x.size == 0 and accept_none: return None
        if x.size == 0 and not accept_none: raise ValueError('Trying to convert an empty value.')
        if x.size == 1: return x.reshape((1, )).tolist()[0]
        if x.size  > 1: return x.tolist()
    elif isinstance(x, str):
        return x
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as python primary type.')


def convert_any_to_numpy(
    x: Union[torch.Tensor, np.ndarray, int, float, list, tuple],
    accept_none: bool=True) -> np.ndarray:
    if x is None and accept_none: return None
    if x is None and not accept_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, np.ndarray): return x
    elif isinstance(x, int) or isinstance(x, float): return np.array([x, ])
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accept_none: return None
        if x.numel() == 0 and not accept_none: raise ValueError('Trying to convert an empty value.')
        if x.numel() >= 1: return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as numpy type.')


def convert_any_to_torch_tensor(
    x: Union[torch.Tensor, np.ndarray, int, float, list, tuple],
    accept_none: bool=True, dtype: torch.dtype=None, device='cpu') -> torch.Tensor:
    if x is None and accept_none: return None
    if x is None and not accept_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple):
        if all([type(element) == int for element in x]):
            if dtype is None: dtype=torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, int):
        if dtype is None: dtype=torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, float):
        if dtype is None: dtype=torch.float32
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        if dtype is not None: x = x.type(dtype)
        if device is not None: x = x.to(device)
        return x
    elif isinstance(x, np.ndarray):
        if dtype is None:
            dtype = DataType.convert_from_numpy(x.dtype)
            dtype = DataType.to_torch(dtype)
        return torch.tensor(x.copy(), dtype=dtype, device=device)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as torch tensor.')


def convert_primary_type_to_list(
    x:Union[int, float, list, tuple]) -> list:
    if isinstance(x, list) or isinstance(x, tuple): return list(x)
    elif isinstance(x, int) or isinstance(x, float): return [x]
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as python list.')


def convert_any_to_string(x: Union[torch.Tensor, np.ndarray, int, float, list, tuple]) -> str:
    if isinstance(x, int) or isinstance(x, float):
        return x.__str__()
    elif isinstance(x, list) or isinstance(x, tuple):
        return '[{content}]'.format(content=''.join([convert_any_to_string(_) + ',' for _ in x]))
    elif isinstance(x, torch.Tensor):
        return convert_any_to_string(convert_any_to_numpy(x))
    elif isinstance(x, np.ndarray):
        return str(x.tobytes())
    else: return x.__str__()
