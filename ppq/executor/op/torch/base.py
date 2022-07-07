from typing import List, Any
from ppq.IR import Operation
from ppq.IR.quantize import QuantableOperation
from ppq.core import ppq_warning
import torch

class TorchBackendContext:
    def __init__(self, executing_device: str) -> None:
        self.executing_device = executing_device

def ASSERT_ALL_TENSORS_AT_CPU(op: Operation, values: List[torch.Tensor], force_convert: bool = False):
    """Dynamic Shape Platform Operations Process with shape related tensors,
    which must not be quantized and must be computed with cpu.

    This function will check all inputs tensors' device, and move all cuda tensor to cpu(if force_convert is true).
    ATTENTION: do not set force_convert as True if not necessary. PPQ will automatically move
        operations to proper platform, there should not be any input tensor deployed at cuda
        when invoking a dynamic shape operation.

    IF THERE IS ANY CUDA INPUT TENSOR FOR DYNAMIC SHAPE OPERATION, THERE PROBABLY ARE SOME SYSTEM FAILURES.
    YOU ARE SUPPOSED TO REPORT THOSE SYSTEM FAILURES TO US.

    Args:
        values (List[torch.Tensor]): values to be checked.
        force_convert (bool, optional): whether to convert all input tensors to cpu.
    """
    for idx, tensor in enumerate(values):
        if tensor is None: continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f'Input at [{idx}] of Operation [{op.name}] is not a torch.Tensor, '
                'which is not supposed to happen in PPQ execution system. '
                'Is there any parsing failure with your graph?')
        if str(tensor.device) != 'cpu':
            if not force_convert:
                raise ValueError(
                    f'Input at [{idx}] of Operation [{op.name}] deploy with incorrect device {tensor.device}, '
                    'which is not supposed to happen in PPQ execution system. This is a critical system failure, '
                    'you can set ppq.core.config.force_convert as True to force convert those values, which might be able to '
                    'continue executing your graph. YOU ARE RECOMMEND TO REPORT THIS FAILURE TO US.'
                )
            else:
                ppq_warning(f'Input at [{idx}] of Operation [{op.name}] deploy with incorrect device {tensor.device}. '
                            'Convert it to cpu tensor now.')
                values[idx] = tensor.cpu()

def ASSERT_NUM_OF_INPUT(op: Operation, values: List[torch.Tensor],
                        min_num_of_input: int = -1, max_num_of_input: int = 99):
    if min_num_of_input == max_num_of_input:
        if len(values) != min_num_of_input:
            raise ValueError(f'Can not feed value to operation {op.name}, '
                             f'expects exact {min_num_of_input} inputs, however {len(values)} was given')
    elif len(values) > max_num_of_input:
        raise ValueError(f'Too many input value for {op.name}, '
                         f'expects {max_num_of_input} inputs at most, however {len(values)} was given')
    elif len(values) < min_num_of_input:
        raise ValueError(f'Too few input value for {op.name}, '
                         f'expects {min_num_of_input} inputs at least, however {len(values)} was given')

def GET_ATTRIBUTE_FROM_OPERATION(op: Operation, attribute: str, compulsive: bool = False, default: Any = None):
    """Try to get an attribute from operation. If an attribute is compulsive,
    then operation must give a value of it, otherwise an error will be thrown.
    If an attribute is not compulsive, a default value will be given if
    operation.attributes do not holds a value of requesting attribute.

    Args:
        op (Operation): Operation instance.
        attribute (str): Attribute name.
        compulsive (bool): Whether is a compulsive attribute.
        default (Any, optional): [description]. default value of attribute.
    """
    if attribute in op.attributes:
        return op.attributes[attribute]
    else:
        if compulsive:
            raise KeyError(
                f'Operation {op.name} is supposed to have a value of attribute {attribute}. ',
                'However this value is missing from currecnt operation.')
        else:
            return default

def ASSERT_ALL_TENSORS_AT_SAME_DEVICE(op: Operation, values: List[torch.Tensor], device: str = None):
    """PPQ Default Backend Suppose all inputs are torch.Tensor at same device.

    This function will check all inputs tensors' device.

    IF THERE IS ANY CUDA INPUT TENSOR FOR DYNAMIC SHAPE OPERATION, THERE PROBABLY ARE SOME SYSTEM FAILURES.
    YOU ARE SUPPOSED TO REPORT THOSE SYSTEM FAILURES TO US.

    Args:
        values (List[torch.Tensor]): values to be checked.
        force_convert (bool, optional): whether to convert all input tensors to cpu.
    """
    devices = []
    for idx, tensor in enumerate(values):
        if tensor is None: continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f'Input at [{idx}] of Operation [{op.name}] is not a torch.Tensor, '
                f'a {type(tensor)} was gievn here instead, which is not supposed to happen in PPQ execution system. '
                'Is there any parsing failure with your graph?')
        devices.append(str(tensor.device))
        if devices[-1] == 'cuda:0': devices[-1] = 'cuda'
        if device is not None and devices[-1] != device:
            raise ValueError(
                f'Found input tensor with unexpected device, input tensor({idx}) of operation {op.name} is expected '
                f'as a {device} tensor, however {devices[-1]} was given')
    if any([devices[0] != d for d in devices]):
        raise ValueError(f'Input tensors do not share a same device. ({[d for d in devices]})')

def GET_VALUE_FROM_INPUTS(values: list, idx: int) -> torch.Tensor:
    assert isinstance(idx, int)
    assert idx > 0
    if len(values) > idx: return values[idx]
    else: return None

def ASSERT_IS_QUANT_OP(op: QuantableOperation):
    if not isinstance(op, QuantableOperation):
        raise TypeError(f'Given Operation is expected as a QuantableOperation, however {type(op)} was given.')

def FORCE_CONVERT_DEVICE(value: torch.Tensor, device: str) -> torch.Tensor:
    # SET LOG HERE FOR DEBUG.
    # value = value.clone()
    return value.to(device=device, copy=True)
