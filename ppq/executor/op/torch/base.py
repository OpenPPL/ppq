from typing import Any, List

from ppq.core import TargetPlatform
from ppq.IR import Operation
from ppq.IR.quantize import QuantableOperation

import torch


class TorchBackendContext:
    def __init__(self, executing_device: str) -> None:
        self.executing_device = executing_device

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

def VALUE_TO_EXECUTING_DEVICE(op: Operation, ctx: TorchBackendContext, values: List[torch.Tensor]) -> List[torch.Tensor]:
    if ctx is None: device = values[0].device
    else: device = ctx.executing_device
    for idx, (plat, value) in enumerate(zip(op.socket.in_plat, values)):
        if value is None: continue
        if plat == TargetPlatform.SOI or op.platform == TargetPlatform.SOI:
            values[idx] = value.cpu()
        else: values[idx] = value.to(device)
    return values