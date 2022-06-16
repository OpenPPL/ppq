from typing import List
from ppq.IR import Operation
from enum import Enum


class OType(Enum):
    UNSPECIFIED   = 1
    QUANTABLE     = 2
    NONQUANTABLE  = 3
    CONTROVERSIAL = 4

class VProperty(Enum):
    ATTRIB  = 0x00000001
    VALUE   = 0x00000002
    SOI     = 0x00000004
    LOGICAL = 0x00000008
    QVALUE  = 0x00000010
    
    def __or__(self, other: int) -> int:
        return self.value + other

    def __ror__(self, other: int) -> int:
        return self.value + other

    def __and__(self, other: int) -> int:
        return self.value & other

    def __rand__(self, other: int) -> int:
        return self.value & other

    def __radd__(self, other: int) -> int:
        return self.value + other

    def __add__(self, other: int) -> int:
        return self.value + other

    def __sub__(self, other: int) -> int:
        return self - (self.value & other)

    def __rsub__(self, other: int) -> int:
        return other - (self.value & other)


class VType():
    def __init__(self, type_value: int) -> None:
        if isinstance(type_value, VProperty):
            type_value = type_value.value
        self._type = type_value
    
    def has_property(self, property: VProperty) -> bool:
        return (self._type & property.value) != 0

    def non_quantable(self) -> bool:
        return (self.has_property(VProperty.ATTRIB) or 
                self.has_property(VProperty.LOGICAL) or 
                self.has_property(VProperty.SOI))

class VLink:
    def __init__(self, input_idx: int, output_idx: int, 
                 direction: str = 'Bidirectional') -> None:
        if not isinstance(input_idx, int):
            raise TypeError(f'Can not create vlink with input_idx {input_idx}, '
                            'only int value is acceptable here.')
        if not isinstance(output_idx, int):
            raise TypeError(f'Can not create vlink with output_idx {output_idx}, '
                            'only int value is acceptable here.')
        
        if direction.lower() not in {'bidirectional', 'forward', 'backwrad'}:
            raise KeyError(f'Can not create vlink with direction {direction}, '
                           f'only "bidirectional", "forward", "backward" is acceptable here.')
        self.source_idx  = input_idx
        self.dest_idx = output_idx
        self.direction  = direction.lower()


class OpSocket:
    def __init__(self, op: Operation, cls_input: List[VProperty] = None, 
                 cls_output: List[VProperty] = None, links: List[VLink] = None) -> None:
        self.cls_input     = [VProperty.VALUE for _ in range(op.num_of_input)]
        self.cls_output    = [VProperty.VALUE for _ in range(op.num_of_output)]
        self.op            = op
        if links is None:
            links = []
            for i in range(op.num_of_input):
                for j in range(op.num_of_output):
                    links.append(VLink(i, j))
        self.links = links
        
        if cls_input is not None:
            for idx, cls in enumerate(cls_input):
                self.set_input_cls(idx, cls)

        if cls_output is not None:
            for idx, cls in enumerate(cls_output):
                self.set_output_cls(idx, cls)

    def set_input_cls(self, idx: int, cls: VProperty):
        if idx >= len(self.cls_input) or idx < 0:
            raise IndexError('Opsocket Can not initilized. Input index out of range. '
                             f'Except index between 0 and {len(self.cls_input) - 1}, while {idx} was given.')
        if not isinstance(cls, VProperty):
            raise TypeError(f'Opsocket Can not initilized. Can not set VClass for op {self.op.name}, '
                            'invalid data type was found, except a VClass instance, '
                            f'however {type(cls)} was given')
        self.cls_input[idx] = cls

    def set_output_cls(self, idx: int, cls: VProperty):
        if idx >= len(self.cls_output) or idx < 0:
            raise IndexError('Opsocket Can not initilized. Input index out of range. '
                             f'Except index between 0 and {len(self.cls_output) - 1}, while {idx} was given.')
        if not isinstance(cls, VProperty):
            raise TypeError(f'Opsocket Can not initilized. Can not set VClass for op {self.op.name}, '
                            'invalid data type was found, except a VClass instance, '
                            f'however {type(cls)} was given')
        self.cls_output[idx] = cls
