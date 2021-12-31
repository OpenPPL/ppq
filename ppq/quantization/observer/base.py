from abc import ABCMeta, abstractmethod
from typing import Any

from ppq.core import (QuantizationStates, TensorQuantizationConfig,
                      ppq_debug_function)
from ppq.IR import Variable


class BaseTensorObserver(metaclass=ABCMeta):
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        self._watch_on = watch_on
        self._quant_cfg = quant_cfg

    @ abstractmethod
    def observe(self, value: Any):
        raise NotImplementedError('Implement this function first.')

    @ abstractmethod
    def render_quantization_config(self):
        raise NotImplementedError('Implement this function first.')

    def __str__(self) -> str:
        return 'PPQ Tensor Observer (' + self.__class__.__name__ + ') mount on variable ' + \
            self._watch_on.name + ' observing algorithm: ' + self._quant_cfg.observer_algorithm

    @ ppq_debug_function
    def report(self) -> str:
        if self._quant_cfg.state == QuantizationStates.ACTIVATED:
            return f'Observer on Variable {self._watch_on.name}, '\
                f'computed scale: {self._quant_cfg.scale}, computed offset: {self._quant_cfg.offset}\n'
