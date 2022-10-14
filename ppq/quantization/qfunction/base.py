from abc import ABCMeta, abstractmethod
from ppq.core import TensorQuantizationConfig
from typing import Any, Callable


class BaseQuantFunction(Callable, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @ abstractmethod
    def __call__(self, input_tensor: Any, quantization_config: TensorQuantizationConfig, **kwargs) -> Any:
        raise NotImplemented('Implement this first.')
