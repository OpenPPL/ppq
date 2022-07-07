from abc import ABCMeta
from typing import Dict, List

from ppq.core import (QuantizationStates, TensorQuantizationConfig,
                      ppq_debug_function)
from ppq.executor import QuantOPRuntimeHook
from ppq.IR import QuantableOperation, Variable

from .base import BaseTensorObserver
from .order import TorchIsotoneObserver
from .range import (TorchHistObserver, TorchMinMaxObserver, TorchMSEObserver,
                    TorchPercentileObserver)

PPQ_OBSERVER_TABLE = {
    'minmax': TorchMinMaxObserver,
    'kl': TorchHistObserver,
    'percentile': TorchPercentileObserver,
    'mse': TorchMSEObserver,
    'isotone': TorchIsotoneObserver
}


class TensorObserverFactroy():
    def __init__(self) -> None:
        raise NotImplementedError(
            'Observer Factory can not be initialized, use TensorObserverFactroy.build_observer instead.')

    @ classmethod
    def build_observer(cls, variable: Variable, config: TensorQuantizationConfig) -> BaseTensorObserver:
        algorithm = str(config.observer_algorithm.lower())
        if algorithm not in PPQ_OBSERVER_TABLE:
            raise ValueError(
                f'Observer type not understand, Except one of {PPQ_OBSERVER_TABLE.keys()}, '\
                f'while {str(algorithm)} was given.')
        return PPQ_OBSERVER_TABLE[algorithm](watch_on=variable, quant_cfg=config)


class CalibrationHook(QuantOPRuntimeHook):
    def __init__(
        self,
        operation: QuantableOperation,
        observer_table: Dict[Variable, BaseTensorObserver]
    ) -> None:
        self._operation = operation
        self._observer_table = observer_table
        super().__init__(operation, operation_meta=operation.meta_data)

    def pre_forward_hook(
        self, inputs: list, quant_inputs: list, quant_configs: List[TensorQuantizationConfig]) -> list:
        for input_var, quant_config in zip(inputs, quant_configs):
            if quant_config in self._observer_table:
                observer = self._observer_table[quant_config]
                observer.observe(input_var)
        return quant_inputs

    def post_forward_hook(
        self, outputs: list, quant_outputs: list, quant_configs: List[TensorQuantizationConfig]) -> list:
        for output_var, quant_config in zip(outputs, quant_configs):
            if quant_config in self._observer_table:
                observer = self._observer_table[quant_config]
                observer.observe(output_var)
        return quant_outputs

    def render_quantization_config(self):
        for _, observer in self._observer_table.items():
            observer.render_quantization_config()
            observer.report()

    def __str__(self) -> str:
        return ''.join([observer.__str__() + '\n' for _, observer in self._observer_table.items()])


class OperationObserver(metaclass=ABCMeta):
    def __init__(
        self,
        operation: QuantableOperation,
        monitor_parameter: bool = True,
        monitor_outputs  : bool = True,
        monitor_inputs   : bool = True
    ) -> None:

        if not isinstance(operation, QuantableOperation):
            raise TypeError(f'Only QuantableOP instance can apply an Observer, '\
                f'while {type(operation)} was given.')

        self._operation = operation
        self._hook = self.build_hook(
            monitor_parameter = monitor_parameter,
            monitor_outputs   = monitor_outputs,
            monitor_inputs    = monitor_inputs
        )

    def render_quantization_config(self):
        self.hook.render_quantization_config()

    def build_hook(self, monitor_parameter: bool,
        monitor_outputs: bool, monitor_inputs: bool) -> CalibrationHook:
        assert isinstance(self._operation, QuantableOperation)
        observer_table = {}
        for var, config in zip(
            self._operation.inputs, self._operation.config.input_quantization_config):
            if config.state == QuantizationStates.INITIAL:
                if var in self._operation.parameters and monitor_parameter:
                    observer_table[config] = TensorObserverFactroy.build_observer(var, config)
                elif monitor_inputs:
                    observer_table[config] = TensorObserverFactroy.build_observer(var, config)

        if monitor_outputs:
            for var, config in zip(
                self._operation.outputs, self._operation.config.output_quantization_config):
                if config.state == QuantizationStates.INITIAL:
                    observer_table[config] = TensorObserverFactroy.build_observer(var, config)

        return CalibrationHook(operation=self._operation, observer_table=observer_table)

    @ property
    def hook(self) -> CalibrationHook:
        return self._hook

    @ ppq_debug_function
    def report(self) -> str:
        return str(self._hook)
