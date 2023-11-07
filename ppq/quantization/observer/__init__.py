import functools
from abc import ABCMeta
from typing import Dict, List, Optional

from ppq.core import QuantizationStates, TensorQuantizationConfig, ppq_debug_function
from ppq.executor import QuantOPRuntimeHook
from ppq.IR import QuantableOperation, Variable

from .base import BaseTensorObserver
from .floating import ConstantObserver, DirectMSEObserver
from .order import TorchIsotoneObserver
from .range import (
    TorchHistObserver,
    TorchMinMaxObserver,
    TorchMSEObserver,
    TorchPercentileObserver,
)

OBSERVER_TABLE: Dict[str, type(BaseTensorObserver)] = {
    "minmax": TorchMinMaxObserver,
    "kl": TorchHistObserver,
    "percentile": TorchPercentileObserver,
    "mse": TorchMSEObserver,
    "isotone": TorchIsotoneObserver,
    "constant": ConstantObserver,
    "floating": DirectMSEObserver,
}


class TensorObserverFactroy:
    def __init__(self) -> None:
        raise NotImplementedError(
            "Observer Factory can not be initialized, "
            "use TensorObserverFactroy.build_observer instead."
        )

    @classmethod
    def build_observer(
        cls, variable: Variable, config: TensorQuantizationConfig
    ) -> BaseTensorObserver:
        algorithm = str(config.observer_algorithm.lower())
        if algorithm not in OBSERVER_TABLE:
            raise ValueError(
                f"Observer type not understand, Except one of {OBSERVER_TABLE.keys()}, "
                f"while {str(algorithm)} was given."
            )
        return OBSERVER_TABLE[algorithm](watch_on=variable, quant_cfg=config)

    @classmethod
    def register_observer(cls, name: Optional[str] = None):
        """Register customized observer to PPQ registry.

        Args:
            name (Optional[str], optional): A query name to the observer.
                If not set, set as class name. Defaults to None.

        Example::

            @register_observer("myob")
            class MyObserver(BaseTensorObserver):
                ...
        """

        def wrapper(cls):
            @functools.wraps
            def _wraps(*args, **kwargs):
                return cls(*args, **kwargs)

            key_name = name or cls.__name__
            if key_name in OBSERVER_TABLE:
                raise KeyError(
                    f"{key_name} is found in observer table, "
                    f"existing class is {OBSERVER_TABLE[key_name]}."
                )
            if not issubclass(cls, BaseTensorObserver):
                raise TypeError(
                    "The observer must be a class derived from `BaseTensorObserver`."
                )
            OBSERVER_TABLE[key_name] = cls
            return _wraps

        return wrapper


# Fix typo
TensorObserverFactory = TensorObserverFactroy


class CalibrationHook(QuantOPRuntimeHook):
    def __init__(
        self,
        operation: QuantableOperation,
        observer_table: Dict[Variable, BaseTensorObserver],
    ) -> None:
        self._operation = operation
        self._observer_table = observer_table
        super().__init__(operation)

    def pre_forward_hook(
        self,
        inputs: list,
        quant_inputs: list,
        quant_configs: List[TensorQuantizationConfig],
    ) -> list:
        for input_var, quant_config in zip(inputs, quant_configs):
            if quant_config in self._observer_table:
                observer = self._observer_table[quant_config]
                observer.observe(input_var)
        return quant_inputs

    def post_forward_hook(
        self,
        outputs: list,
        quant_outputs: list,
        quant_configs: List[TensorQuantizationConfig],
    ) -> list:
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
        return "".join(
            [observer.__str__() + "\n" for _, observer in self._observer_table.items()]
        )


class OperationObserver(metaclass=ABCMeta):
    def __init__(
        self,
        operation: QuantableOperation,
        monitor_parameter: bool = True,
        monitor_outputs: bool = True,
        monitor_inputs: bool = True,
    ) -> None:
        if not isinstance(operation, QuantableOperation):
            raise TypeError(
                f"Only QuantableOP instance can apply an Observer, "
                f"while {type(operation)} was given."
            )

        self._operation = operation
        self._hook = self.build_hook(
            monitor_parameter=monitor_parameter,
            monitor_outputs=monitor_outputs,
            monitor_inputs=monitor_inputs,
        )

    def render_quantization_config(self):
        self.hook.render_quantization_config()

    def build_hook(
        self, monitor_parameter: bool, monitor_outputs: bool, monitor_inputs: bool
    ) -> CalibrationHook:
        assert isinstance(self._operation, QuantableOperation)
        observer_table = {}
        for var, config in zip(
            self._operation.inputs, self._operation.config.input_quantization_config
        ):
            if config.state == QuantizationStates.INITIAL:
                if var in self._operation.parameters and monitor_parameter:
                    observer_table[config] = TensorObserverFactroy.build_observer(
                        var, config
                    )
                elif monitor_inputs:
                    observer_table[config] = TensorObserverFactroy.build_observer(
                        var, config
                    )

        if monitor_outputs:
            for var, config in zip(
                self._operation.outputs,
                self._operation.config.output_quantization_config,
            ):
                if config.state == QuantizationStates.INITIAL:
                    observer_table[config] = TensorObserverFactroy.build_observer(
                        var, config
                    )

        return CalibrationHook(operation=self._operation, observer_table=observer_table)

    @property
    def hook(self) -> CalibrationHook:
        return self._hook

    @ppq_debug_function
    def report(self) -> str:
        return str(self._hook)
