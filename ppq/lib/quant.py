from typing import Callable, List

import torch
from ppq.core import (NetworkFramework, QuantizationPolicy,
                      QuantizationProperty, RoundingPolicy, TargetPlatform,
                      TensorQuantizationConfig)
from ppq.executor.torch import OPERATION_FORWARD_TABLE
from ppq.IR import BaseGraph, GraphExporter, Variable
from ppq.quantization.observer import BaseTensorObserver, TensorObserverFactroy
from ppq.quantization.optim import (QuantizationOptimizationPass,
                                    QuantizationOptimizationPipeline)
from ppq.quantization.qfunction import PPQuantFunction as QuantFunction
from ppq.quantization.quantizer import BaseQuantizer
from ppq.scheduler import DISPATCHER_TABLE, GraphDispatcher

from .common import __EXPORTERS__, __PARSERS__, __QUANTIZER_COLLECTION__


def Quantizer(platform: TargetPlatform, graph: BaseGraph) -> BaseQuantizer:
    """
    Get a pre-defined Quantizer corresponding to your platform.
    Quantizer in PPQ initializes Tensor Quantization Config for each Operation,
        - it describes how operations are going to be quantized.
    
    根据目标平台获取一个系统预定义的量化器。
    
    ## 量化器
    在 PPQ 中，量化器是一个用于为算子初始化量化信息 Tensor Quantization Config 的对象
        - 量化器决定了你的算子是如何被量化的，你也可以设计新的量化器来适配不同的后端推理框架
    
    在 PPQ 中我们为不同的推理后端设计好了一些预定义的量化器，你可以通过 ppq.foundation.Quantizer 来访问它们
    """
    if platform not in __QUANTIZER_COLLECTION__:
        raise KeyError(f'Target Platform {platform} has no related quantizer for now.')
    return __QUANTIZER_COLLECTION__[platform](graph)


def Pipeline(optims: List[QuantizationOptimizationPass]) -> QuantizationOptimizationPipeline:
    """
    
    Build a Pipeline with given Optimization Passes Collection

    Args:
        optims (List[QuantizationOptimizationPass]): A collection of optimization passes
    """
    return QuantizationOptimizationPipeline(optims)


def Observer(
    quant_config: TensorQuantizationConfig, 
    variable: Variable = None) -> BaseTensorObserver:
    """
    Get a Tensor Observer.

    Args:
        quant_config (TensorQuantizationConfig): _description_
        variable (Variable, optional): _description_. Defaults to None.

    Returns:
        BaseTensorObserver: _description_
    """
    return TensorObserverFactroy.build_observer(variable=variable, config=quant_config)


class TensorQuant(torch.nn.Module):
    def __init__(self, quant_config: TensorQuantizationConfig) -> None:
        """
        PPQ Tensor Quant

        Args:
            quant_config (TensorQuantizationConfig): _description_
            name (str, optional): _description_. Defaults to 'PPQ Quant Stub'.
        """
        self._quant_config   = quant_config
        self._delegator      = None
        self._batch_observed = 0
        self._observer       = Observer(quant_config=quant_config)

    @ property
    def delegator(self) -> Callable:
        return self._delegator

    @ delegator.setter
    def delegator(self, func: Callable):
        self._delegator = func

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self._delegator is not None:
            return self._delegator(value, self._quant_config)
        return QuantFunction(tensor=value, config=self._quant_config)

    def observe(self, value: torch.Tensor):
        self._batch_observed += 1
        self._observer.observe(value)

    def render(self):
        if self._batch_observed == 0:
            raise PermissionError('You have not provide any data to this QuantStub, '
                                  'PPQ can not render its quant config yet.')
        self._observer.render_quantization_config()


class ParameterQuant(TensorQuant):
    def __init__(self, quant_config: TensorQuantizationConfig, parameter: torch.Tensor) -> None:
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f'Expect a torch.Tensor here. However {type(parameter)} was given.')
        
        super().__init__(quant_config)
        self.observe(parameter)
        self.render()


def LinearQuantizationConfig(
    symmetrical: bool = True,
    dynamic: bool = False,
    power_of_2: bool = False,
    channel_axis: int = None,
    quant_min: int = -128,
    quant_max: int = 127,
    num_of_bits = 8,
    calibration: str = 'minmax',
    rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN) -> TensorQuantizationConfig:

    sym = QuantizationProperty.SYMMETRICAL if symmetrical else QuantizationProperty.ASYMMETRICAL
    dyn = QuantizationProperty.DYNAMIC if dynamic else 0
    pw2 = QuantizationProperty.POWER_OF_2 if power_of_2 else 0
    chn = QuantizationProperty.PER_TENSOR if channel_axis is None else QuantizationProperty.PER_CHANNEL

    return TensorQuantizationConfig(
        policy = QuantizationPolicy(sym + dyn + pw2 + chn + QuantizationProperty.LINEAR),
        rounding = rounding,
        num_of_bits = num_of_bits,
        quant_min = quant_min,
        quant_max = quant_max,
        observer_algorithm = calibration,
        channel_axis=channel_axis)


def FloatingQuantizationConfig(
    symmetrical: bool = True,
    power_of_2: bool = True,
    channel_axis: int = None,
    quant_min: float = -448.0,
    quant_max: float = 448.0,
    exponent: int = 4,
    mantissa: int = 3,
    calibration: str = 'constant',
    rounding: RoundingPolicy = RoundingPolicy.ROUND_HALF_EVEN) -> TensorQuantizationConfig:

    sym = QuantizationProperty.SYMMETRICAL if symmetrical else QuantizationProperty.ASYMMETRICAL
    pw2 = QuantizationProperty.POWER_OF_2 if power_of_2 else 0
    chn = QuantizationProperty.PER_TENSOR if channel_axis is None else QuantizationProperty.PER_CHANNEL

    return TensorQuantizationConfig(
        policy = QuantizationPolicy(sym + pw2 + chn + QuantizationProperty.FLOATING),
        rounding = rounding,
        num_of_bits = exponent + mantissa + 1,
        exponent_bits = exponent,
        quant_min = quant_min,
        quant_max = quant_max,
        observer_algorithm = calibration)


def Dispatcher(graph: BaseGraph, method: str='conservative') -> GraphDispatcher:
    if method not in DISPATCHER_TABLE:
        raise KeyError(f'Can not find a dispatcher named {method}, check your input again.')
    return DISPATCHER_TABLE[method](graph)


def OperationForwardFunction(optype: str, platform: TargetPlatform) -> Callable:
    if not isinstance(platform, TargetPlatform):
        raise TypeError('Wrong parameter type for invoking this function.')
    if optype not in OPERATION_FORWARD_TABLE[platform]:
        raise KeyError(f'Can not find a forward function related to optype {optype}({platform.name}),'
                       ' Register it first.')
    return OPERATION_FORWARD_TABLE[platform][optype]


def Exporter(platform: TargetPlatform) -> GraphExporter:
    if not isinstance(platform, TargetPlatform):
        raise TypeError('Wrong parameter type for invoking this function.')
    if platform not in __EXPORTERS__:
        raise KeyError(f'Platfrom {platform.name} has no related exporter, register a exporter for it first.')
    return __EXPORTERS__[platform]()


def Parser(framework: NetworkFramework) -> GraphExporter:
    if not isinstance(framework, NetworkFramework):
        raise TypeError('Parameter framework has invalid type, Check your parameter again.')
    if framework not in __PARSERS__:
        raise KeyError(f'Requiring framework {framework} does not support parsing now.')
    return __PARSERS__[framework]()


__all__ = [
    'Parser', 'Exporter', 'OperationForwardFunction', 
    'Dispatcher', 'FloatingQuantizationConfig', 'LinearQuantizationConfig', 
    'QuantStub', 'Quantizer', 'Observer', 'Pipeline', 'QuantFunction']
