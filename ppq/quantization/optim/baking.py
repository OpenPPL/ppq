from typing import Iterable

from ppq.core import empty_ppq_cache
from ppq.executor import BaseGraphExecutor
from ppq.IR import BaseGraph, QuantableOperation
from ppq.quantization.qfunction.linear import PPQLinearQuantFunction

from .base import QuantizationOptimizationPass


class ParameterBakingPass(QuantizationOptimizationPass):
    """ParameterBakingPass is a useful tool for quantization simulation
    acceleration. By default quantizer will bake network parameters once all
    quantization procedures are finished. For a typical Convolution layer or
    Gemm layer, which has a non-empty bias tensor, ParameterBakingPass will
    speed up the layer execution by 30%-50%.

    ParameterBakingPass will rewrite layer parameters with their quantized version,
        the quantization procedure will strictly follow layer quantization configuration.
    Once the quantization process finished, this pass will change all parameter quantization configuration states
        to QuantizationStates.BAKED.

    State QuantizationStates.BAKED indicates corresponding tensor has been pre-quantized and its value
        can be used without further quantization, executor will directly use a baked value during execution.

    ATTENTION: value is baked inplace, so to say it will rewrite all network parameters.
    ATTENTION: For platforms using int32 accumulator, a float32 bias tensor might lose precision
        during the simulation. If you want PPQ simulator to have a consistent result with hardware, it is
        highly-recommended to calling ParameterBakingPass before deployment, baking procedure will limit bias
        precision to 23 bits (float32 only has 23 fraction bits).
    Args:
        quantize_function (BaseQuantFunction): a BaseQuantFunction instance to quantize all parameters.
    """
    def __init__(self) -> None:
        super().__init__(name='PPQ Parameter Baking Pass')
        self._quantize_function = PPQLinearQuantFunction

    @ empty_ppq_cache
    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        **kwargs
    ) -> None:

        for _, operation in graph.operations.items():
            if not isinstance(operation, QuantableOperation): continue
            operation.baking_parameters(self._quantize_function)
