"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, Dict, Optional, Type

import torch

from mppq.data import DataType
from mppq.defs import empty_ppq_cache
from mppq.executor.base import BaseGraphExecutor, GraphInput
from mppq.executor.torch import TorchExecutor
from mppq.ir.base.command import QuantizeOperationCommand
from mppq.ir.base.graph import BaseGraph, Operation
from mppq.ir.base.quantize import QuantableOperation, QuantableVariable
from mppq.ir.deploy import QuantableGraph
from mppq.ir.morph import GraphReplacer
from mppq.logger import error, info, warning
from mppq.quant import OperationQuantizationConfig, QuantizationStates, TargetPrecision
from mppq.quantization.optim.base import (
    OPTIM_ALGORITHMS,
    QuantizationOptimizationPass,
    QuantizationOptimizationPipeline,
)
from mppq.register import Registry


class BaseQuantizer(metaclass=ABCMeta):
    r"""量化起的基类。提供了一个基本的量化框架。

    需要实现以下接口：

    1. init_quantize_config
       实现算子的量化配置的初始化

    2. quant_operation_types (property)
       所支持的量化算子类型

    可选择实现以下接口：

    1. default_prequant_pipeline (property)
       当未指定时，默认的量化前算法

    2. default_quant_pipeline (property)
       当未指定时，默认的量化算法
    """

    def __init__(self, graph: BaseGraph, verbose: bool = True) -> None:
        if not isinstance(graph, BaseGraph):
            raise TypeError(
                "To initialize a Quantizer, a BaseGraph instance is needed. "
                f"While {type(graph)} was given."
            )
        self._verbose = verbose
        self._graph = graph
        self._processor = QuantableGraph(GraphReplacer(self._graph))

    @property
    def graph(self) -> BaseGraph:
        return self._graph

    @empty_ppq_cache
    def quantize(
        self,
        example_inputs: Optional[GraphInput] = None,
        calib_dataloader: Optional[Iterable[Any]] = None,
        executor: Optional[BaseGraphExecutor] = None,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        calib_steps: int = 32,
        prequant_settings: Optional[
            Sequence[dict | QuantizationOptimizationPass]
        ] = None,
        quant_settings: Optional[Sequence[dict | QuantizationOptimizationPass]] = None,
    ) -> BaseGraph:
        r"""Quantize the graph."""

        if executor is None:
            executor = TorchExecutor(self._graph)
        if example_inputs is None:
            new_example_inputs: Dict[str, torch.Tensor] = {}
            for k, v in self._graph.inputs.items():
                dtype = DataType.to_torch(v.dtype)
                shape = [int(i) for i in v.shape]
                new_example_inputs[k] = torch.zeros(shape, dtype=dtype)
            example_inputs = new_example_inputs  # type: ignore
        if calib_dataloader is None:
            calib_dataloader = [example_inputs]
        assert example_inputs is not None
        assert isinstance(executor, BaseGraphExecutor)

        # step - 1, prequant pipeline:
        # prequant pipeline will change your network structure and float value.
        if prequant_settings is None:
            prequant_pipeline = self.default_prequant_pipeline
        else:
            prequant_pipeline = self._build_pipeline(prequant_settings)
        prequant_pipeline.optimize(
            graph=self._graph,
            dataloader=calib_dataloader,
            executor=executor,
            collate_fn=collate_fn,
            calib_steps=calib_steps,
            verbose=self._verbose,
        )

        # step - 2, quantize all operations
        executor.load_graph(self._graph)
        executor.tracing_operation_meta(inputs=example_inputs)

        for op_name, operation in self._graph.operations.items():
            if operation.precision == TargetPrecision.UNSPECIFIED:
                error(f"no precision info for node {op_name}")
                raise ValueError("have you call a correct dispatcher?")
            self._wrap_to_quantized_operation(op_name)

        # quantize operation will modify network structure
        # it is necessary calling self._executor before further execution
        # step - 3, calling graph optimization pipeline
        executor.load_graph(self._graph)
        if quant_settings is None:
            quant_pipeline = self.default_quant_pipeline
        else:
            quant_pipeline = self._build_pipeline(quant_settings)

        quant_pipeline.optimize(
            graph=self._graph,
            dataloader=calib_dataloader,
            executor=executor,
            collate_fn=collate_fn,
            calib_steps=calib_steps,
            verbose=self._verbose,
        )

        if self._verbose:
            info(f"{self}")
            info("Network Quantization Finished.")
        return self._graph

    def _wrap_to_quantized_operation(self, op_name: str) -> Operation:
        if op_name not in self._graph.operations:
            raise KeyError(
                f"Can not find op {op_name} in your graph, check operation name again."
            )
        converting_operation = self._graph.operations[op_name]
        if isinstance(converting_operation, QuantableOperation):
            warning(f"Operation {op_name} has been quantized.")
            return converting_operation

        precision = converting_operation.precision
        if precision in {TargetPrecision.FP32, TargetPrecision.SOI}:
            return self._graph.operations[op_name]

        if (
            precision == TargetPrecision.UNSPECIFIED
            and converting_operation.type not in self.quant_operation_types
        ):
            return self._graph.operations[op_name]

        # create quantize config and convert operation.
        self._processor(
            QuantizeOperationCommand(
                op_name=op_name,
                target_precision=precision,
                config=self.init_quantize_config(operation=converting_operation),
            )
        )
        return self._graph.operations[op_name]

    def _build_pipeline(
        self, configurations: Sequence[dict | QuantizationOptimizationPass]
    ):
        r"""Build a pipeline of optimization algorithms."""
        list_of_passes = []
        for optim in configurations:
            if isinstance(optim, QuantizationOptimizationPass):
                list_of_passes.append(optim)
                continue
            assert isinstance(optim, dict) and "type" in optim
            algo = OPTIM_ALGORITHMS[optim.pop("type")]
            list_of_passes.append(algo(**optim))
        return QuantizationOptimizationPipeline(list_of_passes)

    @abstractmethod
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        r"""Return a query to the operation how it should be quantized."""
        raise NotImplementedError

    @property
    def quant_operation_types(self) -> Collection[str]:
        r"""Return a collection of name that operations should be quantized."""
        return set()  # all operations are quantized by default.

    @property
    def default_prequant_pipeline(self) -> QuantizationOptimizationPipeline:
        r"""A simplified API to return a default quantization pipeline."""
        return QuantizationOptimizationPipeline([])

    @property
    def default_quant_pipeline(self) -> QuantizationOptimizationPipeline:
        r"""A simplified API to return a default quantization pipeline."""
        raise NotImplementedError

    def __repr__(self) -> str:
        debug_str = ""
        # stats:
        quant_ops = [
            op
            for op in self._graph.operations.values()
            if isinstance(op, QuantableOperation)
        ]
        quant_vars = [
            var
            for var in self._graph.variables.values()
            if isinstance(var, QuantableVariable)
        ]
        quant_cfgs = []

        config_states_cnt = {state: 0 for state in QuantizationStates}
        for op in quant_ops:
            for cfg, _ in op.config_with_variable:
                config_states_cnt[cfg.state] += 1
                quant_cfgs.append(cfg)

        debug_str += "--------- Network Snapshot ---------\n"
        debug_str += f"Num of Op:                    [{len(self._graph.operations)}]\n"
        debug_str += f"Num of Quantized Op:          [{len(quant_ops)}]\n"
        debug_str += f"Num of Variable:              [{len(self._graph.variables)}]\n"
        debug_str += f"Num of Quantized Var:         [{len(quant_vars)}]\n"
        debug_str += "------- Quantization Snapshot ------\n"
        debug_str += f"Num of Quant Config:          [{len(quant_cfgs)}]\n"
        for state, cnt in config_states_cnt.items():
            if cnt <= 0:
                continue
            padding = max(28 - len(state.name), 0)
            debug_str += f"{state.name}: [{cnt}]{'.':<{padding}}"
        return debug_str


QUANTIZER: Registry[Type[BaseQuantizer]] = Registry("QUANTIZER")
