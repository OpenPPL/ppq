"""
Copyright mPPQ/PPQ 2025

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Rewrite PPQ APIs.
"""

import os
from typing import Any, Callable, Collection, Dict, Iterable, Optional, Sequence

from onnx import ModelProto

from mppq.api.extension import (
    _PLATFORM_TO_DISPATCHER_,
    _PLATFORM_TO_PARSER_,
    _PLATFORM_TO_QUANTIZER_,
)
from mppq.common import DEFAULT_QUANTIZE_OP
from mppq.defs import empty_ppq_cache
from mppq.dispatcher import DISPATCHER_TABLE
from mppq.dispatcher.base import GraphDispatcher
from mppq.dispatcher.scope import IgnoredScope
from mppq.executor.base import BaseGraphExecutor, GraphInput
from mppq.frontend import EXPORTER, PARSER
from mppq.ir.base.command import GraphCommand, GraphCommandType
from mppq.ir.base.graph import BaseGraph, GraphBuilder, GraphExporter
from mppq.ir.morph import GraphFormatter, GraphMerger, GraphReplacer
from mppq.quant import TargetPrecision
from mppq.quantization.analyse import graphwise_error_analyse, layerwise_error_analyse
from mppq.quantizer import QUANTIZER, BaseQuantizer
from mppq.quantizer.base import QuantizationOptimizationPass
from mppq.utils.qfunction import ppq_fake_quant, ppq_quant_toint
from mppq.utils.round import ppq_tensor_round


def load_graph(graph_file: Any, parser: GraphBuilder) -> BaseGraph:
    r"""加载模型并解析为PPQ IR Graph.

    Args:
        graph_file: 模型文件路径，或者已加载的模型对象。
        parser: 解析模型的GraphBuilder对象。

    Returns:
        BaseGraph: 解析后的IR Graph对象。
    """

    graph = parser.build(graph_file)
    graph = format_graph(graph)
    return graph


def load_onnx_graph(onnx_file: str | os.PathLike | ModelProto) -> BaseGraph:
    r"""加载ONNX模型并解析为PPQ IR Graph.

    Args:
        onnx_file (str | ModelProto): ONNX文件路径，或者已加载的ONNX模型对象。

    Returns:
        BaseGraph: 解析后的IR Graph对象。
    """

    parser = PARSER["onnx"]()
    return load_graph(onnx_file, parser)


def export_graph(graph: BaseGraph, f: str | os.PathLike, *, exporter: GraphExporter):
    r"""导出PPQ IR Graph为目标格式。

    Args:
        graph (BaseGraph): 待导出IR Graph对象。
        f (str|os.PathLike): 导出文件路径。
        exporter (GraphExporter): 导出Graph的GraphExporter对象。
    """

    exporter.export(f, graph)


def export_onnx_graph(graph: BaseGraph, f: str | os.PathLike):
    r"""导出PPQ IR Graph为目标格式。

    Args:
        graph (BaseGraph): 待导出IR Graph对象。
        f (str|os.PathLike): 导出文件路径。
        exporter (GraphExporter): 导出Graph的GraphExporter对象。
    """

    exporter = EXPORTER["onnx"]()
    export_graph(graph, f, exporter=exporter)


def export_config(graph: BaseGraph, f: str | os.PathLike):
    r"""导出PPQ IR Graph的量化配置。
    Args:
        graph (BaseGraph): 待导出IR Graph对象。
        f (str|os.PathLike): 导出文件路径。
    """

    exporter = EXPORTER["onnx"]()
    exporter.dump_quantization_config(f, graph)


def format_graph(
    graph: BaseGraph,
    format_constant_input: bool = True,
    fuse_bias_add: bool = True,
    fuse_bn: bool = True,
    replace_bn_to_conv: bool = True,
    remove_identity: bool = True,
    remove_isolated: bool = True,
) -> BaseGraph:
    r"""这个函数对计算图进行预处理工作，其主要内容是将计算图的格式进行统一。
    这个函数将会统一 cast, slice, parameter, constant 算子的格式，
    并且执行有关 batchnorm 的合并工作.

    在 PPQ 中，我们不希望出现 Constant 算子，所有 Constant 输入将被当作 parameter variable 连接到下游算子上
    在 PPQ 中，我们不希望出现 Batchnorm 算子，所有 Batchnorm 将被合并
    在 PPQ 中，我们不希望出现权重共享的算子，所有被共享的权重将被复制分裂成多份
    在 PPQ 中，我们不希望出现孤立算子，所有孤立算子将被移除

    # 读取 Onnx 图时，将图中所有以 Constant 节点作为输入的变量转换为 Parameter Variable
    FORMATTER_FORMAT_CONSTANT_INPUT
    # 读取 Onnx 图时，合并图中的 Bias add 节点(Conv, ConvTranspose, Gemm)
    FORMATTER_FUSE_BIAS_ADD
    # 读取 Onnx 图时，合并图中的 Batchnorm 节点(Conv, ConvTranspose, Gemm)
    FORMATTER_FUSE_BN
    # 读取 Onnx 图时，将单独的 Batchnorm 替换为卷积
    FORMATTER_REPLACE_BN_TO_CONV
    # 读取 Onnx 图时，移除图中的 Identity 节点
    FORMATTER_REMOVE_IDENTITY
    # 读取 Onnx 图时，移除图中的孤立节点
    FORMATTER_REMOVE_ISOLATED
    """

    # do graph level optimization
    formatter = GraphReplacer(GraphFormatter(GraphMerger(graph)))
    if format_constant_input:
        formatter(GraphCommand(GraphCommandType.FORMAT_CONSTANT_INPUT))
    formatter(GraphCommand(GraphCommandType.CONVERT_TO_TENSOR))
    formatter(GraphCommand(GraphCommandType.FORMAT_PARAMETERS))

    if fuse_bias_add:
        formatter(GraphCommand(GraphCommandType.FUSE_BIAS_ADD))

    if fuse_bn:
        formatter(GraphCommand(GraphCommandType.FUSE_BN))

    if replace_bn_to_conv:
        formatter(GraphCommand(GraphCommandType.REPLACE_BATCHNORM_TO_CONV))

    formatter(GraphCommand(GraphCommandType.FORMAT_CAST))
    formatter(GraphCommand(GraphCommandType.FORMAT_SLICE))
    formatter(GraphCommand(GraphCommandType.FORMAT_CLIP))
    formatter(GraphCommand(GraphCommandType.FORMAT_PAD))
    formatter(GraphCommand(GraphCommandType.FORMAT_RESIZE))

    if remove_identity:
        formatter(GraphCommand(GraphCommandType.REMOVE_IDENTITY))

    if remove_isolated:
        formatter(GraphCommand(GraphCommandType.DELETE_ISOLATED))

    return graph


def dispatch_graph(
    graph: BaseGraph,
    quantize_operations: Collection[str] = DEFAULT_QUANTIZE_OP,
    dispatcher: Optional[str | GraphDispatcher] = None,
    dispatching_override: Optional[Dict[str, TargetPrecision]] = None,
    ignored_scope: Optional[dict | list | IgnoredScope] = None,
    quant_precision: TargetPrecision = TargetPrecision.INT8,
    **kwargs,
) -> BaseGraph:
    """Override the graph dispatching of PPQ. This function split graph into pieces.

    Dispatching results are assigned to each operator in the graph.

    Args:
        graph (BaseGraph): loaded onnx graph
        quantize_operations (Set[str]): an operation list that can be quantized.
        dispatcher (str | GraphDispatcher, optional): dispatch algorithm.
            Defaults to "conservative".
        dispatching_override (DispatchingTable, optional): overwrite the dispatching
            results. Defaults to None.

    Returns:
        BaseGraph: graph that each operator is assigned a dispatching target.
    """

    if isinstance(dispatcher, str):
        dispatcher = dispatcher.lower()
        if dispatcher not in DISPATCHER_TABLE:
            raise ValueError(
                f'Can not found dispatcher type "{dispatcher}", check your input again.'
            )
        dispatcher = DISPATCHER_TABLE[dispatcher](graph)
    else:
        if not isinstance(dispatcher, GraphDispatcher):
            raise TypeError(
                'Parameter "dispachter" of function ppq.api.dispatch_graph must be '
                f"str or GraphDispatcher, however {type(dispatcher)} was given."
            )
    assert isinstance(dispatcher, GraphDispatcher)
    if not quantize_operations:
        quantize_operations = DEFAULT_QUANTIZE_OP

    dispatching_table = dispatcher.dispatch(
        graph=graph,
        quant_types=quantize_operations,
        quant_precision=quant_precision,
        fp32_platform=TargetPrecision.FP32,
        soi_platform=TargetPrecision.SOI,
        **kwargs,
    )

    if ignored_scope is not None:
        if isinstance(ignored_scope, dict):
            ignored_scope = IgnoredScope(**ignored_scope.pop("type"))
        if isinstance(ignored_scope, list):
            ignored_scope = IgnoredScope(operations=ignored_scope)
        assert isinstance(ignored_scope, IgnoredScope)
        ignored_table = ignored_scope.dispatch(graph)
        dispatching_table.update(ignored_table)

    # override dispatching result
    if dispatching_override is not None:
        if not isinstance(dispatching_override, dict):
            raise TypeError(
                'Parameter "dispatching_override" of function ppq.api.dispatch_graph '
                f"must be a dict, however {type(dispatching_override)} was given."
            )
        for opname, platform in dispatching_override.items():
            if opname not in graph.operations:
                continue
            if not isinstance(platform, int):
                raise TypeError(
                    "Your dispatching table contains a invalid setting of "
                    f"operation {opname}. All platform setting given in dispatching "
                    f"table is expected given as int, however {type(platform)} was "
                    "given."
                )
            dispatching_table[opname] = TargetPrecision(platform)

    for operation in graph.operations.values():
        if operation.name not in dispatching_table:
            raise RuntimeError(
                f"Internal Error, Can not find operation {operation.name} in "
                "dispatching table."
            )
        operation.precision = dispatching_table[operation.name]
    return graph


def load_quantizer(
    model: str | os.PathLike | object,
    platform: int,
    dispatcher: Optional[str] = None,
    dispatching_override: Optional[Dict[str, TargetPrecision]] = None,
    ignored_scope: Optional[list | IgnoredScope] = None,
    quant_precision: TargetPrecision = TargetPrecision.INT8,
    **kwargs,
) -> BaseQuantizer:
    r"""Quantize the graph to target platform.

    Args:
        graph (BaseGraph): loaded onnx graph
        quantize_operations (Set[str]): an operation list that can be quantized.
        dispatcher (str | GraphDispatcher, optional): dispatch algorithm.
            Defaults to "conservative".
        dispatching_override (DispatchingTable, optional): overwrite the dispatching
            results. Defaults to None.
        ignored_scope (list | IgnoredScope, optional): a list of operation names
            that will be ignored during quantization. Defaults to None.
    """

    parser = PARSER[_PLATFORM_TO_PARSER_[platform]]()
    graph = load_graph(model, parser)
    if dispatcher is None:
        dispatcher = _PLATFORM_TO_DISPATCHER_[platform]
    quantizer = QUANTIZER[_PLATFORM_TO_QUANTIZER_[platform]](graph)
    graph = dispatch_graph(
        graph,
        quantizer.quant_operation_types,
        dispatcher,
        dispatching_override,
        ignored_scope,
        quant_precision,
        **kwargs,
    )

    return quantizer


def quantize(
    model: str | os.PathLike,
    f: str | os.PathLike,
    platform: int,
    quant_precision: TargetPrecision = TargetPrecision.INT8,
    example_inputs: Optional[GraphInput] = None,
    calib_dataloader: Optional[Iterable[Any]] = None,
    executor: Optional[BaseGraphExecutor] = None,
    collate_fn: Optional[Callable[[Any], Any]] = None,
    calib_steps: int = 32,
    prequant_settings: Optional[Sequence[dict | QuantizationOptimizationPass]] = None,
    quant_settings: Optional[Sequence[dict | QuantizationOptimizationPass]] = None,
    dispatcher: Optional[str] = None,
    ignored_scope: Optional[list | IgnoredScope] = None,
    **kwargs,
) -> None:
    r"""Quantize the onnx model to target platform and export the quantized model.

    Args:
        model (str | os.PathLike): onnx model path.
        f (str | os.PathLike): output file path.
        platform (int): target platform.
        dispatcher (str | GraphDispatcher, optional): dispatch algorithm.
            Defaults to "conservative".
        ignored_scope (list | IgnoredScope, optional): a list of operation names
            that will be ignored during quantization. Defaults to None.
    """

    parser = PARSER[_PLATFORM_TO_PARSER_[platform]]()
    graph = load_graph(model, parser)
    if dispatcher is None:
        dispatcher = _PLATFORM_TO_DISPATCHER_[platform]
    quantizer = QUANTIZER[_PLATFORM_TO_QUANTIZER_[platform]](graph)
    graph = dispatch_graph(
        graph,
        quantizer.quant_operation_types,
        dispatcher,
        ignored_scope=ignored_scope,
        quant_precision=quant_precision,
        **kwargs,
    )
    quantizer.quantize(
        example_inputs=example_inputs,
        calib_dataloader=calib_dataloader,
        executor=executor,
        collate_fn=collate_fn,
        calib_steps=calib_steps,
        prequant_settings=prequant_settings,
        quant_settings=quant_settings,
    )
    exporter = EXPORTER[_PLATFORM_TO_PARSER_[platform]]()
    export_graph(graph, f, exporter=exporter)


__all__ = [
    "load_graph",
    "load_onnx_graph",
    "export_graph",
    "export_onnx_graph",
    "export_config",
    "format_graph",
    "dispatch_graph",
    "load_quantizer",
    "quantize",
    "empty_ppq_cache",
    "graphwise_error_analyse",
    "layerwise_error_analyse",
    "ppq_tensor_round",
    "ppq_fake_quant",
    "ppq_quant_toint",
]
