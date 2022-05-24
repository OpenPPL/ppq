from typing import Dict, Set

from ppq.core import TargetPlatform
from ppq.IR import BaseGraph
from ppq.IR.search import SearchableGraph

from .base import (GraphDispatcher, SOI_generators, SOI_receivers,
                   reverse_tracing_pattern, value_tracing_pattern)


class AggresiveDispatcher(GraphDispatcher):
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable operations,
        Shape-Or-Index related operations and remaining operations, all sub classes of GraphDispatcher will
        give an implementation of function "dispatch" to send all operations to their proper platform.

    ATTENTION: platform attribute will greatly affect quantizer's quantization logic, and the execution result.
        If operation is sent to a quantable platform, then its inputs and outputs will be quantized if necessary.
        if operation is classified as shape-or-index related operation, then its execution will be taken with cpu.
        if operation is sent to a fp32 platform, then its inputs and outputs shall never be quantized.

    ATTENTION: this dispatcher will insert necessary DeviceSwitch operations
        between shape-or-index operations and others.
    """
    @staticmethod
    def dispatch(
        graph: BaseGraph, quant_types: Set[str],
        quant_platform: TargetPlatform,
        fp32_platform: TargetPlatform,
        SOI_platform: TargetPlatform,
        **kwargs
        ) -> Dict[str, TargetPlatform]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_platform - all quantable parts of graph will be dispatched to this platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched to this platform.
            fp32_platform  - there are some operations receiving results from both quant_platform and SOI_platform,
                they will be dispatched to fp32_platform.

        ATTENTION: Quantization follows this dispatching,
            and only the operations within quantable platform will be quantized in the future.

        ATTENTION: this dispatcher will insert necessary DeviceSwitch operations between
            shape-or-index operations and others.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this dispatcher.

            quant_types(Set[str]): all quantable types for given platforms.

            quant_platform (TargetPlatform):
                platform object where quantable parts will goes to.

            fp32_platform (TargetPlatform):
                platform object where SOI parts will goes to.

            SOI_platform (TargetPlatform):
                platform object where remaining parts will goes to.

        Returns:
            Dict[str, TargetPlatform]: [description]
        """

        recivers, generators = SOI_receivers(graph), SOI_generators(graph)
        search_engine, SOI_operations, FP32_operations = SearchableGraph(graph), set(recivers), set()

        quant_operations = search_engine.opset_matching(
            sp_expr = lambda x: x.is_computing_op,
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: x.type in {'Shape', 'TopK', 'NonMaxSuppression'} or x.is_boundary,
            direction = 'down')
        # remove shape operations from computing ops.
        quant_operations.filter(lambda x: x.type == 'Shape')

        # we assume all 'Shape', 'NonMaxSuppression', 'ConstantOfShape', 'Topk' operations are SOI generators.
        shape_forward_matching = search_engine.opset_matching(
            sp_expr = lambda x: x in generators and x.type not in {'Constant'},
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: x in recivers or x in quant_operations or x.is_boundary,
            direction = 'down')

        # update matchings, ready for further searching.
        SOI_operations.update(shape_forward_matching)
        generators.update(SOI_operations)

        while True:
            # there are some particular cases where a single matching can not handle.
            # to cover all shape-related operations, a reverse matching is required.
            shape_backward_matching = search_engine.opset_matching(
                sp_expr = lambda x: x in SOI_operations and x.type != 'Shape' and not x in quant_operations,
                rp_expr = reverse_tracing_pattern,
                ep_expr = lambda x: x in generators or x in quant_operations or x.is_boundary,
                direction = 'up')

            if all([(op in SOI_operations) for op in shape_backward_matching]): break
            # update matchings
            SOI_operations.update(shape_backward_matching)

        # generate dispatching table.
        dispatching_table = {}
        for operation in graph.operations.values():
            if operation in SOI_operations:
                dispatching_table[operation.name] = SOI_platform
            elif operation in quant_operations:
                dispatching_table[operation.name] = quant_platform
            else:
                dispatching_table[operation.name] = fp32_platform

            # move Topk, Shape, NonMaxSuppression to platform as same as their input.
            if operation.type in {'Shape', 'TopK', 'NonMaxSuppression'}:
                if operation.inputs[0].source_op is not None:
                    dispatching_table[operation.name] = operation.inputs[0].source_op.platform
                else: dispatching_table[operation.name] = quant_platform

            # move activations to the platform same as their input.
            if operation.is_linear_activation:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[source_op.name]

        return dispatching_table


class ConservativeDispatcher(GraphDispatcher):
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable operations,
        Shape-Or-Index related operations and remaining operations, all sub classes of GraphDispatcher will
        give an implementation of function "dispatch" to send all operations to their proper platform.

    Conservative Dispatcher cuts graph in a conservative way, which means it takes as much as possible operations
        into fp32 platform.

    ATTENTION: platform attribute will greatly affect quantizer's quantization logic, and the execution result.
        If operation is sent to a quantable platform, then its inputs and outputs will be quantized if necessary.
        if operation is classified as shape-or-index related operation, then its execution will be taken with cpu.
        if operation is sent to a fp32 platform, then its inputs and outputs shall never be quantized.

    ATTENTION: this dispatcher will insert necessary DeviceSwitch operations
        between shape-or-index operations and others.
    """
    @staticmethod
    def dispatch(
        graph: BaseGraph, quant_types: Set[str],
        quant_platform: TargetPlatform,
        fp32_platform: TargetPlatform,
        SOI_platform: TargetPlatform, **kwargs
        ) -> Dict[str, TargetPlatform]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_platform - all quantable parts of graph will be dispatched to this platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched to this platform.
            fp32_platform  - there are some operations receiving results from both quant_platform and SOI_platform,
                they will be dispatched to fp32_platform.

        ATTENTION: Quantization follows this dispatching,
            and only the operations within quantable platform will be quantized in the future.

        ATTENTION: this dispatcher will insert necessary DeviceSwitch operations between
            shape-or-index operations and others.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this dispatcher.

            quant_types(Set[str]): all quantable types for given platforms.

            quant_platform (TargetPlatform):
                platform object where quantable parts will goes to.

            fp32_platform (TargetPlatform):
                platform object where SOI parts will goes to.

            SOI_platform (TargetPlatform):
                platform object where remaining parts will goes to.

        Returns:
            Dict[str, TargetPlatform]: [description]
        """
        recivers, generators = SOI_receivers(graph), SOI_generators(graph)
        search_engine, SOI_operations = SearchableGraph(graph), set(recivers)

        quant_operations = search_engine.opset_matching(
            sp_expr = lambda x: x.is_computing_op,
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: (x.type not in quant_types) or x.is_boundary,
            direction = 'down')
        quant_operations.filter(lambda x: x.type not in quant_types)

        computing_extensions = search_engine.opset_matching(
            sp_expr = lambda x: x.is_computing_op,
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: x.type in {'Shape', 'TopK', 'NonMaxSuppression'} or x.is_boundary,
            direction = 'down')

        # we assume all 'Shape', 'NonMaxSuppression', 'ConstantOfShape', 'Topk' operations are SOI generators.
        shape_forward_matching = search_engine.opset_matching(
            sp_expr = lambda x: x in generators and x.type not in {'Constant'},
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: (x in recivers or
                                 x in quant_operations or
                                 x.is_boundary or
                                 x.is_computing_op),
            direction = 'down')

        # remove computing operations and quant operations from matching
        shape_forward_matching.filter(lambda x: x.is_computing_op or x in quant_operations)

        # update matchings, ready for further searching.
        SOI_operations.update(shape_forward_matching)

        while True:
            # there are some particular cases where a single matching can not handle.
            # to cover all shape-related operations, a reverse matching is required.
            shape_backward_matching = search_engine.opset_matching(
                sp_expr = lambda x: x in SOI_operations and x.type != 'Shape',
                rp_expr = reverse_tracing_pattern,
                ep_expr = lambda x: (x in SOI_operations or
                                     x in quant_operations or
                                     x.is_boundary or
                                     x.is_computing_op),
                direction = 'up')

            # remove computing operations and quant operations from matching
            shape_backward_matching.filter(lambda x: x.is_computing_op or x in quant_operations)

            if all([(op in SOI_operations) for op in shape_backward_matching]): break

            # update matchings
            SOI_operations.update(shape_backward_matching)

        # generate dispatching table.
        dispatching_table = {}
        for operation in graph.operations.values():
            if operation in SOI_operations and operation not in computing_extensions:
                dispatching_table[operation.name] = SOI_platform
            elif operation in quant_operations:
                dispatching_table[operation.name] = quant_platform
            else:
                dispatching_table[operation.name] = fp32_platform

        for operation in graph.operations.values():
            # move Topk, Shape, NonMaxSuppression to the platform same as their input.
            if operation.type in {'Shape', 'TopK', 'NonMaxSuppression'}:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[source_op.name]
                else: dispatching_table[operation.name] = fp32_platform

            # move activations to the platform same as their input.
            if operation.is_linear_activation:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[source_op.name]

        return dispatching_table


class PPLNNDispatcher(GraphDispatcher):
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable operations,
        Shape-Or-Index related operations and remaining operations, all sub classes of GraphDispatcher will
        give an implementation of function "dispatch" to send all operations to their proper platform.

    Conv only Dispatcher cuts graph in a conservative way, which means it takes as much as possible operations
        into fp32 platform.

    ATTENTION: platform attribute will greatly affect quantizer's quantization logic, and the execution result.
        If operation is sent to a quantable platform, then its inputs and outputs will be quantized if necessary.
        if operation is classified as shape-or-index related operation, then its execution will be taken with cpu.
        if operation is sent to a fp32 platform, then its inputs and outputs shall never be quantized.

    ATTENTION: this dispatcher will insert necessary DeviceSwitch operations
        between shape-or-index operations and others.
    """
    @staticmethod
    def dispatch(
        graph: BaseGraph, quant_types: Set[str],
        quant_platform: TargetPlatform,
        fp32_platform: TargetPlatform,
        SOI_platform: TargetPlatform, **kwargs
        ) -> Dict[str, TargetPlatform]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_platform - all quantable parts of graph will be dispatched to this platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched to this platform.
            fp32_platform  - there are some operations receiving results from both quant_platform and SOI_platform,
                they will be dispatched to fp32_platform.

        ATTENTION: Quantization follows this dispatching,
            and only the operations within quantable platform will be quantized in the future.

        ATTENTION: this dispatcher will insert necessary DeviceSwitch operations between
            shape-or-index operations and others.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this dispatcher.

            quant_types(Set[str]): all quantable types for given platforms.

            quant_platform (TargetPlatform): =
                platform object where quantable parts will goes to.

            fp32_platform (TargetPlatform):
                platform object where SOI parts will goes to.

            SOI_platform (TargetPlatform):
                platform object where remaining parts will goes to.

        Returns:
            Dict[str, TargetPlatform]: [description]
        """
        recivers, generators = SOI_receivers(graph), SOI_generators(graph)
        search_engine, SOI_operations = SearchableGraph(graph), set(recivers)

        quant_operations = search_engine.opset_matching(
            sp_expr = lambda x: x.type == 'Conv',
            rp_expr = lambda x, y: value_tracing_pattern(x, y) and y.type in quant_types,
            ep_expr = lambda x: x.type == 'Conv',
            direction = 'down')

        computing_extensions = search_engine.opset_matching(
            sp_expr = lambda x: x.is_computing_op,
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: x.type in {'Shape', 'TopK', 'NonMaxSuppression'} or x.is_boundary,
            direction = 'down')

        # we assume all 'Shape', 'NonMaxSuppression', 'ConstantOfShape', 'Topk' operations are SOI generators.
        shape_forward_matching = search_engine.opset_matching(
            sp_expr = lambda x: x in generators and x.type not in {'Constant'},
            rp_expr = value_tracing_pattern,
            ep_expr = lambda x: (x in recivers or
                                 x in quant_operations or
                                 x.is_boundary or
                                 x.is_computing_op),
            direction = 'down')

        # remove computing operations and quant operations from matching
        shape_forward_matching.filter(lambda x: x.is_computing_op or x in quant_operations)

        # update matchings, ready for further searching.
        SOI_operations.update(shape_forward_matching)

        while True:
            # there are some particular cases where a single matching can not handle.
            # to cover all shape-related operations, a reverse matching is required.
            shape_backward_matching = search_engine.opset_matching(
                sp_expr = lambda x: x in SOI_operations and x.type != 'Shape',
                rp_expr = reverse_tracing_pattern,
                ep_expr = lambda x: (x in SOI_operations or
                                     x in quant_operations or
                                     x.is_boundary or
                                     x.is_computing_op),
                direction = 'up')

            # remove computing operations and quant operations from matching
            shape_backward_matching.filter(lambda x: x.is_computing_op or x in quant_operations)

            if all([(op in SOI_operations) for op in shape_backward_matching]): break

            # update matchings
            SOI_operations.update(shape_backward_matching)

        # generate dispatching table.
        dispatching_table = {}
        for operation in graph.operations.values():
            if operation in SOI_operations and operation not in computing_extensions:
                dispatching_table[operation.name] = SOI_platform
            elif operation in quant_operations:
                dispatching_table[operation.name] = quant_platform
            else:
                dispatching_table[operation.name] = fp32_platform

        for operation in graph.operations.values():
            # move Topk, Shape, NonMaxSuppression to the platform same as their input.
            if operation.type in {'Shape', 'TopK', 'NonMaxSuppression'}:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[source_op.name]
                else: dispatching_table[operation.name] = fp32_platform

            # move activations to the platform same as their input.
            if operation.is_linear_activation:
                source_op = operation.inputs[0].source_op
                if source_op is not None:
                    dispatching_table[operation.name] = dispatching_table[source_op.name]

        return dispatching_table


class PointDispatcher(ConservativeDispatcher):
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.

    For the most part, all operations within graph can be partitioned into quantable operations,
        Shape-Or-Index related operations and remaining operations, all sub classes of GraphDispatcher will
        give an implementation of function "dispatch" to send all operations to their proper platform.

    Point Dispatch send all computing op to quantable platform, while other ops remain unchanged.

    ATTENTION: platform attribute will greatly affect quantizer's quantization logic, and the execution result.
        If operation is sent to a quantable platform, then its inputs and outputs will be quantized if necessary.
        if operation is classified as shape-or-index related operation, then its execution will be taken with cpu.
        if operation is sent to a fp32 platform, then its inputs and outputs shall never be quantized.

    ATTENTION: this dispatcher will insert necessary DeviceSwitch operations
        between shape-or-index operations and others.
    """
    @staticmethod
    def dispatch(
        graph: BaseGraph, quant_types: Set[str],
        quant_platform: TargetPlatform,
        fp32_platform: TargetPlatform,
        SOI_platform: TargetPlatform, **kwargs
        ) -> Dict[str, TargetPlatform]:
        """Graph Dispatcher splits a graph into parts, each part of graph will
        be sent to a specific platform for further execution and quantization.

        There are 3 default platform during dispatching:
            quant_platform - all quantable parts of graph will be dispatched to this platform
            SOI_platform   - Aka. Shape or Index related operations will be dispatched to this platform.
            fp32_platform  - there are some operations receiving results from both quant_platform and SOI_platform,
                they will be dispatched to fp32_platform.

        ATTENTION: Quantization follows this dispatching,
            and only the operations within quantable platform will be quantized in the future.

        ATTENTION: this dispatcher will insert necessary DeviceSwitch operations between
            shape-or-index operations and others.

        Args:
            graph (BaseGraph): graph object which going to be dispatched by this dispatcher.

            quant_types(Set[str]): all quantable types for given platforms.

            quant_platform (TargetPlatform): =
                platform object where quantable parts will goes to.

            fp32_platform (TargetPlatform):
                platform object where SOI parts will goes to.

            SOI_platform (TargetPlatform):
                platform object where remaining parts will goes to.

        Returns:
            Dict[str, TargetPlatform]: [description]
        """
        dispatch_table = ConservativeDispatcher.dispatch(
            graph=graph, quant_types=quant_types, quant_platform=quant_platform, 
            fp32_platform=fp32_platform, SOI_platform=SOI_platform, kwargs=kwargs)
        
        skip_ops = set()
        for op in graph.operations.values():
            if op in skip_ops: continue
            if op.type in quant_types and op.is_computing_op:
                dispatch_table[op.name] = quant_platform
            else:
                if dispatch_table[op.name] == quant_platform:
                    dispatch_table[op.name] = fp32_platform
        
        return dispatch_table
