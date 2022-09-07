from typing import Dict, Set

from ppq.core import TargetPlatform
from ppq.IR import BaseGraph
from .base import GraphDispatcher


class AllinDispatcher(GraphDispatcher):
    """Graph Dispatcher cuts a graph into parts, each part of graph will
    dispatch to a specific platform for further execution and quantization.
    ATTENTION: this dispatcher will enable all ops in quant_types to quant_platform.
    """

    @staticmethod
    def dispatch(
        graph: BaseGraph,
        quant_types: Set[str],
        quant_platform: TargetPlatform,
        fp32_platform: TargetPlatform,
        SOI_platform: TargetPlatform,
        **kwargs,
    ) -> Dict[str, TargetPlatform]:
        """
            We assume all ops in origin model can be quant.
            This is suitable for some npu platform.
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

        dispatching_table = {}
        for op in graph.operations.values():
            if op.type in quant_types:
                dispatching_table[op.name] = TargetPlatform.UNSPECIFIED
            else:
                dispatching_table[op.name] = TargetPlatform.FP32

        return dispatching_table