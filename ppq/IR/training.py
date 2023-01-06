from typing import Callable, List, Union

import torch

from .base.graph import BaseGraph
from .processer import GraphCommandProcessor


class TrainableGraph(GraphCommandProcessor):

    def __init__(self, graph_or_processor: Union[BaseGraph, Callable]) -> None:
        super().__init__(graph_or_processor)

    def parameters(self) -> List[torch.Tensor]:
        parameters = []
        for var in self.graph.variables.values():
            if var.is_parameter and var.dtype == torch.float:
                parameters.append(var.value)
        return parameters

    def zero_grad(self):
        for var in self.graph.variables.values():
            if var.is_parameter and var.dtype == torch.float:
                if var.value._grad is not None:
                    var.value._grad.zero_()

    def state_dict(self) -> dict:
        parameters = {}
        for var in self.graph.variables.values():
            if var.is_parameter and var.dtype == torch.float:
                parameters[var.name] = var.value
        return parameters

    def _acceptable_command_types(self): return None
    def process(self): return None
