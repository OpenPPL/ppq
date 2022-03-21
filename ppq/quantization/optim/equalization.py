from collections import defaultdict
from typing import Callable, Dict, Iterable, List
from ppq.IR.base.graph import BaseGraph

from ppq.core import empty_ppq_cache
from ppq.executor import BaseGraphExecutor
from ppq.IR import GraphCommandProcesser, Operation, QuantableOperation
from ppq.IR.search import SearchableGraph, TraversalCommand
from ppq.executor.torch import TorchExecutor
from ppq.quantization.algorithm.equalization import (EqualizationPair,
                                                     layerwise_equalization)
import torch
from tqdm import tqdm
from .base import QuantizationOptimizationPass

OPTIMIZATION_LAYERTYPE_CONFIG = {
    1: {'Relu', 'MaxPool', 'GlobalMaxPool', 'PRelu', 'AveragePool', 'GlobalAveragePool'},                     # level - 1 optimize
    2: {'Relu', 'MaxPool', 'GlobalMaxPool', 'Add', 'Sub', 'PRelu', 'AveragePool', 'GlobalAveragePool'},       # level - 2 optimize
}
EQUALIZATION_OPERATION_TYPE = {'Conv', 'Gemm', 'ConvTranspose'}

class LayerwiseEqualizationPass(QuantizationOptimizationPass):
    """
    PPQ Custimized Layerwise Equalization Pass
    
    This is a highly custimized implementation of layerwise equalization from:
    
    "Markus Nagel et al., Data-Free Quantization through Weight Equalization and Bias Correction" arXiv:1906.04721, 2019.
    
    Layerwise Equailization is inspired by a simple formula as below:
        Y = W * X + b = (s * W) * (X / s) + b
    
    Here s is called as a scale factor.
    By choosing s carefully, it is always possible to make max(W) = max(X).
    
    Notice that quantization error of W and X will be minimized when max(W) = max(X).
    """
    def __init__(
        self, iterations: int, weight_threshold: float = 0.5, 
        including_bias: bool = False, including_activation: bool = False,
        bias_mutiplier: float = 0.5, activation_mutiplier: float = 0.5,
        interested_layers: List[str] = None, optimize_level: int = 2, 
        verbose:bool = False) -> None:
        """
        PPQ Custimized Layerwise Equalization Pass

        This is a highly custimized implementation of layerwise equalization.
        With PPQ graph searching engine, this implementation can equalize mutilple layer at once,
            Even some layers are behind Add, Sub, Pooling.
        
        Not only weight, bias and activation are also taken into consideration with this implemenation.
        if including_bias and including_activation set as True, all weight, bias, activation will be pull
            equal with this function.

        Args:
            iterations (int): Equalization iterations.
            weight_threshold (float, optional): 
                A minimul value, all weight that belows that value will keep unchanged through this function.
                ATTENTION: this threshold will greatly affects your equalization performance.
                Defaults to 0.5. recommend to try 0.5, 2, 0
            
            including_bias (bool, optional): 
                whether to include bias into consideration. 
                ATTENTION: Some hardware use int16 accumulator or even worse
                    set this to be True if your hardware does not allow a 32-bit bias.
                Defaults to False.

            including_activation (bool, optional): 
                whether to include activation into consideration.
                Defaults to False.

            bias_mutiplier (float, optional): 
                a multipler to bias, if not necessary do not change this. 
                Defaults to 0.5.
            
            activation_mutiplier (float, optional):                 
                a multipler to activation, if not necessary do not change this. 
                Defaults to 0.5.
                
            interested_layers (List[str]):
                a layer collection contains all layers which need to be equalized.
                if None is given, suppose all layer need to be equalized.
            
            optimize_level (int, optional): [description]. Defaults to 2.
            verbose (bool, optional): [description]. Defaults to False.
        """
        self.optimize_level         = optimize_level
        self.iterations             = iterations
        self.weight_threshold       = weight_threshold
        
        self.including_bias         = including_bias
        self.bias_multipler         = bias_mutiplier
        
        self.including_activation   = including_activation
        self.activation_multipler   = activation_mutiplier
        
        self.interested_layers      = interested_layers
        self.verbose                = verbose
        super().__init__(name='PPQ Layerwise Equalization Pass')

    def find_equalization_pair(
        self, graph: BaseGraph, interested_operations: List[Operation]
    ) -> List[EqualizationPair]:

        # create a PPQ graph search engine.
        search_engine = SearchableGraph(graph)
        
        visited_ops, pairs = set(), []
        for operation in interested_operations:
            if operation in visited_ops: continue

            # skip operation that can not be equalized
            if operation.type not in EQUALIZATION_OPERATION_TYPE: continue

            # forward matching equalization pair.
            forward_matchings = search_engine(TraversalCommand(
                sp_expr=lambda x: x == operation,
                rp_expr=lambda x, y: y.type in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                ep_expr=lambda x: x.type not in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                direction='down'))

            downstream_ops = set()
            for matching in forward_matchings:
                downstream_ops.add(matching[-1])

            # backward matching equalization pair
            forward_matchings = search_engine(TraversalCommand(
                sp_expr=lambda x: x in downstream_ops,
                rp_expr=lambda x, y: y.type in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                ep_expr=lambda x: x.type not in OPTIMIZATION_LAYERTYPE_CONFIG[self.optimize_level],
                direction='up'))

            upstream_ops = set()
            for matching in forward_matchings:
                upstream_ops.add(matching[-1])

            # update pairs to visited.
            visited_ops.update(upstream_ops)

            # check if all operation inside this pair can be properly processed.
            valid_flag = True
            for operation in upstream_ops:
                if operation.type not in EQUALIZATION_OPERATION_TYPE:
                    valid_flag = False

            for operation in downstream_ops:
                if operation.type not in EQUALIZATION_OPERATION_TYPE:
                    valid_flag = False

            if not valid_flag: continue
            
            # consturct a new equalization pair.
            if len(upstream_ops) > 0 and len(downstream_ops) > 0:
                pairs.append(EqualizationPair(
                    all_upstream_layers=list(upstream_ops), 
                    all_downstream_layers=list(downstream_ops)))
        return pairs

    def collect_activations(self,
        executor: TorchExecutor, dataloader: Iterable,
        collate_fn: Callable, operations: List[Operation],
        steps: int = 16) -> Dict[Operation, torch.Tensor]:

        def aggragate(tensor: torch.Tensor):
            if tensor.ndim == 4: # Conv result: [n,c,h,w]
                num_of_channel = tensor.shape[1]
                tensor = tensor.permute(dims=[1, 0, 2, 3])
                tensor = tensor.reshape(shape=[num_of_channel, -1])
                tensor = torch.max(tensor.abs(), dim=-1, keepdim=False)[0]
            elif tensor.ndim == 2: # Gemm result: [n, c]
                num_of_channel = tensor.shape[1]
                tensor = tensor.permute(dims=[1, 0])
                tensor = tensor.reshape(shape=[num_of_channel, -1])
                tensor = torch.max(tensor.abs(), dim=-1, keepdim=False)[0]
            return tensor

        output_names = []
        for operation in operations:
            assert operation.num_of_output == 1, (
                f'Num of output of layer {operation.name} is supposed to be 1')
            output_names.append(operation.outputs[0].name)

        output_collector = defaultdict(list)
        for idx, batch in tqdm(enumerate(dataloader), 
                               desc='Equalization Data Collecting.', 
                               total=min(len(dataloader), steps)):
            data    = collate_fn(batch)
            outputs = executor.forward(data, output_names=output_names)
            for name, output in zip(output_names, outputs):
                output_collector[name].append(aggragate(output).unsqueeze(-1))
            if idx > steps: break

        result = {}
        for name, output in zip(output_names, outputs):
            result[name] = torch.max(torch.cat(output_collector[name], dim=-1)[0], dim=-1)
        return result

    @ empty_ppq_cache
    def optimize(
        self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, collate_fn: Callable, **kwargs
    ) -> None:
        interested_operations = []

        if self.interested_layers is None:

            for operation in processer.graph.operations.values():
                if operation.type in EQUALIZATION_OPERATION_TYPE:
                    interested_operations.append(operation)
        else:

            for name in interested_operations:
                if name in processer.graph.operations:
                    interested_operations.append(processer.graph.operations[name])

        pairs = self.find_equalization_pair(
            graph=processer.graph, interested_operations=interested_operations)

        '''
        activations = self.collect_activations(
            executor=executor, dataloader=dataloader, collate_fn=collate_fn, 
            operations=interested_operations)
        '''

        layerwise_equalization(
            equalization_pairs=pairs,
            weight_threshold=self.weight_threshold, 
            incluing_bias=self.including_bias,
            iteration=self.iterations, 
            verbose=self.verbose)

        # equalization progress directly changes fp32 value of weight,
        # store it for following procedure.
        for op in processer.graph.operations.values():
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()
