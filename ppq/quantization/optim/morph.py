from typing import Iterable, List
from ppq.core.defs import ppq_warning

from ppq.executor import BaseGraphExecutor
from ppq.IR import GraphCommandProcesser
from ppq.IR.search import SearchableGraph

from .base import QuantizationOptimizationPass


class NXPResizeModeChangePass(QuantizationOptimizationPass):
    """
    This optimization pass overwrite resize mode to 'nearest' for all resize operations.
    """
    def __init__(self) -> None:
        super().__init__(name='NXP Resize Operation Transformation')
    
    def optimize(self, processer: GraphCommandProcesser, dataloader: Iterable, 
        executor: BaseGraphExecutor, **kwargs) -> None:
        for op in processer.graph.operations.values():
            if op.type == 'Resize':
                op.attributes['mode'] = 'nearest'
                op.attributes['coordinate_transformation_mode'] = 'half_pixel'


class ChannelSplitPass(QuantizationOptimizationPass):
    """
    This Pass split Conv or Gemm layer into some sub layers.
    
        Split Sturcture like Conv - Conv into:

               - - Conv - -
        Conv - +          + - Concat
               - - Conv - - 
    
    This Pass will significantly increase the accuracy of quantized network(per layer quantized),
        however at the sarifice of executing effiecncy.
    
    Use this pass as the last resort when other optimization pass doesn't work.
    
    ATTENTION: We don't check your input layer, some time split your layer is not safe.
    You should choose your interested_layers carefully.
    
    Inspired by following code:
    
    import torch

    A = torch.rand(size=[1024, 8])
    B = torch.rand(size=[8, 1024])
    C = torch.matmul(A, B)

    reorder = [7, 1, 2, 3, 4, 5, 6, 0]
    A = A.permute(dims=[1, 0])
    A = A[reorder]
    A = A.permute(dims=[1, 0])

    B = B[reorder]
    B_1 = B[:, 0: 512]
    B_2 = B[:, 512: ]
    C_2 = torch.cat(
        [torch.matmul(A, B_1), torch.matmul(A, B_2)], dim=-1)
    
    where C_2 == C, which means we can split tensor B into B_1 and B_2,
        to construct a better quantization of tensor B
        (if B_1 and B_2 are delicately solved.)

    Args:
        QuantizationOptimizationPass ([type]): [description]
    """
    def __init__(self, interested_layers: List[str]) -> None:
        self.interested_layers = interested_layers
        super().__init__(name='Channel Split Pass')

    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor, 
                 **kwargs) -> None:

        graph = processer.graph
        search_engine = SearchableGraph(processer)
        
        for name in self.interested_layers:
            if name not in graph.operations: continue
            matching = search_engine.path_matching(
                sp_expr=lambda x: x.name == name,
                rp_expr=lambda x, y: True, # We don't check whether it is a resonable path.
                ep_expr=lambda x: x.is_computing_op,
                direction='up'
            )

            if len(matching) != 1:
                ppq_warning(f'Can not find a proper upstream operation for {name}.')
                continue

            counterpart, me = matching[-1], matching[0]
