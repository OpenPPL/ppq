
from typing import Callable, Iterable, List, Union

import torch
from ppq.core import QuantizationProperty, QuantizationStates
from ppq.executor.base import RuntimeHook
from ppq.executor.torch import TorchExecutor
from ppq.IR import (BaseGraph, GraphCommandProcessor, Operation,
                    QuantableGraph, QuantableOperation)
from tqdm import tqdm


class BatchnormHook(RuntimeHook):
    def __init__(self, operation: Operation) -> None:
        # infer norm shape by operation.
        if operation.type not in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:
            raise TypeError(f'Do not know how to create batchnorm after operation {operation.name}', 
                            f'Operation Type is unsupported. Use BatchnormHook(operation=op, norm_shape=shape) '
                            'to create this batchnorm instead.')
        assert len(operation.outputs) == 1, f'Can not create batchnorm layer after {operation.name}, too many outputs.'
        output_shape = operation.outputs[0].shape
        if output_shape is None:
            raise ValueError('Operation Output Shape has not been correctly intilized, '
                             'Please Trace Varaible Shape at first.')

        self.running_mean = None
        self.running_var  = None
        self.target_mean  = None
        self.target_var   = None
        super().__init__(operation)

    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        
        value = outputs[0]
        torch.batch_norm()
        return outputs

    def render(self):
        pass


class TrainableGraph(GraphCommandProcessor):

    def __init__(self, graph: Union[BaseGraph, Callable]) -> None:
        super().__init__(graph)

    def rebuild_batchnorm(
        self, dataloader: Iterable, interested_layers: List[str] = None,
        collate_fn: Callable = None, output_names: List[str] = None):
        """
        Create batchnorm layer after given interested layers.
        If interested layers = None, batchnorm layers will be created after each computing layer.
        
        This function will go through the entire dataloader for collecting batch stats.
        """
        for sample in tqdm(dataloader, desc='Collecting Batchnorm Stats.'):
            if collate_fn is not None: sample = collate_fn(sample)

    def parameters(self) -> List[torch.Tensor]:
        parameters = []
        for op in self.graph.operations.values():
            if op.is_computing_op or op.type in {'Add', 'Mul'}: # 独立的 Add Mul 算子也可以训练
                for var in op.inputs:
                    if var.is_parameter:
                        parameters.append(var.value)
        return parameters

    def tensor_quant_configs(self) -> List[torch.Tensor]:
        scales = []
        for op in self.graph.operations.values():
            if isinstance(op, QuantableOperation):
                for config, _ in op.config_with_variable:
                    policy_check = not config.policy.has_property(QuantizationProperty.POWER_OF_2)
                    state_check  = ((config.state == QuantizationStates.ACTIVATED) and (config.dominated_by == config))
                    value_check  = isinstance(config.scale, torch.Tensor)
                    if policy_check and state_check and value_check:
                        scales.append(config.scale)
        return scales

    def dequantize(self):
        quant_helper = QuantableGraph(self)
        quant_helper.dequantize_graph()
    
    def restore_quantize_state(self):
        quant_helper = QuantableGraph(self)
        quant_helper.restore_quantize_state()
