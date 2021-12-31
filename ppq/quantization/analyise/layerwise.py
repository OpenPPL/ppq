from typing import Callable, Dict, Iterable, List, Tuple

import torch
from ppq.core import (OperationMeta, QuantizationStates,
                      TensorQuantizationConfig, convert_any_to_torch_tensor)
from ppq.executor import BaseGraphExecutor, QuantOPRuntimeHook
from ppq.executor.torch import TorchExecutor
from ppq.IR import BaseGraph, QuantableOperation
from ppq.quantization.measure.cosine import torch_cosine_similarity

INTERESTED_OP_TYPE = {'Conv', 'Gemm', 'ConvTranspose', 'Relu', 'Sigmoid', 'Softmax'}


class AnalyseHook(QuantOPRuntimeHook):
    def __init__(self,
        executor: TorchExecutor,
        operation: QuantableOperation, 
        operation_meta: OperationMeta
    ) -> None:
        if operation.type not in INTERESTED_OP_TYPE:
            raise TypeError('AnalyseHook can only apply to ' + INTERESTED_OP_TYPE + 'operations')
        self._input_sims = []
        self._output_sims = []
        self._op_sims = []
        self._output_cache = None
        self._executor = executor
        super().__init__(operation, operation_meta=operation_meta)

    @ torch.no_grad()
    def pre_forward_hook(self, inputs: list, quant_inputs: list, 
        quant_configs: List[TensorQuantizationConfig]) -> list:
        # for all CONSINE_INTERESTED_OP_TYPE, input value should be the first value of inputs.
        # other value of inputs might be parameters
        tensor, quant_tensor, quant_config = inputs[0], quant_inputs[0], quant_configs[0]
        assert isinstance(tensor, torch.Tensor) and isinstance(quant_tensor, torch.Tensor) 
        if quant_config.state == QuantizationStates.ACTIVATED:
            self._input_sims.append(torch_cosine_similarity(
                y_real=tensor.flatten(start_dim=1), 
                y_pred=quant_tensor.flatten(start_dim=1), 
                reduction='mean').unsqueeze(0))

        # dequantize operation, calculate diff
        assert isinstance(self._hook_to, QuantableOperation)
        self._hook_to.dequantize()
        dequantized_inputs = [var.value for var in self._hook_to.inputs]
        # all INTERESTED_OP_TYPE operation have exact 1 output.
        [self._output_cache] = self._executor.operation_forward(operation=self._hook_to, inputs=dequantized_inputs)
        # restore quantization state
        self._hook_to.restore_quantize_state()
        return super().pre_forward_hook(inputs, quant_inputs, quant_configs)

    @ torch.no_grad()
    def post_forward_hook(self, outputs: list, quant_outputs: list, 
        quant_configs: List[TensorQuantizationConfig]) -> list:
        assert len(outputs) == 1, 'Oops seems input operation should not have more than 1 output.'
        tensor, quant_tensor, quant_config = outputs[0], quant_outputs[0], quant_configs[0]
        assert isinstance(tensor, torch.Tensor) and isinstance(quant_tensor, torch.Tensor) 
        if quant_config.state == QuantizationStates.ACTIVATED:
            self._output_sims.append(torch_cosine_similarity(
                y_real=tensor.flatten(start_dim=1), 
                y_pred=quant_tensor.flatten(start_dim=1), 
                reduction='mean').unsqueeze(0))
        # calculate op sim
        assert isinstance(self._output_cache, torch.Tensor), f'{type(self._output_cache)}'
        self._op_sims.append(torch_cosine_similarity(
            y_real=self._output_cache.flatten(start_dim=1), 
            y_pred=quant_tensor.flatten(start_dim=1), 
            reduction='mean').unsqueeze(0))
        self._output_cache = None
        return super().post_forward_hook(outputs, quant_outputs, quant_configs)

    @ torch.no_grad()
    def finialize(self) -> Tuple[float]:
        input_sim, output_sim = None, None
        try:
            if len(self._input_sims) > 0:
                input_sim = torch.mean(torch.cat(self._input_sims, dim=0))
                input_sim = input_sim.cpu().item()
            if len(self._output_sims) > 0:
                output_sim = torch.mean(torch.cat(self._output_sims, dim=0))
                output_sim = output_sim.cpu().item()
            op_sim = torch.mean(torch.cat(self._op_sims, dim=0))
            op_sim = op_sim.cpu().item()
        except RuntimeError as e:
            # RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated
            raise e
        self._input_sims.clear()
        self._output_sims.clear()
        return input_sim, output_sim, op_sim


def layerwise_min_max(graph: BaseGraph) -> dict:
    reports = {}
    for operation in graph.operations.values():
        if not len(operation.parameters) > 0: continue
        stats = []
        for param in operation.parameters:
            if param.value is None: continue
            tensor_param = convert_any_to_torch_tensor(param.value, device='cpu', accepet_none=False)
            _min, _max = torch.min(tensor_param).item(), torch.max(tensor_param).item()
            stats.append((param.name, _min, _max))
        reports[operation.name] = stats
    return reports


def tracing_cosine_similarity(
    graph: BaseGraph, 
    executor: BaseGraphExecutor, 
    dataloader: Iterable,
    collate_fn: Callable = None
    ) -> Dict[str, tuple]:
    # build hooks:
    hooks = {}
    for op_name, operation in graph.operations.items():
        if isinstance(operation, QuantableOperation) and operation.type in INTERESTED_OP_TYPE:
            hooks[op_name] = AnalyseHook(operation=operation, operation_meta=operation.meta_data, executor=executor)

    for batch in dataloader:
        if collate_fn is not None: batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)
    
    reports = {}
    for op_name in hooks:
        hook = hooks[op_name]
        assert isinstance(hook, AnalyseHook)
        reports[op_name] = hook.finialize()
    return reports
