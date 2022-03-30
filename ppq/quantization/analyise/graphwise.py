
from typing import Callable, Dict, Iterator

import torch
from ppq.core import OperationMeta
from ppq.core.quant import QuantizationStates
from ppq.executor import RuntimeHook, TorchExecutor
from ppq.IR import BaseGraph, Operation
from ppq.IR.quantize import QuantableOperation
from ppq.utils.fetch import batch_random_fetch
from tqdm import tqdm

from .util import MeasurePrinter, MeasureRecorder


class OutputRecorder(RuntimeHook):
    def __init__(self, operation: Operation, 
        operation_meta: OperationMeta = None, fetchs: int = 4096) -> None:
        self.fetched     = None
        self.fetchs      = fetchs
        super().__init__(operation, operation_meta=operation_meta)

    def pre_forward_hook(self, inputs: list, **kwargs) -> list:
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        assert len(outputs) == 1, ('Multiple output tensor detected. '
            'Can not monitoring an operation with more than 1 output.')
        output_tensor = outputs[0]
        assert isinstance(output_tensor, torch.Tensor), (
            'Output of monitoring operation is not a torch.Tensor')
        self.fetched = batch_random_fetch(
            output_tensor, seed=10086, fetchs_per_batch=self.fetchs
        ).to('cpu')
        return super().post_forward_hook(outputs, **kwargs)

    def pop(self) -> torch.Tensor:
        fetched = self.fetched
        self.fetched = None
        return fetched


def graphwise_error_analyse(
    graph: BaseGraph, running_device: str,
    dataloader: Iterator, collate_fn: Callable = None, method: str = 'snr', 
    steps: int = 8, verbose: bool = True, fetchs: int = 4096) -> Dict[str, float]:
    """
    Measure the difference from a quantized graph to its dequantized graph.

    A dictionary contains output differences for all operation will be returned as a result.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}

    if verbose is set as True, this function will display error report at last.
    
    The key of the dictionary is an opeartion name while the value of corresponding key 
        is the difference between quantized output and float output of this operation.
    
    Result {'operation name 1': 0.933} means that quantized graph and fp32 graph have a difference
        (or similarity, based on your measurement) 0.933 at the output tensor of 'operation name 1'.
    
    ATTENTION: Output difference is measured at graph-level, it includes the difference accmulated from the
        very beginning operation along to the target operation.

    Args:
        graph (BaseGraph): 
            A fully quantized graph instance.
        
        running_device (str): 
            A device string used to initilize a graph executor for the graph execution.
                if a executor was given, this parameter will be skipped.

        dataloader (Iterator): 
            Test dataloader, this function will measure the output difference based on given data.

        collate_fn (Callable, optional):
            An data preprocessing function provided by user to convert data from dataloader towards
                executable format. If set as None, then no action will be taken during preprocessing.

        method (str, optional): 
            A string indicates a measurement to calculate the difference of quantized output and fp32 one.
                'cosine', 'snr', and 'mse' is supported in PPQ for now.
        
        steps (Int, optional)
            computation steps.

    Returns:
        A dictionary contains output differences for all operation will be returned from this function.
    
        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}
    """
    executor = TorchExecutor(graph=graph, device=running_device)

    # find all quantable operations.
    interested_op = [operation for operation in graph.operations.values()
                     if (isinstance(operation, QuantableOperation) and 
                         operation.config.output_quantization_config[0].state == QuantizationStates.ACTIVATED)]
    if len(interested_op) == 0: 
        print('Oops. you got nothing to analyse.')
        return

    # set up all hooks.
    recorders, hooks, caches = {}, {}, {}
    for operation in interested_op:
        if isinstance(operation, QuantableOperation):
            recorders[operation.name] = MeasureRecorder(measurement=method)
            hooks[operation.name] = OutputRecorder(
                operation=operation, operation_meta=operation.meta_data, fetchs=fetchs)
            caches[operation.name] = []

    # dequantize all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.dequantize()

    # run for each quantable opeartions:
    for idx, batch in tqdm(enumerate(dataloader), 
                           desc='Analysing Graphwise Quantization Error(Phrase 1):', 
                           total=(min(len(dataloader), steps))):
        if collate_fn is not None: batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)

        for operation in interested_op:
            hook = hooks[operation.name]
            caches[operation.name].append(hook.pop())
            
        if idx >= steps: break

    # restore all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.restore_quantize_state()

    # run for each quantable opeartions:
    for idx, batch in tqdm(enumerate(dataloader), 
                           desc='Analysing Graphwise Quantization Error(Phrase 2):',
                           total=(min(len(dataloader), steps))):
        if collate_fn is not None: batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)

        for operation in interested_op:
            recorder = recorders[operation.name]
            hook     = hooks[operation.name]
            cache    = caches[operation.name]
            recorder.update(y_real = cache[idx], y_pred = hook.pop())
        
        if idx >= steps: break

    results = {}
    for operation in interested_op:
        assert isinstance(operation, QuantableOperation)
        results[operation.name] = recorders[operation.name].measure
    
    if verbose: 
        method_str = 'MEASUREMENT'
        if method == 'snr': method_str = 'NOISE:SIGNAL POWER RATIO'
        if method == 'cosine': method_str = 'COSINE SIMILARITY'
        if method == 'mse': method_str = 'MSE LOSS(UNSCALED)'
        MeasurePrinter(results, order='large_to_small', measure=method_str).print()
    return results