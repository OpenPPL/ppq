from functools import partial
from typing import Callable, Dict, Iterator, List, Set

import torch
from ppq.core import OperationMeta, TensorQuantizationConfig
from ppq.executor import TorchExecutor, RuntimeHook
from ppq.IR import BaseGraph, Operation
from ppq.IR.quantize import QuantableOperation
from ppq.executor.base import BaseGraphExecutor
from ppq.quantization.measure import (torch_cosine_similarity,
                                      torch_mean_square_error, torch_snr_error)
from tqdm import tqdm


class OutputRecorder(RuntimeHook):
    def __init__(self, operation: Operation, 
        operation_meta: OperationMeta = None) -> None:
        self.last_output = None
        super().__init__(operation, operation_meta=operation_meta)
    
    def pre_forward_hook(self, inputs: list, **kwargs) -> list:
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        assert len(outputs) == 1, ('Multiple output tensor detected. '
            'Can not monitoring an operation with more than 1 output.')
        output_tensor = outputs[0]
        assert isinstance(output_tensor, torch.Tensor), (
            'Output of monitoring operation is not a torch.Tensor')
        self.last_output = output_tensor.to('cpu')
        return super().post_forward_hook(outputs, **kwargs)

    def pop_output(self) -> torch.Tensor:
        last_output = self.last_output
        self.last_output = None
        return last_output


def graph_similarity_analyse(
    quant_graph: BaseGraph, running_device: str, interested_op_type: Set[str],
    dataloader: Iterator,  quant_op_only: bool = True, interested_op_names: List[str] = None,
    collate_fn: Callable = None, measurement: str = 'cosine', max_steps: int = None,
    executor: BaseGraphExecutor = None) -> Dict[str, float]:
    """
    Calcuate the difference between a quantized graph and underlying float graph.

    A dictionary contains output differences for all operation will be returned as a result.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}
    
    The key of the dictionary is an opeartion name while the value of corresponding key 
        is the difference between quantized output and float output of this operation.
    
    Result {'operation name 1': 0.933} means that quantized graph and fp32 graph have a difference
        (or similarity, based on your measurement) 0.933 at the output tensor of 'operation name 1'.
    
    ATTENTION: Output difference is measured at graph-level, it includes the difference accmulated from the
        very beginning operation along to the target operation.

    Args:
        quant_graph (BaseGraph): 
            A fully quantized graph instance.
        
        running_device (str): 
            A device string used to initilize a graph executor for the graph execution.
                if a executor was given, this parameter will be skipped.

        interested_op_type (Set[str]): 
            A list or set of string, contains operation types that you want to monitor with.
            ['Conv', 'Gemm'] for example.

        dataloader (Iterator): 
            Test dataloader, this function will measure the output difference based on given data.

        quant_op_only (Bool): 
            whether to monitor quantizable operation only

        interested_op_names (List[str]): 
            A list or set of string, contains operation names that you want to monitor with.
            operations listed in interested_op_names will always in surveillance, 
                ignoring the setting of interested_op_type and quant_op_only.

        collate_fn (Callable, optional):
            An data preprocessing function provided by user to convert data from dataloader towards
                executable format. If set as None, then no action will be taken during preprocessing.

        measurement (str, optional): 
            A string indicates a measurement to calculate the difference of quantized output and fp32 one.
                'Cosine', 'snr', and 'mse' is supported in PPQ for now.
        
        executor (BaseExecutor, optional):
            An executor instance for graph execution. If no executor was given, this function will create one.
            
        max_steps (Int, optional)
            max computation steps.

    Returns:
        A dictionary contains output differences for all operation will be returned from this function.
    
        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}
    """
    if str(measurement).lower() == 'cosine':
        measure_fn = partial(torch_cosine_similarity, reduction='mean')
    elif str(measurement).lower() == 'mse':
        measure_fn = partial(torch_mean_square_error, reduction='mean')
    elif str(measurement).lower() == 'snr':
        measure_fn = partial(torch_snr_error, reduction='mean')
    else:
        raise ValueError('Unsupported measurement detected. '
            f'PPQ only support mse, snr and consine now, while {measurement} was given.')

    interested_op_list, measure_recorder, samples_counter = [], {}, {}
    for op in quant_graph.operations.values():
        if op.type in interested_op_type:
            if quant_op_only and not isinstance(op, QuantableOperation): continue
            interested_op_list.append(op.name)
            measure_recorder[op.name] = 0
            samples_counter[op.name] = 0

    if interested_op_names is not None:
        for name in set(interested_op_names):
            if name in quant_graph.operations and name not in interested_op_list:
                interested_op_list.append(name)
                measure_recorder[name] = 0
                samples_counter[name] = 0            

    # initialize all hooks
    dequant_hooks, quant_hooks = {}, {}
    for op_name in interested_op_list:
        quant_op = quant_graph.operations[op_name]
        dequant_hooks[op_name] = OutputRecorder(quant_op)
        quant_hooks[op_name]   = OutputRecorder(quant_op)

    # execute graph
    if executor is None:
        executor = TorchExecutor(graph=quant_graph, device=running_device)
        
    if max_steps is None: max_steps = len(dataloader)
    for step, batch in tqdm(enumerate(dataloader), 
                            desc='Measure Similarity...', 
                            total=min(len(dataloader), max_steps)):
        if step >= max_steps: break
        
        if collate_fn is not None: batch = collate_fn(batch)

        for op in quant_graph.operations.values():
            if isinstance(op, QuantableOperation): op.dequantize()
        executor.forward(batch, hooks=dequant_hooks)

        for op in quant_graph.operations.values():
            if isinstance(op, QuantableOperation): op.restore_quantize_state()
        executor.forward(batch, hooks=quant_hooks)

        for op_name in interested_op_list:
            dequant_output = dequant_hooks[op_name].pop_output()
            quant_output   = quant_hooks[op_name].pop_output()
            assert (isinstance(dequant_output, torch.Tensor) and 
                isinstance(quant_output, torch.Tensor))

            # for some case output value are flattened, unsqueeze it.
            while dequant_output.ndim <= 1: dequant_output = dequant_output.unsqueeze(0)
            while quant_output.ndim <= 1: quant_output = dequant_output.unsqueeze(0)
            
            num_of_samples = dequant_output.shape[0]
            local_measurement = measure_fn(
                dequant_output.flatten(start_dim=1), 
                quant_output.flatten(start_dim=1))

            recorded_samples = samples_counter[op_name]
            recorded_measurement = measure_recorder[op_name]

            recorded_measurement = (recorded_measurement * recorded_samples + 
                local_measurement.item() * num_of_samples)
            recorded_measurement = recorded_measurement / (num_of_samples + recorded_samples)
            recorded_samples = recorded_samples + num_of_samples

            measure_recorder[op_name] = recorded_measurement
            samples_counter[op_name] = recorded_samples

    return measure_recorder
