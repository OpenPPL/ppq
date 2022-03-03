import logging
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import torch
from numpy import ceil
from ppq.core import *
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.executor.base import GLOBAL_DISPATCHING_TABLE
from ppq.IR import (BaseGraph, GraphCommandProcesser, Operation,
                    QuantableOperation)
from ppq.IR.quantize import QuantableVariable
from ppq.quantization.algorithm.training import *
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from ppq.utils.round import ppq_tensor_round
from torch.cuda import empty_cache
from tqdm import tqdm

from .base import QuantizationOptimizationPass

logger = logging.getLogger('PPQ')


def has_bias(op: Operation):
    if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
        return op.meta_data.num_of_input == 3
    else: return False


class TrainingBasedPass(QuantizationOptimizationPass):
    """
    Training Based Pass is a basic class that provides necessary function for
        all training optimizition passes. Optimization will be more stable and
        accurate with functions provided by this pass. (Might be a little slower).

    This pass will collect result of interested outputs after optimization and
        check if the optimized result has a lower SNR. If so, the optimization will be
        accepted, layer weight will be updated, otherwise optimization will be rejected and
        takes no effects.

    Choose interested_outputs carefully, cause we compare loss only with those output variables.
        If interested_outputs is None, all graph output variables will be choosen.

    YOUR SHOULD NOTICE THAT SNR REFERS TO: POWER OF NOISE / POWER OF SIGNAL IN PPQ.

    Args:
        QuantizationOptimizationPass ([type]): [description]
    """
    def __init__(self, name: str = 'Default Quanzation Optim', 
                 interested_outputs: List[str] = None, verbose: bool = True) -> None:
        self._loss_fn = torch_snr_error
        self._interested_outputs = interested_outputs
        self._checkpoints = {}
        self._verbose = verbose
        self._quant_state_recorder = {}
        super().__init__(name=name)

    @ empty_ppq_cache
    def initialize_checkpoints(
        self, graph: BaseGraph, executor: BaseGraphExecutor, 
        dataloader: Iterable, collate_fn: Callable):
        """
        Establish a series of network checkpoints with your network.
            Checkpoint is a data structure that helps us compare quant results and fp32 results. 
        Args:
            graph (BaseGraph): [description]
            executor (BaseGraphExecutor): [description]
            dataloader (Iterable): [description]
            collate_fn (Callable): [description]

        Raises:
            PermissionError: [description]
        """
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                        raise PermissionError('Can not initialize checkpoints when weight value is baked. '
                                              f'Variable {var.name} has a baked value.')

        if self._interested_outputs is None or len(self._interested_outputs) == 0:
            self._interested_outputs = [name for name in graph.outputs]
        
        for name in self._interested_outputs:
            self._checkpoints[name] = FinetuneCheckPoint(variable=name)

        # dequantize graph, collect references
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation): 
                op.dequantize()

        for data in tqdm(dataloader, desc='Collecting Referecens'):
            if collate_fn is not None: data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=self._interested_outputs)
            for name, output in zip(self._interested_outputs, outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.push(tensor=output, is_reference=True)

        # restore quantization state:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation): 
                op.restore_quantize_state()
        
        # update state
        verbose, self._verbose = self._verbose, False
        self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn)
        self._verbose = verbose

    def check(self, executor: BaseGraphExecutor,
        dataloader: Iterable, collate_fn: Callable):
        """
        Check quantization error with a given dataloader with current checkpoints.
            Return whether quantization error is lower than before.

        Args:
            executor (BaseGraphExecutor): [description]
            dataloader (Iterable): [description]
            collate_fn (Callable): [description]

        Returns:
            [type]: [description]
        """
        
        # step - 1, collecting data
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=self._interested_outputs)
            for name, output in zip(self._interested_outputs, outputs):
                self._checkpoints[name].push(tensor=output, is_reference=False)

        # step - 2, calculating loss
        losses = []
        for name in self._interested_outputs:
            ckpt = self._checkpoints[name]
            assert isinstance(ckpt, FinetuneCheckPoint)
            qt_out, fp_out = ckpt.pop()
            qt_out = torch.cat([tensor for tensor in qt_out])
            fp_out = torch.cat([tensor for tensor in fp_out])
            losses.append(self._loss_fn(y_pred=qt_out, y_real=fp_out).item())
            ckpt.clear()

        # step - 3, comparing loss
        loss_now, loss_old = sum(losses), sum([ckpt.best_loss for ckpt in self._checkpoints.values()])
        loss_now, loss_old = loss_now / len(losses), loss_old / len(losses)
        if self._verbose: print(f'Power of Quant Noise: {loss_old * 100 :.4f}% -> {loss_now * 100:.4f}%.')

        # if there is a loss drop, update all losses.
        if loss_old > (loss_now * CHECKPOINT_TOLERANCE):
            for idx, name in enumerate(self._interested_outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.best_loss = losses[idx]
            return True
        return False

    def optimize(
        self, processer: GraphCommandProcesser,
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        raise NotImplementedError('Can not invoke this function. '
                                  'Please inherit this class and give an implmenetation to override this function.')

    def dequantize_immediately(self, operation: Operation):
        """
        Dequantize an operation inplace, use this function carefully.
            if parameter value has been changed during your optimization procedure,
            then it is not safe to dequantize an operation via this function,
            use operation.dequantize to load stored fp32 value instead.
        
        This function will change quantization state to dequantize an operation,
            Only quantization state will be changed by this function so that it is
            extremely fast.
        
        If your parameter value has already been baked, an exception will be thrown.
        Args:
            operation (Operation): [description]
        """
        if isinstance(operation, QuantableOperation):
            for cfg, _ in operation.config_with_variable:
                assert cfg.state not in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}, (
                    'Value has already been baked, can not dequantize it via this function.')

                if cfg not in self._quant_state_recorder:
                    self._quant_state_recorder[cfg] = cfg.state
                    cfg.state = QuantizationStates.DEQUANTIZED

    def quantize_immediately(self, operation: Operation):
        """
        Restore quantization state of an operation, use this function carefully.
            if parameter value has been changed during your optimization procedure,
            then it is not safe to restore state via this function,
            use operation.restore_quantize_state to load stored quant value instead.

        This function will change quantization state to quantize an operation,
            Only quantization state will be changed by this function so that it is
            extremely fast.

        If your parameter value has already been baked, an exception will be thrown.
        Args:
            operation (Operation): [description]
        """
        if isinstance(operation, QuantableOperation):
            for cfg, _ in operation.config_with_variable:
                if cfg in self._quant_state_recorder:
                    stored_state = self._quant_state_recorder[cfg]
                    cfg.state = stored_state
                    self._quant_state_recorder.pop(cfg)

    def dequantize_graph_immediately(self, graph: BaseGraph):
        """
        Dequantize entire graph inplace, use this function carefully.
            if parameter value has been changed during your optimization procedure,
            then it is not safe to dequantize graph via this function, as this function 
            only changes quantization state to dequantize entire graph.

        If your parameter value has already been baked, an exception will be thrown.
        Args:
            operation (Operation): [description]
        """
        for operation in graph.operations.values():
            self.dequantize_immediately(operation)

    def quantize_graph_immediately(self, graph: BaseGraph):
        """
        Restore quantization state of entire graph, use this function carefully.
            if parameter value has been changed during your optimization procedure,
            then it is not safe to restore state via this function.

        If your parameter value has already been baked, an exception will be thrown.
        Args:
            operation (Operation): [description]
        """
        for operation in graph.operations.values():
            self.quantize_immediately(operation)
         

class BiasCorrectionPass(TrainingBasedPass):
    def __init__(self, auto_check: bool=False, interested_output: List[str] = None, 
                 verbose: bool = True, max_steps:int = 8) -> None:
        """
        Quantization can introduce a biased error in the activations.
            Bias correction serves as a useful prosedure to eliminate those introduced bias error.

        let: Y = WX + b
             Quant(Y) = Qunat(W) Quant(X) + b
             
             bias_error = reduce_mean(Y - Quant(Y))
             
        Correct bias by: b = b + bias_error
        
        Args:
            quantize_function (BaseQuantFunction): [description]
            auto_check (bool, optional): [description]. Defaults to False.
        """
        super().__init__(name='PPQ Bias Correction Pass', 
                         interested_outputs=interested_output, verbose=verbose)
        self._auto_check = auto_check
        self._max_steps = max_steps

    @ empty_ppq_cache
    def optimize(
        self,
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        collate_fn: Callable,
        **kwargs
    ) -> None:
        def collect_bias(output: torch.Tensor, collector: list, op_type: str):
            if op_type in {'Conv', 'ConvTranspose'}: 
                collector.append(torch.mean(output, dim=(0, 2, 3)).unsqueeze(0))
            elif op_type in {'Gemm'}: 
                collector.append(torch.mean(output, dim=(0, )).unsqueeze(0))
            else: raise TypeError(f'Unsupported Operation type: {op_type}')

        assert isinstance(executor, TorchExecutor), (
            'PPQ Training-based optimization algorithm needs a TorchExecutor.')
    
        if self._auto_check:
            self.initialize_checkpoints(graph=processer.graph, executor=executor, 
                                        dataloader=dataloader, collate_fn=collate_fn)    
    
        for idx, operation in tqdm(enumerate(executor._executing_order), 
                                   desc='Bias Correction Procedure ...', 
                                   total=len(executor._executing_order)):
            assert isinstance(operation, Operation)
            if not has_bias(operation): continue
            
            bias, output_var = operation.inputs[-1].value, operation.outputs[0]
            qt_collector, fp_collector = [], []

            for idx, data in enumerate(dataloader):
                if collate_fn is not None: data = collate_fn(data)
                [output] = executor.forward(inputs=data, output_names=[output_var.name])
                collect_bias(output, qt_collector, op_type=operation.type)
                if idx >= self._max_steps: break
            self.dequantize_immediately(operation)
            
            for idx, data in enumerate(dataloader):
                if collate_fn is not None: data = collate_fn(data)
                [output] = executor.forward(inputs=data, output_names=[output_var.name])
                collect_bias(output, fp_collector, op_type=operation.type)
                if idx >= self._max_steps: break
            self.quantize_immediately(operation)

            bias_error = (torch.mean(torch.cat(fp_collector), dim=0) - torch.mean(torch.cat(qt_collector), dim=0))
            if self._auto_check:
                backup = bias.clone()
                operation.inputs[-1].value = bias + bias_error
                if not self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn):
                    operation.inputs[-1].value = backup
            else: operation.inputs[-1].value = bias + bias_error


class AdaRoundPass(QuantizationOptimizationPass):
    def __init__(self,
                 collecting_device: str = 'cpu',
                 epoch: int = 512,
                 batch_size: int = 32) -> None:
        super().__init__(name='PPQ AdaRound Pass')
        self._collecting_device = collecting_device
        self.epoch = epoch
        self.batch_size = batch_size

    @ empty_ppq_cache
    def optimize(
        self,
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        collate_fn: Callable,
        **kwargs
    ) -> None:
        assert isinstance(executor, TorchExecutor), ('PPQ Training-based optimization algorithm needs a TorchExecutor.')
        graph = processer.graph
        sorted_ops = graph.topological_sort()
        for idx, target_op in tqdm(enumerate(sorted_ops), desc='AdaRound...', total=len(graph.operations)):
            if not isinstance(target_op, QuantableOperation): continue
            if not target_op.type in {'Conv', 'ConvTranspose', 'Gemm'}: continue

            fp_outputs, quant_inputs = [], []
            interested_var = (target_op.inputs[0].name, target_op.outputs[0].name)

            for op in sorted_ops[: idx + 1]:
                if isinstance(op, QuantableOperation): op.dequantize()
            for data in tqdm(dataloader, desc='AdaRound Procedure 1', total=len(dataloader)):
                if collate_fn is not None: data = collate_fn(data)
                fp_input, fp_output = executor.forward(inputs=data, output_names=interested_var)
                fp_outputs.append(fp_output)
            fp_weight = target_op.parameters[0].value.clone()

            for op in sorted_ops[: idx + 1]:
                if isinstance(op, QuantableOperation): op.restore_quantize_state()
            for data in tqdm(dataloader, desc='AdaRound Procedure 2', total=len(dataloader)):
                if collate_fn is not None: data = collate_fn(data)
                quant_input, _ = executor.forward(inputs=data, output_names=interested_var)
                quant_inputs.append(quant_input)

            fp_outputs_concat = torch.cat(fp_outputs)
            quant_inputs_concat = torch.cat(quant_inputs)
            weight, bias = target_op.parameters[0].value, None
            if target_op.num_of_input == 3:
                bias = target_op.parameters[1].value
                bias = bias.clone()
            weight = weight.clone()
            params = [weight, bias] if bias is not None else [weight]
            for param in params: param.requires_grad = True

            print ('Adaround optimize {}'.format(target_op.name))
            weight_quantization_config = target_op.config.input_quantization_config[1].dominated_by
            weight_scale = weight_quantization_config.scale
            weight_offset = weight_quantization_config.offset

            max_iter = self.epoch * fp_outputs_concat.shape[0] / self.batch_size
            reg = AdaroundRegTerm(max_iter)

            # per-channel scale preprocess
            if weight_quantization_config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                view_shape = [
                    1 if axis != weight_quantization_config.channel_axis else -1
                    for axis in range(fp_weight.ndim)]
                weight_scale = weight_scale.view(view_shape)
                weight_offset = weight_offset.view(view_shape)

            # init continuous_v, make sure h(v) = round_diff
            round_diff = (fp_weight / weight_scale) - (fp_weight / weight_scale).floor()
            v_init = -torch.log((reg.zeta - reg.gamma) / (round_diff - reg.gamma) - 1)
            continuous_v = torch.nn.Parameter(v_init.to(executor._device), True)
            optimizer = torch.optim.Adam([continuous_v])

            cur_iter = 0
            data_len = quant_inputs_concat.shape[0]
            for ep_idx in range(self.epoch):
                batch_num = int(ceil(data_len / self.batch_size))
                # shuffle data
                index = np.arange(data_len)
                np.random.shuffle(index)
                for idx in range(batch_num):
                    st = idx * self.batch_size
                    ed = min(st + self.batch_size, data_len)

                    # soft AdaRound quant weight
                    params[0] = self.adaround_quant_weight(fp_weight, weight_scale, weight_offset, weight_quantization_config, continuous_v)
                    in_snap = [ quant_inputs_concat[index[st:ed,]] ]
                    [quant_output] = executor.operation_forward(target_op, inputs=in_snap + params)
                    fp32_output = fp_outputs_concat[index[st:ed],]

                    loss = Lp_norm(fp32_output, quant_output) + reg(continuous_v, cur_iter)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    cur_iter += 1

                if ep_idx % 100 == 0:
                    print("Epoch: {:<4} L2 Loss: {:>10.3f} Beta: {:>3.3f}".format(ep_idx, loss, reg.beta))
            h_v = AdaroundRegTerm().rectified_sigmoid(continuous_v)
            print("Loss: {:>5.3f} Ceil: {:>5} Floor: {:>5} Total: {:>5} Ratio: {:>.3f}".format(
                loss,
                h_v[h_v + 1e-4 >= 1.0].numel(), h_v[h_v <= 1e-4].numel(), torch.numel(h_v),
                (h_v[h_v + 1e-4 >= 1.0].numel() + h_v[h_v <= 1e-4].numel()) / torch.numel(h_v)))

            # update weight
            rounded_weight = self.adaround_quant_weight(fp_weight, weight_scale, weight_offset, weight_quantization_config, continuous_v, soft=False)
            target_op.parameters[0].value.copy_(rounded_weight)
            del fp_outputs_concat
            del quant_inputs_concat
            target_op.config.input_quantization_config[1].state = QuantizationStates.ACTIVATED
            if bias is not None:
                target_op.parameters[1].value.copy_(bias)
                target_op.config.input_quantization_config[-1].state = QuantizationStates.PASSIVE

    def adaround_quant_weight(self, weight, scale, offset, weight_quantization_config, round_var, soft=True):
        quant_max = weight_quantization_config.quant_max
        quant_min = weight_quantization_config.quant_min
        if soft:
            weight = (weight / scale).floor() + AdaroundRegTerm().rectified_sigmoid(round_var)
        else:
            weight = (weight / scale).floor() + (round_var >= 0).float()
        weight = torch.clamp(weight + offset, quant_min, quant_max)
        weight = (weight - offset) * scale
        return weight


class BlockwiseReconstructionPass(TrainingBasedPass):
    """Blockwise Reconstruction Pass, blockwisely perform adaround, only linear blocks are supported for now,
    if you specify interested_layers in the setting, then only block which containes any of operations specified
    in interested_layers will be optimized, otherwise all searched blocks will be optimized. A standard procedure
    is, first turn all training-based optimization passes off in your setting and run a plain quantization, then
    use error analysis tool(provided by ppq) to analysis snr error or cosine similarities of every layer, choose 
    names of those with significant snr or poor similarities as your interested_layers, then turn on this pass and
    do optimization.

       Note that you could control the maximum number of operations in a block by setting max_block_size, and by
    default every block will be trained for 300 epochs, the optimization goal is

                Loss = MSELoss(y, y^) + lamda * rounding_loss(v)

    where y is the output of the current block running in fp mode, and y^ is the output of the current block running
    in quant mode, lamda is a hyperparameter adjusting scales of rounding loss, and v is the element-wise rounding
    parameter applied to weights of every computing op in the block
    """
    def __init__(self,
                name: str = 'block-wise reconstruction',
                interested_layers: List[str] = [],
                tune_act_scale: bool = True,
                max_block_size: int = 4,
                epochs: int = 300,
                lr: float = 1e-3,
                lamda: float = 1.0,
                scale_multiplier: float = 2.0
    ) -> None:
        super().__init__(name = name)
        self.interested_layers = interested_layers
        self.tune_act_scale = tune_act_scale
        self.max_block_size = max_block_size
        self.epochs = epochs
        self.lr = lr
        self.lamda = lamda
        self.scale_multuplier = scale_multiplier

    def find_all_blocks(self, graph: BaseGraph) -> List[List[Operation]]:
        # find linear block which contains single-input-output ops, i.e.
        # receiving only one non-parameter variable as first input and
        # producing only one output

        visited_ops = set()
        blocks = []
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation) and op.is_computing_op\
                and op not in visited_ops:
                block = [op]
                next_ops = graph.get_downstream_operations(op)
                while len(next_ops) == 1 and isinstance(next_ops[0], QuantableOperation)\
                    and next_ops[0].num_of_input - next_ops[0].num_of_parameters == 1 and\
                    not next_ops[0].inputs[0].is_parameter:
                    block.extend(next_ops)
                    if len(block) >= self.max_block_size:
                        break
                    next_ops = graph.get_downstream_operations(next_ops[0])
                for op in block:
                    visited_ops.add(op)
                blocks.append(block)
        return blocks

    
    def enable_block_grad(self, block: List[Operation]) -> None:
        for op in block:
            for var in op.parameters:
                var.value.requires_grad = True

    def disable_block_grad(self, block: List[Operation]) -> None:
        for op in block:
            for var in op.parameters:
                var.value.requires_grad = False


    def initiate_block_params(self,
                            block: List[Operation],
                            reg: AdaroundRegTerm,
                            device: Union[str, torch.device]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        params = {}
        for idx, op in enumerate(block):
            # tune roundings for weight of computing ops
            if op.is_computing_op:
                weight, cfg = op.inputs[1].value, op.config.input_quantization_config[1]
                scale = convert_any_to_torch_tensor(cfg.scale, device=device, dtype=torch.float32)
                if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                    shape = [1 if axis != cfg.channel_axis else -1 for axis in range(weight.ndim)]
                    scale = scale.view(shape)
                round_diff = (weight / scale) - (weight / scale).floor()
                v_init = -torch.log((reg.zeta - reg.gamma) / (round_diff - reg.gamma) - 1)
                continuous_v = torch.nn.Parameter(v_init.to(device), True)
                params[op.inputs[1].name] = {'rounding' : continuous_v}
            
            # tune scale and offset for input
            cfg = op.config.input_quantization_config[0]
            if self.tune_act_scale and QuantizationStates.is_activated(cfg.state) and cfg.dominated_by == cfg:
                scale = convert_any_to_torch_tensor(cfg.scale, device=device, dtype=torch.float32)
                scale = torch.nn.Parameter(scale, requires_grad=True)
                offset = convert_any_to_torch_tensor(cfg.offset, device=device, dtype=torch.float32)
                bias = offset * scale.detach()
                if cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                    bias = torch.nn.Parameter(bias, requires_grad=True)
                params[op.inputs[0].name + f'_{op.name}'] = {'scale': scale, 'bias': bias}
            
            # tune scale and offset for output
            cfg = op.config.output_quantization_config[0]
            if self.tune_act_scale and QuantizationStates.is_activated(cfg.state) and cfg.dominated_by == cfg:
                scale = convert_any_to_torch_tensor(cfg.scale, device=device, dtype=torch.float32)
                scale = torch.nn.Parameter(scale, requires_grad=True)
                offset = convert_any_to_torch_tensor(cfg.offset, device=device, dtype=torch.float32)
                bias = offset * scale.detach()
                if cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                    bias = torch.nn.Parameter(bias, requires_grad=True)
                params[op.outputs[0].name + f'_{op.name}'] = {'scale': scale, 'bias': bias}

        return params


    def quantize_var(self,
                    tensor: torch.Tensor,
                    var: QuantableVariable,
                    op: QuantableOperation,
                    cfg: TensorQuantizationConfig,
                    all_params: Dict[str, Dict[str, torch.Tensor]],
                    reg: AdaroundRegTerm,
                    executor: BaseGraphExecutor
    ) -> torch.Tensor:

        # quantize weight
        if var.name in all_params:
            scale = cfg.scale
            offset = cfg.offset
            if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                shape = [1 if axis != cfg.channel_axis else -1 for axis in range(tensor.ndim)]
                scale = scale.view(shape)
                offset = offset.view(shape)
            tensor = (tensor / scale).floor() + reg.rectified_sigmoid(all_params[var.name]['rounding'])
            tensor = torch.clamp(tensor + offset, cfg.quant_min, cfg.quant_max)
            tensor = (tensor - offset) * scale

        # quantize activated activation
        elif self.tune_act_scale and var.name + f'_{op.name}' in all_params:
            scale = all_params[var.name + f'_{op.name}']['scale']
            bias = all_params[var.name + f'_{op.name}']['bias']
            if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                shape = [1 if axis != cfg.channel_axis else -1 for axis in range(tensor.ndim)]
                scale = scale.view(shape)
                bias = bias.view(shape)
            scale = scale.abs()
            tensor = (tensor + bias) / scale
            tensor_round = ppq_tensor_round(tensor, cfg.rounding)
            tensor_round = (tensor_round - tensor).detach() + tensor
            tensor_round = torch.clamp(tensor_round, cfg.quant_min, cfg.quant_max)
            tensor = tensor_round * scale - bias

        # for anything else which doesn't require gradient
        else:
            tensor = executor._quant_function(tensor, cfg)
        
        return tensor


    def execute_block(self,
                    inputs: List[torch.Tensor],
                    block: List[Operation],
                    all_params: Dict[str, Dict[str, torch.Tensor]],
                    reg: AdaroundRegTerm,
                    executor: BaseGraphExecutor
    ) -> List[torch.Tensor]:
        # execute linear block, every op in the block is assumed to have only one non-param input
        # and one output
        for op in block:
            assert(all([var.is_parameter for var in op.inputs[1:]]))
            inputs = inputs + [var.value for var in op.inputs[1:]]
            if isinstance(op, QuantableOperation):
                input_configs = [_ for _ in op.config.input_quantization_config]
                inputs = [self.quantize_var(tensor, var, op, cfg, all_params, reg, executor) for\
                    tensor, var, cfg in zip(inputs, op.inputs, input_configs)]
            outputs = executor.operation_forward(op, inputs, False, False)
            if isinstance(op, QuantableOperation):
                output_configs = [_ for _ in op.config.output_quantization_config]
                outputs = [self.quantize_var(tensor, var, op, cfg, all_params, reg, executor) for\
                    tensor, var, cfg in zip(outputs, op.outputs, output_configs)]
            inputs = outputs
        return outputs

    def dequantize_graph(self, graph: BaseGraph, exceptions: List[Operation]=[]) -> None:
       for op in graph.operations.values():
           if isinstance(op, QuantableOperation) and op not in exceptions:
               op.dequantize(expire_device=None)
   
    def restore_quantization_state(self, graph: BaseGraph, exceptions: List[Operation]=[]) -> None:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation) and op not in exceptions:
                op.restore_quantize_state(expire_device=None)
    

    def assign(self, 
            block: List[QuantableOperation],
            all_params: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        for op in block:
            if isinstance(op, QuantableOperation):
                for cfg, var in op.config_with_variable:
                    # update weight
                    if var.name in all_params:
                        rounding = all_params[var.name]['rounding']
                        weight = var.value
                        scale = cfg.scale
                        offset = cfg.offset
                        if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                            shape = [1 if axis != cfg.channel_axis else -1 for axis in range(weight.ndim)]
                            scale = scale.view(shape)
                            offset = offset.view(shape)
                        weight = (weight / scale).floor() +  AdaroundRegTerm().rectified_sigmoid(rounding)
                        weight = torch.clamp(weight, cfg.quant_min, cfg.quant_max)
                        weight = (weight - offset) * scale
                        var.value = weight
                    # activation scale, offset
                    elif var.name + f'_{op.name}' in all_params:
                        cfg.scale = all_params[var.name + f'_{op.name}']['scale'].data.abs()
                        cfg.offset = (all_params[var.name + \
                            f'_{op.name}']['bias'].data / cfg.scale).type(torch.int32)

    def compute_original_loss(self,
                            output_names: List[str],
                            graph: BaseGraph,
                            dataloader: Iterable,
                            collate_fn: Callable,
                            executor: BaseGraphExecutor
    ) -> float:
        loss = 0.0
        for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Computing Original Loss'):
            if collate_fn is not None:
                data = collate_fn(data)
            self.dequantize_graph(graph)
            fp_outputs = executor.forward(data, output_names)[0]
            self.restore_quantization_state(graph)
            quant_outputs = executor.forward(data, output_names)[0]
            batch_loss = torch_mean_square_error(quant_outputs, fp_outputs)
            loss += batch_loss.detach().item()
        return loss / (idx + 1)

    def tune_block_weight_scale(self,
                            block: List[QuantableOperation],
                            device: Union[str, torch.device],
                            epochs: int=30
    ) -> None:
        # before we tune weight roundings and activation scales, we optimize weight scale by
        # minimizing MSE(W, W^)
        for op in block:
            if op.is_computing_op:
                cfg = op.config.input_quantization_config[1]
                scale = convert_any_to_torch_tensor(cfg.scale, dtype=torch.float32, device=device)
                scale = torch.nn.Parameter(scale, requires_grad=True)
                params = [scale]
                offset = convert_any_to_torch_tensor(cfg.offset, device=device, dtype=torch.float32)
                bias = offset * scale.detach()
                if cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                    bias = torch.nn.Parameter(bias, requires_grad=True)
                    params.append(bias)
                weight = op.parameters[0].value
                grad_scale = 1.0 / (weight.numel() * cfg.quant_max)**0.5
                optimizer = torch.optim.Adam(params, lr=1e-3)
                initial_loss, final_loss = None, None
                for _ in tqdm(range(epochs), total=epochs, desc=f'tune weight scale for {op.name}'):
                    scale_ = (scale - scale * grad_scale).detach() + scale * grad_scale
                    bias_  = bias
                    if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                        shape = [1 if axis != cfg.channel_axis else -1 for axis in range(weight.ndim)]
                        scale_ = scale_.view(shape)
                        bias_ = bias_.view(shape)
                    scale_ = scale_.abs()
                    weight_quant = (weight + bias_) / scale_
                    weight_quant_round = ppq_tensor_round(weight_quant, cfg.rounding)
                    weight_quant_round = (weight_quant_round - weight_quant).detach() + weight_quant
                    weight_quant_round = torch.clamp(weight_quant_round, cfg.quant_min, cfg.quant_max)
                    weight_dequant = weight_quant_round * scale_ - bias_
                    loss = torch_mean_square_error(weight_dequant, weight, reduction='sum')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if initial_loss is None:
                        initial_loss = loss.detach().item()
                    final_loss = loss.detach().item()
                logger.info(f'Optimize {op.name} weight scale, initial loss {initial_loss}, optimized loss {final_loss}')
                if final_loss < initial_loss:
                    cfg.scale = scale.data.abs()
                    cfg.offset = (bias.data / cfg.scale).type(torch.int32)
                    if len(op.inputs) > 2:
                        [cfg_X, cfg_W, cfg_b] = op.config.input_quantization_config
                        cfg_b.scale = self.scale_multuplier * cfg_X.scale * cfg_W.scale
                else:
                    logger.info('Loss increased, abandon trained values...')


    def optimize(self,
                processer: GraphCommandProcesser,
                dataloader: Iterable,
                executor: BaseGraphExecutor,
                collate_fn: Callable,
                **kwargs
    ) -> None:
        graph = processer.graph
        all_blocks = self.find_all_blocks(graph)
        for block in all_blocks:
            # if interested_layers are not empty, we only optimize block which contains
            # desired ops specified in interested_layers
            if len(self.interested_layers) > 0 and all([op.name not in self.interested_layers for op in block]):
                continue
            # tune weight scale first
            self.tune_block_weight_scale(block, executor._device)
            # compute original loss for loss checking
            original_loss = self.compute_original_loss([block[-1].outputs[0].name], graph, dataloader, collate_fn, executor)
            # rounding loss intialization
            reg = AdaroundRegTerm(max_iter=len(dataloader) * self.epochs)
            all_params = self.initiate_block_params(block, reg, executor._device)
            self.enable_block_grad(block)
            param_need_grad, continue_vs = [], []
            
            # collect rounding parameters for rounding loss computing,
            # and all gradient-needed parameters for optimization
            for var_name in all_params:
                for item in all_params[var_name]:
                    if item == 'rounding':
                        continue_vs.append(all_params[var_name][item])
                    if all_params[var_name][item].requires_grad:
                        param_need_grad.append(all_params[var_name][item])
            
            # Adam optimizer with a two-step scheduler
            optimizer = torch.optim.Adam(param_need_grad, lr=self.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(self.epochs / 2), int(self.epochs * 2 / 3)])

            logger.info('Optimize block ' + '->'.join([op.name for op in block]))
            cur_iter = 0
            for epoch in range(self.epochs):
                epoch_mse_loss, epoch_rounding_loss = 0.0, 0.0
                for idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{self.epochs}'):
                    if collate_fn is not None:
                        data = collate_fn(data)
                    self.dequantize_graph(graph)
                    fp_outputs = executor.forward(data, [block[-1].outputs[0].name])
                    self.restore_quantization_state(graph)
                    quant_inputs = executor.forward(data, [block[0].inputs[0].name])
                    quant_outputs = self.execute_block(quant_inputs, block, all_params, reg, executor)
                    mse_loss = torch_mean_square_error(fp_outputs[0], quant_outputs[0])
                    rounding_loss = torch.tensor(0.0, dtype=torch.float32, device=executor._device)
                    for continue_v in continue_vs:
                        rounding_loss = rounding_loss + reg(continue_v, cur_iter, reduction='mean')
                    rounding_loss = self.lamda * rounding_loss
                    loss = mse_loss + rounding_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_mse_loss += mse_loss.detach().item()
                    epoch_rounding_loss += rounding_loss.detach().item()
                    cur_iter += 1
                avg_mse_loss = epoch_mse_loss / (idx + 1)
                avg_rounding_loss = epoch_rounding_loss / (idx + 1)
                logger.info(f'Epoch {epoch} || mse loss {avg_mse_loss} || rounding loss {avg_rounding_loss}')
                scheduler.step()

            self.disable_block_grad(block)
            logger.info(f'Original Loss {original_loss} || Optimized loss {avg_mse_loss}')
            if avg_mse_loss < original_loss:
                self.assign(block, all_params)
            else:
                logger.info('Loss increased, abandon trained values...')


class LearningStepSizeOptimization(TrainingBasedPass):
    """Learned Step Size optimization, a training-based optimization pass which tunes weight, weight scale, weight offset(aym quantization)
    and activation scale, activation offset(asym quantization) of computing layers. You can perform a graphwise optimization which optimizes
    parameters all together by setting mode to graphwise, or layerwise optimization which optimizes in local range(layer by layer) by setting
    mode to layerwise. Similar to block-wise reconstruction pass, interested_layers contains computing layers which suffers from large precision
    loss introduced by quantization, and if it's not specified, this pass will try to tune all condition-satisfied comptuing layers.

       In graphwise mode, if the output_names is not specified, then every graph output will be used to compute final loss, note that in some
    cases gradient can't flow back from graph outputs all the way back to every computing ops, then you should specify output_names by choosing
    variables which guarantee valid gradient backward to every computing op specified in the interested_layers.

    """
    def __init__(self,
                name: str = 'PPQ LSQ Optimization',
                interested_layers: List[str] = [],
                output_names: List[str] = [],
                epochs: int = 30,
                lr: float = 1e-4,
                scale_multiplier: float = 2,
                mode: str = 'graphwise'
    ) -> None:
        super().__init__(name=name)
        self.interested_layers = interested_layers
        self.output_names = output_names
        self.epochs = epochs
        self.lr = lr
        self.scale_multiplier = scale_multiplier
        self.mode = mode

    def initiate_param(self, ops: List[Operation], device: Union[str, torch.device]) -> Dict[str, Dict[str, torch.Tensor]]:
        params = {}
        for idx, (var, cfg) in enumerate(zip(ops[0].inputs[1:2] + ops[-1].outputs[0:1], \
            ops[0].config.input_quantization_config[1:2] + ops[-1].config.output_quantization_config[0:1])):
            scale = convert_any_to_torch_tensor(cfg.scale, device=device, dtype=torch.float32)
            scale = torch.nn.Parameter(scale, requires_grad=True)
            offset = convert_any_to_torch_tensor(cfg.offset, device=device, dtype=torch.float32)
            bias = offset * scale.detach()
            if cfg.policy.has_property(QuantizationProperty.ASYMMETRICAL):
                bias = torch.nn.Parameter(bias, requires_grad=True)
            params[var.name] = {'scale': scale, 'bias': bias}
        return params
    
    def enable_grad(self, ops: List[Operation]) -> None:
        for var in ops[0].inputs[1:]:
            var.value.requires_grad = True
    
    def disable_grad(self, ops: List[Operation]) -> None:
        for var in ops[0].inputs[1:]:
            var.value.requires_grad = False

    def dequantize_graph(self, graph: BaseGraph, exceptions: List[Operation]=[]) -> None:
       for op in graph.operations.values():
           if isinstance(op, QuantableOperation) and op not in exceptions:
               op.dequantize(expire_device=None)
   
    def restore_quantization_state(self, graph: BaseGraph, exceptions: List[Operation]=[]) -> None:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation) and op not in exceptions:
                op.restore_quantize_state(expire_device=None)

    def check(self, op: Operation, graph: BaseGraph) -> List[Operation]:
        # make sure we only tune activation scales for variables whose state
        # is activated
        if not op.is_computing_op:
            return []
        downstream_ops = graph.get_downstream_operations(op)
        if op.config.output_quantization_config[0].state == QuantizationStates.ACTIVATED:
            return [op]
        elif op.config.output_quantization_config[0].state == QuantizationStates.OVERLAPPED\
            and len(downstream_ops) == 1 and isinstance(downstream_ops[0], QuantableOperation):
            op_type, cfg = downstream_ops[0].type, downstream_ops[0].config.output_quantization_config[0]
            if (op_type in PPLCUDA_ACTIVATIONS or op_type in LINEAR_ACTIVATIONS) and \
                    op.config.output_quantization_config[0].dominated_by == cfg:
                    return [op, downstream_ops[0]]
        return []
    
    def prepare_quantize_input(self, op: Operation, params: Dict[str, Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        # perform quantization of weight and bias of computing ops using straight through estimation
        all_params = []
        weight_name, weight, cfg = op.inputs[1].name, op.inputs[1].value, op.config.input_quantization_config[1]
        grad_scale = 1.0 / (weight.numel() * cfg.quant_max)**0.5
        s_scale = (params[weight_name]['scale'] - params[weight_name]['scale'] * grad_scale).detach()\
             + params[weight_name]['scale'] * grad_scale
        bias = params[weight_name]['bias']
        if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            shape = [1 if axis != cfg.channel_axis else -1 for axis in range(weight.ndim)]
            s_scale = s_scale.view(shape)
            bias = bias.view(shape)

        s_scale = s_scale.abs()
        weight_quant = (weight + bias) / s_scale
        weight_quant_round = ppq_tensor_round(weight_quant, cfg.rounding)

        weight_quant_round = (weight_quant_round - weight_quant).detach() + weight_quant
        weight_quant_round = torch.clamp(weight_quant_round, cfg.quant_min, cfg.quant_max)
        weight_dequant = weight_quant_round * s_scale - bias
        all_params.append(weight_dequant)

        if len(op.inputs) > 2:
            Bias, cfg = op.inputs[2].value, op.config.input_quantization_config[2]
            input_scale = convert_any_to_torch_tensor(op.config.input_quantization_config[0].scale,\
                device=Bias.device, dtype=torch.float32)
            if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                input_scale = input_scale.view(shape)
            Bias_scale = (s_scale * input_scale * self.scale_multiplier).reshape(-1)
            Bias_quant = Bias / Bias_scale
            Bias_quant_round = ppq_tensor_round(Bias_quant, cfg.rounding)
            Bias_quant_round = (Bias_quant_round - Bias_quant).detach() + Bias_quant
            Bias_quant_round = torch.clamp(Bias_quant_round, cfg.quant_min, cfg.quant_max)
            Bias_dequant = Bias_quant_round * Bias_scale
            all_params.append(Bias_dequant)
        return all_params
    
    def assign(self, ops: List[Operation], params: List[torch.Tensor]) -> None:
        # assign trained scales and offsets to config
        for (var, cfg) in zip(ops[0].inputs[1:2] + ops[-1].outputs[0:1], \
            ops[0].config.input_quantization_config[1:2] + ops[-1].config.output_quantization_config[0:1]):
            cfg.scale = params[var.name]['scale'].data.abs()
            cfg.offset = (params[var.name]['bias'].data / cfg.scale).type(torch.int32)
        if len(ops[0].inputs) > 2:
            [cfg_X, cfg_W, cfg_b] = ops[0].config.input_quantization_config
            cfg_b.scale = self.scale_multiplier * cfg_X.scale * cfg_W.scale
    
    def compute_original_loss(self,
                            ops: List[Operation],
                            graph: BaseGraph,
                            dataloader: Iterable,
                            collate_fn: Callable,
                            executor: BaseGraphExecutor
    ) -> float:
        loss = 0.0
        for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Computing Original Loss'):
            if collate_fn is not None:
                data = collate_fn(data)
            self.dequantize_graph(graph)
            fp_outputs = executor.forward(data, [ops[-1].outputs[0].name])[0]
            self.restore_quantization_state(graph)
            quant_outputs = executor.forward(data, [ops[-1].outputs[0].name])[0]
            batch_loss = torch_mean_square_error(quant_outputs, fp_outputs)
            loss += batch_loss.detach().item()
        return loss / (idx + 1)            
    
    def recover(self, ops: List[Operation]) -> None:
        for op in ops:
            op.dequantize()
            op.store_parameter_value()
            op.restore_quantize_state()

    def quant_activation(self,
                        operation: Operation,
                        activations: List[torch.Tensor],
                        all_params: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        # perform activation quantization by straight through estimation
        act, cfg = activations[0], operation.config.output_quantization_config[0]
        scale, bias = all_params[operation.outputs[0].name]['scale'], all_params[operation.outputs[0].name]['bias']
        if cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            shape = [1 if axis != cfg.channel_axis else -1 for axis in range(act.ndim)]
            scale = scale.view(shape)
            bias = bias.view(shape)
        scale = scale.abs()
        act_quant = (act + bias) / scale
        act_quant_round = ppq_tensor_round(act_quant, cfg.rounding)
        act_quant_round = (act_quant_round - act_quant).detach() + act_quant
        act_quant_round = torch.clamp(act_quant_round, cfg.quant_min, cfg.quant_max)
        act_dequant = act_quant_round * scale - bias
        return [act_dequant]

    def manual_execute(self,
                    inputs: Union[dict, list, torch.Tensor],
                    all_ops: List[List[Operation]],
                    all_params: Dict[str, Dict[str, torch.Tensor]],
                    executor: BaseGraphExecutor,
                    output_names: List[str]=[]
    ) -> List[torch.Tensor]:
        # manually execute for graphwise optimization, for variables needing optimization, we
        # substitude the quantization function with straight through estimation functions for
        # gradient transport
        if isinstance(inputs, dict):
            for name, value in inputs.items():
                if name not in executor._graph.variables:
                    raise KeyError(f'Can not find variable {name} in your graph.')
                else:
                    var = executor._graph.variables[name]
                    var.value = value
        else:
            inputs = executor.prepare_input(inputs=inputs)
            for key, value in inputs.items():
                assert isinstance(value, torch.Tensor), \
                    f'TorchExecutor can only accept tensor as its input, while {type(value)} was given'
                executor._graph_input_dictionary[key].value = value

        last_idx = 0
        if len(output_names) == 0:
            output_names = [name for name in executor._graph.outputs]
        for name in output_names:
            if name not in executor._graph.variables:
                raise KeyError(f'You are requiring output value of variable {name}(is not a variable name), '
                    'however it is not a valid variable of current graph.')
            source_op = executor._graph.variables[name].source_op
            if source_op is not None:
                last_idx = max(last_idx, executor._executing_order.index(source_op) + 1)
        
        visited_op, result_collector = [], [None for _ in output_names]

        for name in output_names:
            if name in inputs: 
                result_collector[output_names.index(name)] = inputs[name]

        for operation in executor._executing_order[: last_idx]:
            try:
                platform_dispatching_table = GLOBAL_DISPATCHING_TABLE[operation.platform]
                operation_forward_func = platform_dispatching_table[operation.type]
                inputs = [var.value for var in operation.inputs]

                if isinstance(operation, QuantableOperation):
                    input_configs = [_ for _ in operation.config.input_quantization_config]
                    # replace by scale-trainable quant func
                    if any([operation in ops[0:1] for ops in all_ops]):
                        inputs = [executor._quant_function(inputs[0], input_configs[0])]
                        inputs = inputs + self.prepare_quantize_input(operation, all_params)
                    else:
                        inputs = [executor._quant_function(input, config) for input, config in zip(inputs, input_configs)]

                outputs = operation_forward_func(operation, inputs, executor._executing_contenxt)
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]

                if isinstance(operation, QuantableOperation):
                    output_configs = [_ for _ in operation.config.output_quantization_config]
                    # replace by scale-trainable quant func
                    if any([operation in ops[-1:] for ops in all_ops]):
                        outputs = self.quant_activation(operation, outputs, all_params)
                    else:
                        outputs = [executor._quant_function(output, config) for output, config in zip(outputs, output_configs)]

                for output_idx, output_var in enumerate(operation.outputs):
                    output_var       = operation.outputs[output_idx]
                    output_var.value = outputs[output_idx]
                    if output_var.name in output_names:
                        result_collector[output_names.index(output_var.name)] = outputs[output_idx]

            except Exception as _:
                raise RuntimeError(f'Error happens when dealing with operation {str(operation)}')

            visited_op.append(operation)
            for var in operation.inputs:
                if var.is_parameter: continue
                if all(op in visited_op for op in var.dest_ops):
                    var.value = None

        for var in executor._graph.variables.values():
            if not var.is_parameter:
                var.value = None

        return result_collector


    def LSQ_optimize_layer(self,
                     ops: List[Operation],
                     graph: BaseGraph,
                     dataloader: Iterable,
                     collate_fn: Callable,
                     executor: BaseGraphExecutor
    ) -> None:
        original_loss = self.compute_original_loss(ops, graph, dataloader, collate_fn, executor)
        params = self.initiate_param(ops, executor._device)
        self.enable_grad(ops)
        all_params = [var.value for var in ops[0].inputs[1:]]
        for var_name in params:
            all_params.append(params[var_name]['scale'])
            all_params.append(params[var_name]['bias'])

        optimizer = torch.optim.Adam([param for param in all_params if param.requires_grad], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(self.epochs / 2), int(self.epochs * 2 / 3)])
        
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {_ + 1}/{self.epochs}'):
                if collate_fn is not None:
                    data = collate_fn(data)
                self.dequantize_graph(graph)
                fp_outputs = executor.forward(data, [ops[-1].outputs[0].name])
                self.restore_quantization_state(graph)
                inputs = executor.forward(data, [ops[0].inputs[0].name])
                inputs = inputs + self.prepare_quantize_input(ops[0], params)
                for op in ops:
                    outputs = executor.operation_forward(op, inputs, False, False)
                    inputs = outputs
                act_dequant = self.quant_activation(ops[-1], outputs, params)
                optimizer.zero_grad()
                batch_loss = torch_mean_square_error(fp_outputs[0], act_dequant[0])
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().item()
            scheduler.step()
            avg_loss = epoch_loss / (idx + 1)
            logger.info(f'Epoch {_ + 1} || avg loss = {avg_loss}')

        self.disable_grad(ops)
        logger.info(f'Original Loss {original_loss} || LSQ Loss {avg_loss}')
        if avg_loss < original_loss:
            self.assign(ops, params)
        else:
            self.recover(ops)
            logger.info('Loss not improved, abandon trained values...')

    def LSQ_optimize_graph(self,
                     all_ops: List[List[Operation]],
                     graph: BaseGraph,
                     dataloader: Iterable,
                     collate_fn: Callable,
                     executor: BaseGraphExecutor
    ) -> None:
        all_params, trainable_params = {}, []
        for ops in all_ops:
            all_params.update(self.initiate_param(ops, executor._device))
            self.enable_grad(ops)
            trainable_params.extend([var.value for var in ops[0].inputs[1:]])
        for var_name in all_params:
            trainable_params.append(all_params[var_name]['scale'])
            trainable_params.append(all_params[var_name]['bias'])

        optimizer = torch.optim.Adam([param for param in trainable_params if param.requires_grad], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(self.epochs / 2), int(self.epochs * 2 / 3)])

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {_ + 1}/{self.epochs}'):
                if collate_fn is not None:
                    data = collate_fn(data)
                self.dequantize_graph(graph)
                if len(self.output_names) == 0:
                    fp_outputs = executor.forward(data)
                else:
                    fp_outputs = executor.forward(data, self.output_names)
                self.restore_quantization_state(graph)
                quant_outputs = self.manual_execute(data, all_ops, all_params, executor, self.output_names)
                optimizer.zero_grad()
                batch_loss = 0.0
                for fp_output, quant_output in zip(fp_outputs, quant_outputs):
                    batch_loss += torch_mean_square_error(fp_output, quant_output)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().item()
            scheduler.step()
            avg_loss = epoch_loss / (idx + 1)
            logger.info(f'Epoch {_ + 1} || avg loss = {avg_loss}')

        for ops in all_ops:
            self.disable_grad(ops)
            self.assign(ops, all_params)


    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 collate_fn: Callable,
                 **kwargs) -> None:
        graph = processer.graph
        sorted_ops = graph.topological_sort()

        if len(self.interested_layers) == 0:
            logger.info('Note that no layers are given, we will try to tune every computing layer')
            for op in sorted_ops:
                if isinstance(op, QuantableOperation) and op.is_computing_op:
                    self.interested_layers.append(op.name)
 
        if self.mode == 'layerwise':
            logger.info('Perform layerwise LSQ optimization...')
            for idx, op in enumerate(sorted_ops):
                if op.name in self.interested_layers:
                    assert op.is_computing_op, "only computing ops can be selected as interested\
                    layers and tuned"
                    ops = self.check(op, graph)
                    if not ops:
                        logger.warning(f'Operation {op.name} is not supported for LSQ finetuning for now')
                    else:
                        logger.info(f"Optimize Op {'--'.join([op.name for op in ops])}")
                        self.LSQ_optimize_layer(ops, graph, dataloader, collate_fn, executor)
        else:
            logger.info('Perform graphwise LSQ optimization...')
            all_ops = []
            for idx, op in enumerate(sorted_ops):
                if op.name in self.interested_layers:
                    assert op.is_computing_op, "only computing ops can be selected as interested\
                    layers and tuned"
                    ops = self.check(op, graph)
                    if not ops:
                        logger.warning(f'Operation {op.name} is not supported for LSQ finetuning for now')
                    else:
                        all_ops.append(ops)
            self.LSQ_optimize_graph(all_ops, graph, dataloader, collate_fn, executor)


class AdvancedQuantOptimization(TrainingBasedPass):
    """
    PPQ Advanced Quantization Optimization

    This optimization pass minimize the quantization errors of each subgraph separately
        by optimizing its parameters over the calibration set.

    Where:
        qout = quant( quant(W + W_offset) * quant(X) + quant(bias + bias_offset) )
    
        fout = W * B + bias
    
        error = Mean((qout - fout)^2)
    
    This training procedure trys to solve best W_offest and bias_offset to minimize error
        Based on your setting and network size, the training procedure will takes 5~120 minutes.
    
    This function will treat your network as series of subgraphs, you should notice that
        ONLY THE OUTPUT VALUE OF A SUBGRAPH IS OPTIMIZED IN THIS PASS, 
        ACTIVATIONS THAT INSIDE YOUR SUBGRAPH MIGHT BE GREATLY CHANGED!
        DO NOT ATTEMPT TO COMPARE THOSE INTERNAL VALUE WITH ITS FP32 VERSION.
    
    We use graph search engine to build subgraph from your network with pattern below,
        see function build_block_from_start for detail information

    Args:
        TrainingBasedPass ([type]): [description]
    """
    def __init__(self, collecting_device: str, limit: float = 3.0, steps: int = 5000,
                 lr: float = 3e-4, interested_outputs: List[str] = None, 
                 interested_layers: List[str] = None,
                 verbose: bool = True, check: bool = True) -> None:

        super().__init__(
            name='PPQ Advanced Optimization Procedure', 
            interested_outputs=interested_outputs, verbose=verbose)

        self.lr                = lr
        self.collecting_device = collecting_device
        self.check_flag        = check
        self.limit             = limit
        self.interested_layers = interested_layers
        self.target_step       = steps
        
        self._bidx = 0
        self._num_of_blocks = 0
        
        if isinstance(self.interested_layers, list) and len(self.interested_layers) == 0:
            self.interested_layers = None

    def collect_training_data(
        self, output_name: str,
        dataloader: Iterable,
        executor: BaseGraphExecutor, 
        collate_fn: Callable) -> List[List[torch.Tensor]]:

        output_collector = []
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            [output] = executor.forward(data, output_names=[output_name])
            output_collector.append(output.to(self.collecting_device))
        return output_collector

    @ empty_ppq_cache
    def finetune(
        self, quant_inputs: List[torch.Tensor], fp32_outputs: List[torch.Tensor],
        executor: TorchExecutor, block: TrainableBlock, 
        dataloader: Iterable, collate_fn:Callable) -> None:

        # initialize training environment.
        losses      = []
        cur_iter    = 0
        delegators  = []
        device      = executor._executing_contenxt.executing_device
        output_var  = block.ep.outputs[0]
        input_var   = block.sp.inputs[0]
        dataset = RandomMemDataset(data=[[qt, fp] for qt, fp in zip(quant_inputs, fp32_outputs)])

        # create trainable delegators for each parameter.
        trainable_vars = []
        for operation in block.rps:
            if operation.is_computing_op and isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if not var.is_parameter: continue
                    trainable_vars.append((var, cfg))
    
        delegators = [RQTDelegator(config=cfg, limit=self.limit, binding=var) for var, cfg in trainable_vars]     
        optimizer = torch.optim.Adam(params=[d.binding.value for d in delegators], lr=self.lr)
        # register all quantization delegators
        for d in delegators: executor.register_quantize_delegate(d.config, d)
        
        with tqdm(total=self.target_step) as t:
            while cur_iter < self.target_step:
                qt_input, fp_output = dataset.pop()

                qt_input, fp_output = qt_input.to(device), fp_output.to(device)
                qt_output = executor.partial_graph_forward(
                    operations=block.rps, feed_dict={input_var.name: qt_input}, 
                    output_names=[output_var.name])[0]

                # compute loss
                optimizer.zero_grad()
                round_loss = torch.sum(torch.cat([PPQRoundingLoss(d.binding.value, d.config) for d in delegators]))
                quant_loss = torch_mean_square_error(qt_output, fp_output)
                total_loss = quant_loss + round_loss * OPTIM_ADVOPT_RLOSS_MULTIPLIER
                total_loss.backward()
                optimizer.step()

                cur_iter += 1
                
                if cur_iter % 50 == 0:
                    t.set_description(desc=f'Block [{self._bidx + 1}/{self._num_of_blocks}]')
                    t.set_postfix(loss = total_loss.item())
                    t.update(50)

            # clear loss state
            losses.clear()

        # DEBUG INFO
        '''
        for raw, binding, alpha in zip([d.raw for d in delegates], 
                                       [d.binding for d in delegates], 
                                       [d.alpha for d in delegates]):
            print(alpha.shape)
            print(' ------ GARD ------')
            print(alpha._grad.flatten()[:10])
            print(' ------ VALUE ------')
            print(alpha.flatten()[:10])
            print(' ------ RAW ------')
            print(raw.flatten()[:10])
            print(' ------ BINDING ------')
            print(binding.value.flatten()[:10])
        '''
        
        # finalize all delegates
        for delegator in delegators:
            assert isinstance(delegator, RQTDelegator)
            delegator.finalize()
            executor.remove_quantize_delegate(delegator.config)

        # Check
        if self.check_flag:
            if not self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn):
                for delegator in delegators:
                    assert isinstance(delegator, RQTDelegator)
                    delegator.withdraw()

        # detach weight
        for delegator in delegators:
            assert isinstance(delegator, RQTDelegator)
            delegator.binding.value = delegator.binding.value.detach()

    def optimize(
        self, processer: GraphCommandProcesser, dataloader: Iterable,
        executor: TorchExecutor, collate_fn: Callable, **kwargs) -> None:
        
        if self._interested_outputs is None:
            self._interested_outputs = [name for name in processer.graph.outputs]

        if self.collecting_device == 'executor': 
            self.collecting_device = executor._device

        graph         = processer.graph
        block_builder = BlockBuilder(graph=graph, topo_order=executor._executing_order)

        # check if there is any baked value inside your graph
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                        raise PermissionError('Can not apply advanced optimization pass when weight value is baked. '
                                              f'Variable {var.name} has a baked value.')

        # find all operations that need to be finetuned.
        interested_ops = []
        for target_op in graph.topological_sort():
            if isinstance(target_op, QuantableOperation) and target_op.is_computing_op:
                if self.interested_layers is None: interested_ops.append(target_op)
                elif self.interested_layers is not None and target_op.name in self.interested_layers:
                    interested_ops.append(target_op)
        
        # build all blocks, drop overlapped layers.
        blocks, visited = [], set()
        for op in interested_ops:
            if op in visited: continue
            block = block_builder.build(op, limit=OPTIM_ADVOPT_GRAPH_MAXSIZE)
            for rp in block.rps: visited.add(rp)
            blocks.append(block)

        # set up checkpoints
        if self.check_flag:
            self.initialize_checkpoints(
                graph=graph, executor=executor, 
                dataloader=dataloader, collate_fn=collate_fn)

        for bidx, block in enumerate(blocks):
            self._bidx, self._num_of_blocks = bidx, len(blocks)
            assert isinstance(block, TrainableBlock)

            end_op       = block.ep
            block_input  = block.sp.inputs[0]
            block_output = end_op.outputs[0]
            
            # dequantize prefix operations and block operations
            for op in graph.operations.values():
                if isinstance(op, QuantableOperation): 
                    op.dequantize()
                    # can not use dequantize_immediately cause weight has been changed.
                    # self.dequantize_immediately(op)

            fp32_outputs = self.collect_training_data(
                output_name=block_output.name, dataloader=dataloader, 
                executor=executor, collate_fn=collate_fn)

            # quantize prefix operations and block operations
            for op in graph.operations.values():
                if isinstance(op, QuantableOperation): 
                    op.restore_quantize_state()

            quant_inputs = self.collect_training_data(
                output_name= block_input.name, dataloader=dataloader, 
                executor=executor, collate_fn=collate_fn)

            # start training, solve the best parameters
            self.finetune(
                quant_inputs=quant_inputs, fp32_outputs=fp32_outputs,
                executor=executor, block=block,
                dataloader=dataloader, collate_fn=collate_fn)

            # empty cache.
            fp32_outputs.clear()
            quant_inputs.clear()
            empty_cache()


class LearningToCalibPass(TrainingBasedPass):
    """
        This is an Experimental Pass, do not invoke.
        
        PPQ Leraning Based Calibration Pass
        For int8 quantization, you need to calibrate or estimate the value range, 
            i.e, (min, max) of all floating-point tensors in the model. 
        
        Choose value range carefully is really importance procedure during quantization.
            Usually we use methods like MSE, Percentile, KL to solve a good value range
            from prospective view, while this pass offers you another possibility.
        
        This pass will make all your quantization range as trainable, and learn to quantize
            your network with sampling methods.

        ATTENTION: YOU SHALL USE THIS FUNCTION AFTER ACTIVATIONS HAVE BEEN CORRECTLY CALIBRATED
            SINCE THIS FUNCTION NEEDS A SCALE AND OFFSET AS INITIALIZED VALUE.

        ATTENTION: ONLY CONFIGURATION WITH STATE "ACTIVED" WILL BE TUNED VIA THIS FUNCTION.
    """
    
    def __init__(self, 
                 interested_output: List[str] = None, method: str = 'TS', 
                 calib_act: bool = True, calib_weight: bool = True) -> None:
        self.interested_output = interested_output
        self.method = method
        self.calib_act = calib_act
        self.calib_weight = calib_weight
        self.bandit_arms = [0.7, 0.82, 0.9, 0.97, 1, 1.03, 1.1, 1.18, 1.3] # for power-of-2 policy, use bandit like [0.5, 1, 2]
        super().__init__('Sampling Based Calibration Pass')

    def compute_loss(self, y_preds: List[torch.Tensor], y_reals: List[torch.Tensor]) -> float:
        return sum(
            torch_mean_square_error(y_pred=y_pred, y_real=y_real).item() 
            for y_pred, y_real in zip(y_preds, y_reals))

    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: TorchExecutor, 
                 collate_fn: Callable, **kwargs) -> None:
        graph = processer.graph
        interested_outputs = [name for name in graph.outputs]
        target_iter, samples_per_iter = 5, 2048

        # find all trainable configs
        training_configs = []
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if (cfg.state == QuantizationStates.ACTIVATED and 
                        cfg.policy.has_property(QuantizationProperty.PER_TENSOR)):
                        training_configs.append(cfg)

        for iter in range(target_iter):
            bandits = [BanditDelegator(arms=self.bandit_arms, config=config) for config in training_configs]
            for bandit in bandits: executor.register_quantize_delegate(bandit.config, bandit)

            for data in tqdm(dataloader):
                data = collate_fn(data)
                
                self.dequantize_graph_immediately(graph)
                fp_refs = executor.forward(data)
                
                for bandit in bandits: bandit.active = False
                self.quantize_graph_immediately(graph)
                qt_refs = executor.forward(data)
                ref_loss = self.compute_loss(y_preds=qt_refs, y_reals=fp_refs)
                
                # 那有赌狗一直输 ...
                for bandit in bandits: bandit.active = True
                for i in range(10):
                    results = executor.forward(data)
                    loss = self.compute_loss(y_preds=results, y_reals=fp_refs)
                    for bandit in bandits: bandit.mark(ref_loss - loss)
    
            for bandit in bandits: bandit.finalize()
            for bandit in bandits: executor.remove_quantize_delegate(bandit.config)
