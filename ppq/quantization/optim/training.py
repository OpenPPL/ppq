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
from ppq.quantization.algorithm.training import *
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from torch.cuda import empty_cache
from tqdm import tqdm

from .base import QuantizationOptimizationPass

logger = logging.getLogger('PPQ')


def has_bias(op: Operation):
    if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
        return op.meta_data.num_of_input == 3
    else: return False


def compute_loss(output_names: List[str],
                graph: BaseGraph,
                dataloader: Iterable,
                collate_fn: Callable,
                executor: TorchExecutor,
                loss_fn: Callable=torch_mean_square_error
) -> Dict[str, float]:
    losses = {name: 0.0 for name in output_names}
    for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Computing Original Loss'):
        if collate_fn is not None:
            data = collate_fn(data)
        dequantize_graph(graph)
        fp_outputs = executor.forward(data, output_names)
        restore_quantization_state(graph)
        quant_outputs = executor.forward(data, output_names)
        
        for name, fp_output, quant_output in zip(output_names, fp_outputs, quant_outputs):
            batch_loss = loss_fn(quant_output, fp_output)
            losses[name] += batch_loss.detach().item()

    for name in losses:
        losses[name] /= (idx + 1)     
    return losses

def dequantize_graph(graph: BaseGraph, exceptions: List[Operation]=[]) -> None:
    for op in graph.operations.values():
        if isinstance(op, QuantableOperation) and op not in exceptions:
            op.dequantize(expire_device=None)

def restore_quantization_state(graph: BaseGraph, exceptions: List[Operation]=[]) -> None:
    for op in graph.operations.values():
        if isinstance(op, QuantableOperation) and op not in exceptions:
            op.restore_quantize_state(expire_device=None)


def find_all_blocks(graph: BaseGraph,
                executing_order: List[Operation],
                block_limit: int=None
    ) -> List[TrainableBlock]:
        visited_ops = set()
        blocks = []
        block_builder = BlockBuilder(graph=graph, topo_order=executing_order)

        for op in graph.operations.values():
            if op not in visited_ops:
                if block_limit is None:
                    block = block_builder.build(op, OPTIM_ADVOPT_GRAPH_MAXSIZE)
                else:
                    block = block_builder.build(op, block_limit)
                for op in block.rps:
                    visited_ops.add(op)
                if len(graph.get_downstream_operations(block.ep)) > 1:
                    visited_ops.remove(block.ep)
                blocks.append(block)
        return blocks


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

       Note that you could control the maximum number of operations in a block by setting OPTIM_ADVOPT_GRAPH_MAXSIZE,
    and by default every block will be trained for 300 epochs, the optimization goal is

                Loss = LpNormLoss(y, y^) + lamda * rounding_loss(v)

    where y is the output of the current block running in fp mode, and y^ is the output of the current block running
    in quant mode, lamda is a hyperparameter adjusting scales of rounding loss, and v is the element-wise rounding
    parameter applied to weights of every computing op in the block
    """
    def __init__(self,
                name: str = 'block-wise reconstruction',
                interested_layers: List[str] = [],
                tune_act_scale: bool = True,
                epochs: int = 625,
                lr: float = 1e-3,
                lamda: float = 1.0,
                scale_multiplier: float = 2.0
    ) -> None:
        super().__init__(name = name)
        self.interested_layers = interested_layers
        self.tune_act_scale    = tune_act_scale
        self.epochs            = epochs
        self.lr                = lr
        self.lamda             = lamda
        self.scale_multuplier  = scale_multiplier


    def initiate_block_params(self,
                            block: TrainableBlock,
                            reg: AdaroundRegTerm,
                            device: Union[str, torch.device]
    ) -> Dict[TensorQuantizationConfig, BlockwiseReconstructionDelegator]:
        params = {}
        for op in block.rps:
            if isinstance(op, QuantableOperation):
                for (cfg, var) in op.config_with_variable:
                    if (not self.tune_act_scale and not var.is_parameter) or cfg in params:
                        continue
                    masters = []
                    scale_multiplier = 1.0
                    if cfg.state == QuantizationStates.PASSIVE:
                        # bias
                        if op.is_computing_op:
                            scale_multiplier = self.scale_multuplier
                            for cfg_ in op.config.input_quantization_config[:2]:
                                if cfg_.dominated_by in params:
                                    masters.append(params[cfg_.dominated_by])
                                else:
                                    scale_multiplier *= convert_any_to_torch_tensor(cfg_.scale,\
                                                device=device, dtype=torch.float32)

                            delegator = BlockwiseReconstructionDelegator(var, cfg, reg, scale_multiplier, device)
                            delegator.masters = masters
                            params[cfg] = delegator
                    
                    # for those vars who controls their own scales and offsets
                    elif cfg.state == QuantizationStates.ACTIVATED:
                        delegator = BlockwiseReconstructionDelegator(var, cfg, reg, scale_multiplier, device)
                        params[cfg] = delegator
                    elif cfg.state == QuantizationStates.SLAVE and cfg.dominated_by == cfg:
                        delegator = BlockwiseReconstructionDelegator(var, cfg, reg, scale_multiplier, device)
                        params[cfg] = delegator
                        for op_ in block.rps:
                            if isinstance(op_, QuantableOperation):
                                for (cfg_, var_) in op_.config_with_variable:
                                    if cfg_.state == QuantizationStates.SLAVE and cfg_.dominated_by == cfg:
                                        delegator_ = BlockwiseReconstructionDelegator(var_, cfg_, reg, scale_multiplier, device)
                                        delegator_.masters = [delegator]
                                        params[cfg_] = delegator_

        return params


    def tune_block_weight_scale(self,
                            block: TrainableBlock,
                            device: Union[str, torch.device],
                            epochs: int=900
    ) -> None:
        # before we tune weight roundings and activation scales, we optimize weight scale by
        # minimizing MSE(W, W^), 900 epochs would be enough in this non-overfit setting
        for op in block.rps:
            if op.is_computing_op:
                cfg = op.config.input_quantization_config[1]
                weight = op.inputs[1].value
                assert cfg.state == QuantizationStates.ACTIVATED, 'the config of weight param\
                should be ACTIVATED for tuning'
                delegator = StraightThroughEstimateDelegator(cfg, True, 1.0, device)
                params = delegator.collect_params()
                optimizer = torch.optim.Adam(params, lr=1e-3)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(epochs / 2), int(epochs * 2 / 3)])
                initial_loss, final_loss = None, None
                for _ in tqdm(range(epochs), total=epochs, desc=f'tune weight scale for {op.name}'):
                    loss = torch_mean_square_error(delegator(weight, cfg), weight, reduction='sum')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if initial_loss is None:
                        initial_loss = loss.detach().item()
                    final_loss = loss.detach().item()
                    scheduler.step()
                logger.info(f'Optimize {op.name} weight scale, initial loss {initial_loss}, optimized loss {final_loss}')
                if final_loss < initial_loss:
                    delegator.finalize()
                else:
                    logger.info('Loss increased, abandon trained values...')


    def optimize(self,
                processer: GraphCommandProcesser,
                dataloader: Iterable,
                executor: TorchExecutor,
                collate_fn: Callable,
                **kwargs
    ) -> None:
        graph = processer.graph
        all_blocks = find_all_blocks(graph, executor._executing_order)
        for block in all_blocks:
            # if interested_layers are not empty, we only optimize block which contains
            # desired ops specified in interested_layers
            if len(self.interested_layers) > 0 and all([op.name not in self.interested_layers for op in block.rps]):
                continue

            # tune weight scale first
            output_names = [var.name for var in block.ep.outputs]

            self.tune_block_weight_scale(block, executor._device)

            # compute original loss for loss checking
            original_loss = compute_loss(output_names, graph, \
                dataloader, collate_fn, executor, loss_fn=Lp_norm)
            original_loss = sum(list(original_loss.values()))

            # rounding loss intialization
            reg = AdaroundRegTerm(max_iter=len(dataloader) * self.epochs)
            all_params = self.initiate_block_params(block, reg, executor._device)
            scale_params, continue_vs = [], []
            
            # collect rounding parameters for rounding loss computing,
            # and all gradient-needed parameters for optimization
            for (cfg, delegator) in all_params.items():
                if delegator.rounding is not None:
                    continue_vs.append(delegator.rounding)
                if self.tune_act_scale and not delegator.is_parameter:
                    scale_params.extend(delegator.collect_params())
                executor.register_quantize_delegate(cfg, delegator)

            optimizers, schedulers = [torch.optim.Adam(continue_vs, lr=self.lr)], []
            if self.tune_act_scale and len(scale_params) > 0:
                # refer to implementation details in brecq paper
                scale_optimizer = torch.optim.Adam(scale_params, lr=4e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(scale_optimizer, \
                    T_max=len(dataloader) * self.epochs, eta_min=0.)
                optimizers.append(scale_optimizer)
                schedulers.append(scheduler)
                

            logger.info('Optimize block ' + f'{block.sp.name} -> ... -> {block.ep.name}')
            cur_iter = 0
            for epoch in range(self.epochs):
                epoch_rounding_loss = 0.0
                epoch_reconstruction_loss = {name: 0.0 for name in output_names}
                for idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{self.epochs}'):
                    if collate_fn is not None:
                        data = collate_fn(data)
                    dequantize_graph(graph)
                    fp_outputs = executor.forward(data, output_names)
                    restore_quantization_state(graph)
                    quant_outputs = executor.forward_with_gradient(data, output_names)
                    reconstruction_loss = 0.0
                    for (name, fp_output, quant_output) in zip(output_names, fp_outputs, quant_outputs):
                        loss = Lp_norm(fp_output, quant_output)
                        reconstruction_loss += loss
                        epoch_reconstruction_loss[name] += loss.detach().item()
                    rounding_loss = torch.tensor(0.0, dtype=torch.float32, device=executor._device)
                    for continue_v in continue_vs:
                        rounding_loss = rounding_loss + reg(continue_v, cur_iter)
                    rounding_loss = self.lamda * rounding_loss
                    total_loss = reconstruction_loss + rounding_loss
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    total_loss.backward()
                    for optimizer in optimizers:
                        optimizer.step()
                    epoch_rounding_loss += rounding_loss.detach().item()
                    cur_iter += 1

                for scheduler in schedulers:
                    scheduler.step()

                for name in epoch_reconstruction_loss:
                    logger.info(f'Epoch {epoch + 1} || {name} || reconstruction loss = {epoch_reconstruction_loss[name] / (idx + 1) :.5f}')
                avg_recon_loss = sum(list(epoch_reconstruction_loss.values())) / (idx + 1)
                avg_rounding_loss = epoch_rounding_loss / (idx + 1)
                logger.info(f'Epoch {epoch + 1} || reconstruction loss {avg_recon_loss :.5f} || rounding loss {avg_rounding_loss :.5f}')
                for _,continue_v in enumerate(continue_vs):
                    h_v = continue_v.detach()
                    logger.info("Rounding var {} Ceil: {:>5} Floor: {:>5} Total: {:>5} Ratio: {:>.3f}".format(
                        _ + 1, h_v[h_v + 1e-4 >= 1.0].numel(), h_v[h_v <= 1e-4].numel(), torch.numel(h_v),
                        (h_v[h_v + 1e-4 >= 1.0].numel() + h_v[h_v <= 1e-4].numel()) / torch.numel(h_v))
                    )

            logger.info(f'Original Reconstruction Loss {original_loss} || Optimized Reconstruction loss {avg_recon_loss}')
            if avg_recon_loss < original_loss:
                for (cfg, delegator) in all_params.items():
                    delegator.finalize()
            else:
                logger.info('Loss increased, abandon trained values...')

            for (cfg, delegator) in all_params.items():
                executor.remove_quantize_delegate(cfg)
            
            # process passive parameter of Pad or Clip for coherence
            for op in block.rps:
                if isinstance(op, QuantableOperation) and op.type == 'Clip':
                    input_config = op.config.input_quantization_config[0].dominated_by
                    for config in op.config.input_quantization_config[1: ]:
                        config.scale  = input_config.scale
                        config.offset = input_config.offset
                elif isinstance(op, QuantableOperation) and op.type == 'Pad':
                    if op.num_of_input != 3: continue
                    input_config = op.config.input_quantization_config[0].dominated_by
                    pad_config = op.config.input_quantization_config[-1]
                    pad_config.scale  = input_config.scale
                    pad_config.offset = input_config.offset


class LearningStepSizeOptimization(TrainingBasedPass):
    """Learned Step Size optimization, a training-based optimization pass which tunes weight, weight scale, weight offset(aym quantization)
    and activation scale, activation offset(asym quantization) of computing layers. You can perform a graphwise optimization which optimizes
    parameters all together by setting mode to global, or layerwise/blockwise optimization which optimizes in local range by setting mode to
    local. Similar to block-wise reconstruction pass, interested_layers contains computing layers which suffers from large precision loss 
    introduced by quantization, and if it's not specified, this pass will try to tune all condition-satisfied comptuing layers.

        In global mode, if the output_names is not specified, then every graph output will be used to compute final loss, note that in some
    cases gradient can't flow back from graph outputs all the way back to every computing ops, then you should specify output_names by choosing
    variables which guarantee valid gradient backward to every computing op specified in the interested_layers.
        
        When the graph structure becomes more complicated or the global mode gets overfitting effect, you might prefer the local mode,
    where scales and offsets are tuned blockwisely and you don't have to specify names of output variables.
        
        For more information about step learning algorithm, please refer to
            Esser, Steven K., et al. "Learned step size quantization." arXiv preprint arXiv:1902.08153 (2019).
    """
    def __init__(self,
                name: str = 'PPQ LSQ Optimization',
                interested_layers: List[str] = [],
                interested_layers_only: bool = False,
                output_names: List[str] = [],
                loss_weights: Dict[str, float] = {},
                epochs: int = 30,
                lr: float = 5e-5,
                scale_multiplier: float = 2,
                mode: str = 'global'
    ) -> None:
        super().__init__(name=name)
        self.interested_layers = interested_layers
        self.interested_layers_only = interested_layers_only
        self.output_names = output_names
        self.loss_weights = loss_weights
        self.epochs = epochs
        self.lr = lr
        self.scale_multiplier = scale_multiplier
        self.mode = mode

    def initiate_param(self,
                    block: TrainableBlock,
                    device: Union[str, torch.device]
    ) -> Dict[TensorQuantizationConfig, StraightThroughEstimateDelegator]:
        params = {}
        for op in block.rps:
            if isinstance(op, QuantableOperation):
                for (cfg, var) in op.config_with_variable:
                    scale_multiplier = 1.0
                    masters = []
                    if cfg.state == QuantizationStates.PASSIVE and op.is_computing_op:
                        scale_multiplier = self.scale_multiplier
                        for cfg_ in op.config.input_quantization_config[:2]:
                            if cfg_.dominated_by in params:
                                masters.append(params[cfg_.dominated_by])
                            else:
                                scale_multiplier *=  convert_any_to_torch_tensor(cfg_.scale,\
                                                device=device, dtype=torch.float32)
                        delegator = StraightThroughEstimateDelegator(cfg, var.is_parameter, scale_multiplier, device)
                        delegator.masters = masters
                        params[cfg] = delegator

                    elif cfg.state == QuantizationStates.ACTIVATED:
                        delegator = StraightThroughEstimateDelegator(cfg, var.is_parameter, scale_multiplier, device)
                        params[cfg] = delegator
                    elif cfg.state == QuantizationStates.SLAVE and cfg.dominated_by == cfg:
                        delegator = StraightThroughEstimateDelegator(cfg, var.is_parameter, scale_multiplier, device)
                        params[cfg] = delegator
                        for op_ in block.rps:
                            if isinstance(op_, QuantableOperation):
                                for (cfg_, var_) in op_.config_with_variable:
                                    if cfg_.state == QuantizationStates.SLAVE and cfg_.dominated_by == cfg:
                                        delegator_ = StraightThroughEstimateDelegator(cfg_, var_.is_parameter, scale_multiplier, device)
                                        delegator_.masters = [delegator]
                                        params[cfg_] = delegator_

        return params

    def enable_grad(self, block: TrainableBlock) -> None:
        for op in block.rps:
            if isinstance(op, QuantableOperation) and op.is_computing_op:
                for var in op.inputs[1:]:
                    var.value.requires_grad = True
    
    def disable_grad(self, block: TrainableBlock) -> None:
        for op in block.rps:
            if isinstance(op, QuantableOperation) and op.is_computing_op:
                for var in op.inputs[1:]:
                    var.value.requires_grad = True

    def recover(self, block: TrainableBlock) -> None:
        for op in block.rps:
            if isinstance(op, QuantableOperation):
                op.dequantize()
                op.store_parameter_value()
                op.restore_quantize_state()

    def LSQ_optimize_local(self,
                blocks: List[TrainableBlock],
                graph: BaseGraph,
                dataloader: Iterable,
                collate_fn: Callable,
                executor: TorchExecutor
    ) -> None:
        for blk in blocks:
            output_names = [var.name for var in blk.ep.outputs]
            original_loss = compute_loss(output_names, graph, dataloader, collate_fn, executor)
            params = self.initiate_param(blk, executor._device)
            block_params = []
            self.enable_grad(blk)
            for op in blk.rps:
                if isinstance(op, QuantableOperation) and op.is_computing_op:
                    for var in op.inputs[1:]:
                        block_params.append(var.value)
            for (cfg, delegator) in params.items():
                executor.register_quantize_delegate(cfg, delegator)
                block_params.extend(delegator.collect_params())
   
            optimizer = torch.optim.Adam([param for param in block_params if param.requires_grad], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(self.epochs / 2), int(self.epochs * 2 / 3)])

            logger.info(f'Optimizing block {blk.sp.name} --> ... --> {blk.ep.name}, total {len(blk.rps)} ops')
            for _ in range(self.epochs):
                epoch_loss = {name: 0.0 for name in output_names}
                for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {_ + 1}/{self.epochs}'):
                    if collate_fn is not None:
                        data = collate_fn(data)
                    dequantize_graph(graph)
                    fp_outputs = executor.forward(data, output_names)
                    restore_quantization_state(graph)
                    quant_outputs = executor.forward_with_gradient(data, output_names)
                    optimizer.zero_grad()
                    batch_loss = 0.0
                    for name, fp_output, quant_output in zip(output_names, fp_outputs, quant_outputs):
                        loss = torch_mean_square_error(fp_output, quant_output)
                        batch_loss += loss
                        epoch_loss[name] += loss.detach().item()
                    batch_loss.backward()
                    optimizer.step()
                scheduler.step()
                for name in epoch_loss:
                    logger.info(f'Epoch {_ + 1} || {name} || avg epoch loss = {epoch_loss[name] / (idx + 1) :.5f}')
                logger.info(f'Total epoch loss {sum(list(epoch_loss.values())) / (idx + 1) :.5f}')

            original_block_loss = sum(list(original_loss.values()))
            lsq_block_loss = sum(list(epoch_loss.values())) / (idx + 1)
            logger.info(f'Original loss {original_block_loss :.5f} || LSQ loss {lsq_block_loss :.5f}')

            for cfg, delegator in params.items():
                if lsq_block_loss < original_block_loss:
                    delegator.finalize()
                executor.remove_quantize_delegate(cfg)
            
            self.disable_grad(blk)
            if original_block_loss < lsq_block_loss:
                logger.info('Loss not improved, abandon trained values...')
                self.recover(blk)


    def LSQ_optimize_global(self,
                     blocks: List[TrainableBlock],
                     graph: BaseGraph,
                     dataloader: Iterable,
                     collate_fn: Callable,
                     executor: TorchExecutor
    ) -> None:
        output_names = [name for name in graph.outputs] if len(self.output_names) == 0 else self.output_names
        logger.info('The following variable will be used for loss computing and gradient backward')
        logger.info(', '.join(output_names))
        original_loss = compute_loss(output_names, graph, dataloader, collate_fn, executor)

        all_params, trainable_params = {}, []
        for blk in blocks:
            all_params.update(self.initiate_param(blk, executor._device))
            self.enable_grad(blk)
            for op in blk.rps:
                if isinstance(op, QuantableOperation) and op.is_computing_op:
                    trainable_params.extend([var.value for var in op.inputs[1:]])
        for cfg, delegator in all_params.items():
            executor.register_quantize_delegate(cfg, delegator)
            trainable_params.extend(delegator.collect_params())

        optimizer = torch.optim.Adam([param for param in trainable_params if param.requires_grad], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(self.epochs / 2), int(self.epochs * 2 / 3)])

        for _ in range(self.epochs):
            epoch_loss = {name: 0.0 for name in output_names}
            for idx,data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {_ + 1}/{self.epochs}'):
                if collate_fn is not None:
                    data = collate_fn(data)
                dequantize_graph(graph)
                fp_outputs = executor.forward(data, output_names)
                restore_quantization_state(graph)
                quant_outputs = executor.forward_with_gradient(data, output_names)
                optimizer.zero_grad()
                batch_loss = 0.0
                for name, fp_output, quant_output in zip(output_names, fp_outputs, quant_outputs):
                    loss = torch_mean_square_error(fp_output, quant_output)
                    batch_loss += self.loss_weights.get(name, 1.0) * loss
                    epoch_loss[name] += loss.detach().item()
                batch_loss.backward()
                optimizer.step()
            scheduler.step()
            weighted_loss = 0.0
            for name in epoch_loss:
                epoch_loss[name] /= idx + 1
                weighted_loss += self.loss_weights.get(name, 1.0) * epoch_loss[name]
                logger.info(f'Epoch {_ + 1} || {name} || avg epoch loss = {epoch_loss[name]}')
            logger.info(f'Epoch {_ + 1} || weighted loss = {weighted_loss}')
        
        weighted_original_loss = 0.0
        for name in original_loss:
            logger.info(f'{name} || Original Loss {original_loss[name] :.5f} || LSQ Loss {epoch_loss[name] :.5f}')
            weighted_original_loss += self.loss_weights.get(name, 1.0) * original_loss[name]
        logger.info(f'Original weighted loss {weighted_original_loss :.5f} || LSQ weighted loss {weighted_loss :.5f}')

        for cfg, delegator in all_params.items():
            if weighted_loss < weighted_original_loss:
                delegator.finalize()
            executor.remove_quantize_delegate(cfg)

        if weighted_original_loss < weighted_loss:
            logger.info('Loss not improved, abandon trained values...')

        for blk in blocks:
            self.disable_grad(blk)
            if weighted_original_loss < weighted_loss:
                self.recover(blk)


    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 collate_fn: Callable,
                 **kwargs) -> None:
        graph = processer.graph
        assert not(self.interested_layers_only and len(self.interested_layers) == 0), "you must specify interested_layers\
        when interested_layers_only is set to True"
        if self.interested_layers_only:
            # only tune interested_layers
            blocks = find_all_blocks(graph, executor._executing_order, block_limit=1)
        else:
            # tune whole subgraph which contains interested ops
            blocks = find_all_blocks(graph, executor._executing_order)

        if len(self.interested_layers) == 0:
            logger.info('No layers are given, all blocks will be tuned by default')
            final_blocks = blocks
        else:
            final_blocks = []
            for blk in blocks:
                if any([op.name in self.interested_layers for op in blk.rps]):
                    final_blocks.append(blk)

        if self.mode == 'global':
            logger.info('Begin globalwise LSQ Optimization...')
            self.LSQ_optimize_global(final_blocks, graph, dataloader, collate_fn, executor)
        else:
            logger.info('Begin localwise LSQ Optimization...')
            self.LSQ_optimize_local(final_blocks, graph, dataloader, collate_fn, executor)


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
            for rp in block.rps:
                if rp != block.sp and rp != block.ep:
                    visited.add(rp)
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
    
    def __init__(self, interested_output: List[str] = None, method: str = 'e-greedy', 
                 calib_act: bool = True, calib_weight: bool = True) -> None:
        self.interested_output = interested_output
        self.method            = method
        self.calib_act         = calib_act
        self.calib_weight      = calib_weight
        self.e                 = 0.1
        self.bandit_arms       = [0.7, 0.82, 0.9, 0.97, 0.99, 1, 1.01, 1.03, 1.1, 1.18, 1.3] 
        # for power-of-2 policy, use bandit like [0.5, 1, 2]
        super().__init__('RL Based Calibration Pass')

    def compute_loss(self, y_preds: List[torch.Tensor], y_reals: List[torch.Tensor]) -> float:
        return sum(
            torch_mean_square_error(y_pred=y_pred, y_real=y_real).item() 
            for y_pred, y_real in zip(y_preds, y_reals))

    def calib_block(self, quant_inputs: List[torch.Tensor], fp32_outputs: List[torch.Tensor],
        executor: TorchExecutor, block: TrainableBlock, dataloader: Iterable, collate_fn: Callable):
        # create trainable delegators for each parameter.
        delegators = []
        for operation in block.rps:
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state == QuantizationStates.ACTIVATED:
                        delegators.append(BanditDelegator(arms=self.bandit_arms, config=cfg))

        for delegator in delegators:
            assert isinstance(delegator, BanditDelegator)
        
        pass
            
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

    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: TorchExecutor, 
                 collate_fn: Callable, **kwargs) -> None:
        
        graph         = processer.graph
        block_builder = BlockBuilder(graph=graph, topo_order=executor._executing_order)

        # check if there is any baked value inside your graph
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if cfg.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
                        raise PermissionError('Can not apply advanced optimization pass when weight value is baked. '
                                              f'Variable {var.name} has a baked value.')

        # build all blocks, drop overlapped layers.
        blocks, visited = [], set()
        for op in graph.operations.values():
            if op in visited: continue
            block = block_builder.build(op, limit=OPTIM_ADVOPT_GRAPH_MAXSIZE)
            for rp in block.rps:
                if rp != block.sp and rp != block.ep:
                    visited.add(rp)
            blocks.append(block)
            
        graph         = processer.graph
        block_builder = BlockBuilder(graph=graph, topo_order=executor._executing_order)

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
            self.calib_block(
                quant_inputs=quant_inputs, fp32_outputs=fp32_outputs,
                executor=executor, block=block,
                dataloader=dataloader, collate_fn=collate_fn)

            # empty cache.
            fp32_outputs.clear()
            quant_inputs.clear()
            empty_cache()
