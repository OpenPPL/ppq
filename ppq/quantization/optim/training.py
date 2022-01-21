import collections
from typing import Callable, Iterable, List

import numpy as np
import torch
from numpy import ceil
from ppq.core import (CHECKPOINT_TOLERANCE, OPTIM_ADVOPT_GRAPH_MAXSIZE,
                      OPTIM_ADVOPT_INITIAL_THRES, OPTIM_ADVOPT_PASSIVE_BOOST,
                      OPTIM_ADVOPT_PATIENT, OPTIM_ADVOPT_STEP_PER_EPOCH,
                      OPTIM_ADVOPT_THRESHOLD_STEP, QuantizationProperty,
                      QuantizationStates, empty_ppq_cache)
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import (BaseGraph, GraphCommandProcesser, Operation,
                    QuantableOperation)
from ppq.quantization.algorithm.training import (AdaroundRegTerm,
                                                 FinetuneCheckPoint, Lp_norm,
                                                 RandomMemDataset,
                                                 TrainableDelegate)
from ppq.quantization.analyise.util import MeasurePrinter
from ppq.quantization.measure import torch_mean_square_error, torch_snr_error
from torch.cuda import empty_cache
from tqdm import tqdm

from .base import QuantizationOptimizationPass


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

        for data in dataloader:
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
        if self._verbose: print(f'SNR after optimization: {loss_old * 100 :.4f}% -> {loss_now * 100:.4f}%.')

        # if there is a loss drop, update all losses.
        if loss_old > (loss_now * CHECKPOINT_TOLERANCE):
            for idx, name in enumerate(self._interested_outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.best_loss = losses[idx]
            return True

        if self._verbose: print(f'Not a perfect loss drop, skip this optimization.')
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


class LearningStepSizeOptimization(TrainingBasedPass):
    def __init__(self, name: str = 'PPQ LSQ Optimization') -> None:
        super().__init__(name=name)
    
    def optimize(self, processer: GraphCommandProcesser, 
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 **kwargs) -> None:
        
        return super().optimize(processer, dataloader, executor, **kwargs)


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
    
    This function will treat your network as a series of subgraph, you should notice that
        ONLY THE OUTPUT VALUE OF A SUBGRAPH IS OPTIMIZED IN THIS PASS, 
        ACTIVATIONS THAT INSIDE YOUR SUBGRAPH MIGHT BE GREATLY CHANGED!
        DO NOT ATTEMPT TO COMPARE THOSE QUANTIZED ACTIVATION WITH ITS FP32 VERSION.
    
    We use graph search engine to build subgraph from your network with pattern below:

    while len(graph.get_downstream_operations(start_op)) == 1:
        end_op = graph.get_downstream_operations(start_op)[0]
        if len(graph.get_upstream_operations(end_op)) == 1:
            path.append(end_op)
            start_op = end_op
        else: break

    Args:
        TrainingBasedPass ([type]): [description]
    """
    def __init__(self, collecting_device: str, limit: float = 3.0, lr: float = 1e-3,
                 interested_outputs: List[str] = None, interested_layers: List[str] = None,
                 verbose: bool = True, check: bool = True) -> None:

        super().__init__(
            name='PPQ Advanced Optimization Procedure(Blockwise)', 
            interested_outputs=interested_outputs, verbose=verbose)

        self.lr                = lr
        self.collecting_device = collecting_device
        self.check_flag        = check
        self.offset_limit      = limit
        self.interested_layers = interested_layers
        self.t_step            = OPTIM_ADVOPT_THRESHOLD_STEP
        self.steps_per_epoch   = OPTIM_ADVOPT_STEP_PER_EPOCH
        self.patient           = OPTIM_ADVOPT_PATIENT
        self.passive_boost     = OPTIM_ADVOPT_PASSIVE_BOOST
        self.max_iter          = 10000
        
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
        executor: TorchExecutor, block: List[Operation], 
        dataloader: Iterable, collate_fn:Callable) -> None:

        # initialize training environment.
        losses     = []
        last_loss  = 1e9
        threshold  = OPTIM_ADVOPT_INITIAL_THRES
        trys_count = 0
        cur_iter   = 0
        delegates  = []
        device     = executor._executing_contenxt.executing_device
        loss_recorder = {}
        output_var = block[-1].outputs[0]
        input_var  = block[0].inputs[0]
        
        dataset = RandomMemDataset(data=[[qt, fp] for qt, fp in zip(quant_inputs, fp32_outputs)])

        # create trainable delegates for each parameter.
        for operation in block:
            if operation.is_computing_op and isinstance(operation, QuantableOperation):
                for cfg, var in operation.config_with_variable:
                    if not var.is_parameter: continue
                    boost = 1 if cfg.state == QuantizationStates.PASSIVE else self.passive_boost
                    delegates.append(TrainableDelegate(
                        value=var.value, config=cfg, 
                        limit=self.offset_limit, boost=boost, binding=var
                        )
                    )

        # set up optimizer, ready for training.
        optimizer = torch.optim.Adam(params=[d.offset for d in delegates], lr=self.lr)
        while cur_iter < self.max_iter:
            for _ in range(self.steps_per_epoch):
                qt_input, fp_output = dataset.pop()
                # update weights:
                for parameter in delegates:
                    assert isinstance(parameter, TrainableDelegate)
                    parameter.quantize(threshold=threshold)

                qt_input, fp_output = qt_input.to(device), fp_output.to(device)
                qt_output = executor.partial_graph_forward(
                    operations=block, feed_dict={input_var.name: qt_input}, 
                    output_names=[output_var.name])[0]

                # compute loss
                optimizer.zero_grad()
                loss = torch_mean_square_error(qt_output, fp_output)
                loss.backward()
                optimizer.step()

                cur_iter += 1
                losses.append(loss.detach().item())

            # pleatu interval schedule.
            cur_loss = sum(losses) / len(losses)
            if cur_loss < last_loss * .99:
                last_loss, trys_count = cur_loss, 0
                # record loss
                loss_recorder[threshold] = cur_loss
            else:
                trys_count += 1
                if trys_count > self.patient:
                    # rebuild optimizer, clear all state.
                    optimizer.state = collections.defaultdict(dict)
                    trys_count, last_loss, threshold = 0, 1e9, threshold - self.t_step
                    if threshold <= 0.5: break

            # clear loss state
            losses.clear()

        # DEBUG INFO, JUST IN CASE.
        '''
        for offset in [d.offset for d in delegates]:
            print(offset.shape)
            print(' ------ GARD ------')
            print(offset._grad.flatten().max())
            print(' ------ VALUE ------')
            print(offset.flatten().max())
        '''

        # clear all delegates
        for delegate in delegates:
            assert isinstance(delegate, TrainableDelegate)
            delegate.clear()
        
        # display loss
        if self._verbose:
            loss_recorder = {f'{threshold * 100:.1f}%': loss for threshold, loss in loss_recorder.items()}
            print(f'Optimize Result For Block: ', end='')
            # display your block with following ugly code.
            print(block[0].name, end='')
            for operation in block[1:]: print('->' + operation.name, end='')
            print('')
    
            MeasurePrinter(
                data=loss_recorder, measure='MSE', 
                label='Threshold', order=None).print()

        # Check
        if self.check_flag:
            if not self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn):
                for delegate in delegates:
                    assert isinstance(delegate, TrainableDelegate)
                    delegate.withdraw()

        # detach weight
        for delegate in delegates:
            assert isinstance(delegate, TrainableDelegate)
            delegate.binding.value = delegate.binding.value.detach()

    def build_block_from_start(self, graph: BaseGraph, start_op: QuantableOperation) -> List[Operation]:
        path = [start_op]
        while len(graph.get_downstream_operations(start_op)) == 1:
            end_op = graph.get_downstream_operations(start_op)[0]
            if len(graph.get_upstream_operations(end_op)) == 1:
                path.append(end_op)
                start_op = end_op
            else: break

        num_of_computing_ops = sum([1 for op in path if op.is_computing_op])
        while num_of_computing_ops > OPTIM_ADVOPT_GRAPH_MAXSIZE:
            if path[-1].is_computing_op: num_of_computing_ops -= 1
            path.pop(-1)
        return path

    def optimize(
        self, processer: GraphCommandProcesser, dataloader: Iterable,
        executor: TorchExecutor, collate_fn: Callable, **kwargs) -> None:
        
        if self._interested_outputs is None:
            self._interested_outputs = [name for name in processer.graph.outputs]

        if self.collecting_device == 'executor': 
            self.collecting_device = executor._device

        graph = processer.graph
        visited = set()

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

        # set up checkpoints
        if self.check_flag:
            self.initialize_checkpoints(
                graph=graph, executor=executor, 
                dataloader=dataloader, collate_fn=collate_fn)

        for start_op in tqdm(interested_ops, total=len(interested_ops), desc='Advanced Optim Procedure...'):
            assert isinstance(start_op, QuantableOperation)

            if start_op in visited: continue
            block = self.build_block_from_start(graph=graph, start_op=start_op)

            end_op       = block[-1]
            block_input  = start_op.inputs[0]
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
            
            for op in block: visited.add(op)

            # empty cache.
            fp32_outputs.clear()
            quant_inputs.clear()
            empty_cache()
