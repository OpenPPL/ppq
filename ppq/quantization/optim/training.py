import collections
import random
from typing import Callable, Iterable, List, Set, Tuple

import numpy as np
import torch
from numpy import ceil
from ppq.core import (QuantizationProperty, QuantizationStates, empty_ppq_cache)
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import (BaseGraph, GraphCommandProcesser, Operation,
                    QuantableOperation)
from ppq.IR.quantize import QuantableVariable
from ppq.quantization.algorithm.training import (AdaroundRegTerm,
                                                 FinetuneCheckPoint,
                                                 LinearQuantSieve, Lp_norm, make_operation_trainable, make_operation_untrainable)
from ppq.quantization.analyise.graphwise import graph_similarity_analyse
from ppq.quantization.measure import (torch_cosine_similarity_as_loss,
                                      torch_mean_square_error)
from ppq.quantization.qfunction import BaseQuantFunction
from torch.cuda import empty_cache
from tqdm import tqdm

from .base import QuantizationOptimizationPass

BIAS_CORRECTION_INTERST_TYPE = {'Conv', 'Gemm', 'ConvTranspose'}


def has_bias(op: Operation):
    if op.type in {'Conv', 'ConvTranspose', 'Gemm'}:
        return op.meta_data.num_of_input == 3
    else: return False


class TrainingBasedPass(QuantizationOptimizationPass):
    def __init__(self, name: str = 'Default Quanzation Optim') -> None:
        self._loss_fn = torch_cosine_similarity_as_loss
        self._checkpoints = {}
        super().__init__(name=name)

    @ empty_ppq_cache
    def initialize_checkpoints(
        self, graph: BaseGraph, checkpoints: List[str], executor: BaseGraphExecutor,
        dataloader: Iterable, collate_fn: Callable, speckle_size: int=1024):

        for name in checkpoints:
            operation = graph.operations[name]
            assert operation.type in {'Conv', 'ConvTranspose', 'Gemm'}, (
                'Only Conv and Gemm operation can set as a checkpoint operation.')
            output_var = operation.outputs[0]

            numel_per_sample = 1
            for size in output_var.meta.shape[1: ]:
                numel_per_sample *= size

            self._checkpoints[output_var.name] = FinetuneCheckPoint(
                variable=output_var.name, best_loss=float('inf'), references=[],
                speckle=torch.tensor(
                    [random.randint(0, numel_per_sample - 1) for _ in range(speckle_size)],
                    dtype=torch.int, device=executor._device
                )
            )

        # collecting reference:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation): op.dequantize()

        output_names = [name for name in self._checkpoints]
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=output_names)
            for name, output in zip(output_names, outputs):
                ckpt = self._checkpoints[name]
                assert isinstance(ckpt, FinetuneCheckPoint)
                ckpt.references.append(output.flatten(start_dim=1).index_select(dim=1, index=ckpt.speckle))
        for name in self._checkpoints:
            self._checkpoints[name].references = torch.cat(self._checkpoints[name].references, dim = 0)

        # restore quantization state:
        for op in graph.operations.values():
            if isinstance(op, QuantableOperation): op.restore_quantize_state()

    @ empty_ppq_cache
    def anaylse(
        self, executor: BaseGraphExecutor,
        dataloader: Iterable, graph: BaseGraph,
        collate_fn: Callable, measurement: str = 'cosine'):
        assert isinstance(executor, TorchExecutor), (
            'PPQ Training-based optimization algorithm needs a TorchExecutor.')
        similarity_report = graph_similarity_analyse(
            quant_graph=graph, running_device=None, executor=executor,
            interested_op_type=['Gemm', 'Conv'], dataloader=dataloader,
            collate_fn=collate_fn, measurement=measurement, max_steps=32)
        return similarity_report

    def check(self, executor: BaseGraphExecutor,
        dataloader: Iterable, collate_fn: Callable):

        output_names = [name for name in self._checkpoints]
        output_recorder = {name: [] for name in self._checkpoints}

        # step - 1, collecting data
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            outputs = executor.forward(inputs=data, output_names=output_names)
            for name, output in zip(output_names, outputs):
                speckle = self._checkpoints[name].speckle
                output_recorder[name].append(output.flatten(start_dim=1).index_select(dim=1, index=speckle))

        # step - 2, calculating loss
        current_losses = {}
        for name in self._checkpoints:
            fp32_out = self._checkpoints[name].references
            quant_out = torch.cat(output_recorder[name], dim=0)
            current_loss = self._loss_fn(y_pred=quant_out, y_real=fp32_out)
            current_losses[name] = current_loss.item()

        # step - 3, comparing loss, check if there is any loss larger than before.
        loss_toleration, check_flag = 1.02, True
        for name in current_losses:
            current_loss, best_loss = current_losses[name], self._checkpoints[name].best_loss
            if current_loss > best_loss * loss_toleration: check_flag = False
        if sum([value for value in current_losses.values()]) > \
            sum([ckpt.best_loss for ckpt in self._checkpoints.values()]):
            check_flag = False

        # if there is a perfect loss drop, update all losses.
        if check_flag:
            for name in current_losses:
                self._checkpoints[name].best_loss = current_losses[name]
        return check_flag

    def optimize(
        self, processer: GraphCommandProcesser,
        dataloader: Iterable, executor: BaseGraphExecutor, **kwargs) -> None:
        return super().optimize(processer, dataloader, executor, **kwargs)


class BiasCorrectionPass(TrainingBasedPass):
    def __init__(self,
        quantize_function: BaseQuantFunction,
        auto_check: bool=False) -> None:
        super().__init__(name='PPQ Bias Correction Pass')
        self._quantize_function = quantize_function
        self._auto_check = auto_check

    @ empty_ppq_cache
    def optimize(
        self,
        processer: GraphCommandProcesser,
        dataloader: Iterable,
        executor: BaseGraphExecutor,
        collate_fn: Callable,
        **kwargs
    ) -> None:
        def collect_bias(output: torch.Tensor, collector: torch.Tensor, op_type: str):
            if op_type in {'Conv', 'ConvTranspose'}: collector += torch.sum(output, dim=(0, 2, 3))
            elif op_type in {'Gemm'}: collector += torch.sum(output, dim=(0, ))
            else: raise TypeError(f'Unsupported Operation type: {op_type}')

        graph = processer.graph
        assert isinstance(executor, TorchExecutor), ('PPQ Training-based optimization algorithm needs a TorchExecutor.')

        op_reports = self.anaylse(graph=graph, executor=executor, dataloader=dataloader, collate_fn=collate_fn)
        op_reports = sorted(list(op_reports.items()), key=lambda x: x[-1])[: 10]
        for op, sim in op_reports: print('PPQ checkpoint is created with %s, similarity now %.3f' %(op, sim * 100))
        checkpoints=[name for (name, _) in op_reports]
        self.initialize_checkpoints(graph=graph, executor=executor, dataloader=dataloader,
            collate_fn=collate_fn, checkpoints=checkpoints)

        sorted_ops = graph.topological_sort()
        for idx, target_op in enumerate(tqdm(sorted_ops, desc='Bias Correction Procedure.', total=len(sorted_ops))):
            if (isinstance(target_op, QuantableOperation) and
                target_op.type in BIAS_CORRECTION_INTERST_TYPE and has_bias(target_op)):
                interested_var = [target_op.outputs[0].name]

                bias_value = target_op.inputs[-1].value
                q_collector, f_collector = torch.zeros_like(bias_value), torch.zeros_like(bias_value)
                num_of_samples, preorder_ops = 0, sorted_ops[: idx + 1]

                for preorder_op in preorder_ops:
                    preorder_op = graph.operations[preorder_op]
                    if isinstance(preorder_op, QuantableOperation):
                        preorder_op.dequantize()
                for data in dataloader:
                    if collate_fn is not None: data = collate_fn(data)
                    [output] = executor.forward(inputs=data, output_names=interested_var)
                    collect_bias(output, f_collector, op_type=target_op.type)

                for preorder_op in preorder_ops:
                    preorder_op = graph.operations[preorder_op]
                    if isinstance(preorder_op, QuantableOperation):
                        preorder_op.restore_quantize_state()
                for data in dataloader:
                    if collate_fn is not None: data = collate_fn(data)
                    [output] = executor.forward(inputs=data, output_names=interested_var)
                    collect_bias(output, q_collector, op_type=target_op.type)
                    num_of_samples += (output.numel() / output.shape[1])

                backup = target_op.inputs[-1].value.clone()
                target_op.inputs[-1].value = bias_value + (f_collector - q_collector) / num_of_samples
                if target_op.config.input_quantization_config[-1].state == QuantizationStates.PASSIVE_BAKED:
                    target_op.baking_parameters(self._quantize_function)
                if not self.check(executor=executor, dataloader=dataloader, collate_fn=collate_fn):
                    target_op.inputs[-1].value = backup
                for ckpt in self._checkpoints.values():
                    print('PPQ checkpoint is created with %s, loss now %.3f' %(ckpt.monitor_var, ckpt.best_loss))


class AdaRoundPass(QuantizationOptimizationPass):
    def __init__(self,
                 collecting_device: str = 'cpu',
                 epoch: int = 512,
                 batch_size: int = 32) -> None:
        super().__init__(name='PPQ Adaquant Pass')
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
            if target_op.meta_data.num_of_input == 3:
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

                    # soft ada quant weight
                    params[0] = self.ada_quant_weight(fp_weight, weight_scale, weight_offset, weight_quantization_config, continuous_v)
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
            rounded_weight = self.ada_quant_weight(fp_weight, weight_scale, weight_offset, weight_quantization_config, continuous_v, soft=False)
            target_op.parameters[0].value.copy_(rounded_weight)
            del fp_outputs_concat
            del quant_inputs_concat
            target_op.config.input_quantization_config[1].state = QuantizationStates.ACTIVATED
            if bias is not None:
                target_op.parameters[1].value.copy_(bias)
                target_op.config.input_quantization_config[-1].state = QuantizationStates.PASSIVE

    def ada_quant_weight(self, weight, scale, offset, weight_quantization_config, round_var, soft=True):
        quant_max = weight_quantization_config.quant_max
        quant_min = weight_quantization_config.quant_min
        if soft:
            weight = (weight / scale).floor() + AdaroundRegTerm().rectified_sigmoid(round_var)
        else:
            weight = (weight / scale).floor() + (round_var >= 0).float()
        weight = torch.clamp(weight + offset, quant_min, quant_max)
        weight = (weight - offset) * scale
        return weight


class AdvancedQuantOptimization(TrainingBasedPass):
    """
    PPQ Advanced Quantization Optimization Pass
    
    This optimization pass minimize the quantization errors of each layer separately
        by optimizing its parameters over the calibration set.

    Where:
        qout = quant( quant(W + W_offset) * quant(X) + quant(bias + bias_offset) )
    
        fout = W * B + bias
    
        error = Mean(((qout - fout) / scale)^2)
    
    This training procedure manage to find W_offest and bias_offset to minimize error

    Based on your setting and network size, the training procedure will takes 10~120 minutes.

    Args:
        TrainingBasedPass ([type]): [description]
    """
    def __init__(self, collecting_device: str, offset_limit: float = 2.0, lr: float = 1e-4,
                 interested_types: Set[str] = {'Gemm', 'Conv', 'ConvTranspose'}, max_trys: int = 8,
                 step: float = 0.1, check: bool = True, correct_bias: bool = True, 
                 rand_initial: bool = False) -> None:
        super().__init__(name='PPQ Advanced Quantization Optimization Procedure')
        
        self.lr                = lr
        self.collecting_device = collecting_device
        self.check_flag        = check
        self.offset_limit      = offset_limit
        self.interested_types  = interested_types
        self.step              = step
        self.correct_bias      = correct_bias
        self.max_trys          = max_trys
        self.rand_initial      = rand_initial
        self.passive_boost     = 128
        self.max_iter          = 10000

    def get_output_names(self, target_op: QuantableOperation) -> str:
        return target_op.outputs[0].name

    def collect_training_data(self,
        target_op: QuantableOperation,
        dataloader: Iterable, 
        executor: BaseGraphExecutor, 
        collate_fn: Callable) -> List[List[torch.Tensor]]:

        # optimization takes effects only with Gemm and Conv layer.
        # For those layer input[0] and output[0] represent input data and output data.
        input_names  = [target_op.inputs[0].name]
        output_names = [target_op.outputs[0].name]

        quant_inputs, fp32_outputs = [], []
        # dequantize target operation to get fp32 outputs.
        target_op.dequantize()
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            output = executor.forward(data, output_names=output_names)[0]
            fp32_outputs.append(output.to(self.collecting_device))

        # restore target operation to get quant input.
        target_op.restore_quantize_state()
        for data in dataloader:
            if collate_fn is not None: data = collate_fn(data)
            output = executor.forward(data, output_names=input_names)[0]
            quant_inputs.append(output.to(self.collecting_device))

        return fp32_outputs, quant_inputs

    @ empty_ppq_cache
    def finetune(self, quant_inputs: List[torch.Tensor], fp32_outputs: List[torch.Tensor],
        executor: TorchExecutor, target_op: QuantableOperation) -> List[torch.Tensor]:
        
        # initialize all parameters.
        losses     = []
        last_loss  = 1e9
        threshold  = .99
        trys_count = 0
        cur_iter   = 0
        device     = executor._executing_contenxt.executing_device
        scale      = target_op.config.output_quantization_config[0].scale
        
        # create offset will make operation trainable.
        offsets    = make_operation_trainable(operation=target_op, random_initial=self.rand_initial)
        parameters = [parameter.value for parameter in target_op.parameters]
        configs    = [cfg for cfg, var in target_op.config_with_variable if var.is_parameter]
        best_offsets = [offset.clone() for offset in offsets]
        optimizer = torch.optim.Adam(params=offsets, lr=self.lr)

        while cur_iter < self.max_iter:
            for input_value, output_ref in zip(quant_inputs, fp32_outputs):
                optimizer.zero_grad()
                
                input_value = input_value.to(device)
                output_ref  = output_ref.to(device)

                inputs = [input_value]
                for offset, parameter, config in zip(offsets, parameters, configs):
                    # PASSIVE Parameters has a much smaller scale and was quantized with 32 bit precision.
                    # boost it first.
                    if config.state == QuantizationStates.PASSIVE:
                        inputs.append(LinearQuantSieve(
                            offset * self.passive_boost, 
                            parameter, self.offset_limit, config, threshold))

                    else:
                        inputs.append(LinearQuantSieve(
                            offset, parameter, self.offset_limit, config, threshold))
                
                output = executor.operation_forward(
                    target_op, inputs=inputs, 
                    quantize_output=True, 
                    quantize_input=False)[0]

                # compute loss
                loss = torch_mean_square_error(output / scale, output_ref / scale)
                loss.backward()
                optimizer.step()

                cur_iter += 1
                losses.append(loss.detach().item())

            # pleatu interval schedule.
            cur_loss = sum(losses) / len(losses)
            if cur_loss < last_loss * .99:
                last_loss, trys_count = cur_loss, 0
                print('Loss: %.4f, threshold: %.2f, last Loss: %.4f' % (cur_loss, threshold, last_loss))
                # update best offsets
                for tensor, offset in zip(best_offsets, offsets):
                    tensor.copy_(offset)
            else:
                trys_count += 1
                if trys_count > self.max_trys:
                    # rebuild optimizer, clear all state.
                    optimizer.state = collections.defaultdict(dict)
                    trys_count, last_loss, threshold = 0, 1e9, threshold - self.step
                    if threshold <= 0.5: break

            # clear loss state
            losses.clear()

        # clear all grads(save gpu memeory):
        for offset, parameter in zip(offsets, parameters):

            offset.requires_grad = False
            offset._grad = None

            parameter.requires_grad = False
            parameter._grad = None
        
        return [LinearQuantSieve(offset, parameter, self.offset_limit, config, 0) 
                for offset, parameter, config in zip(offsets, parameters, configs)]

    def bias_correction(self,
        quant_inputs: List[torch.Tensor], 
        fp32_outputs: List[torch.Tensor],
        executor: TorchExecutor, 
        target_op: QuantableOperation):

        if not target_op.is_computing_op or not has_bias(target_op): return
        bias = target_op.inputs[-1].value

        errors = []
        for input_value, output_reference in zip(quant_inputs, fp32_outputs):
            input_value      = input_value.to(executor._device)
            output_reference = output_reference.to(executor._device)

            params = [var.value for var in target_op.inputs[1: ]]
            output = executor.operation_forward(
                target_op, inputs=[input_value, ] + params)[0]

            if target_op.type in {'Conv', 'ConvTranspose'}:
                output_diff = (output_reference - output).mean(dim=[0, 2, 3]).unsqueeze(0)
            else: output_diff = (output_reference - output).mean(dim=[0]).unsqueeze(0)
            errors.append(output_diff)

        bias_err = torch.cat(errors, dim=0).mean(dim=0)
        assert bias_err.shape == bias.shape
        bias += bias_err

    def optimize(
        self, processer: GraphCommandProcesser, dataloader: Iterable,
        executor: TorchExecutor, collate_fn: Callable, **kwargs) -> None:

        if self.collecting_device == 'executor': 
            self.collecting_device = executor._device

        graph = processer.graph

        # mark all operations that need to be tuned.
        interested_ops = []
        for target_op in graph.topological_sort():
            if isinstance(target_op, QuantableOperation) and target_op.type in self.interested_types:
                interested_ops.append(target_op)

        if self.check_flag:
            # set up check points.
            op_reports = self.anaylse(graph=graph, executor=executor, 
                                      dataloader=dataloader, collate_fn=collate_fn)
            op_reports = sorted(list(op_reports.items()), key=lambda x: x[-1])[: 4]

            checkpoints=[name for (name, _) in op_reports]
            self.initialize_checkpoints(graph=graph, executor=executor, dataloader=dataloader,
                collate_fn=collate_fn, checkpoints=checkpoints)

        for target_op in tqdm(interested_ops, total=len(interested_ops), desc='Advanced Optim procedure...'):
            assert isinstance(target_op, QuantableOperation)
            fp32_outputs, quant_inputs = self.collect_training_data(
                target_op=target_op, dataloader=dataloader, 
                executor=executor, collate_fn=collate_fn)

            # start training, solve the best parameters
            optimized_params = self.finetune(
                quant_inputs=quant_inputs, fp32_outputs=fp32_outputs,
                executor=executor, target_op=target_op)

            # rewrite target operation's weight
            for parameter, optimized_value in zip(target_op.parameters, optimized_params):
                parameter.value.copy_(optimized_value)

            # bias correction procedure.
            if self.correct_bias:
                self.bias_correction(quant_inputs=quant_inputs, fp32_outputs=fp32_outputs,
                    executor=executor, target_op=target_op)

            if self.check_flag and not self.check(
                executor=executor, dataloader=dataloader, collate_fn=collate_fn):
                for parameter in target_op.parameters:
                    assert isinstance(parameter, QuantableVariable), (
                        f'Oops, seems we got an non-quantizable parameter {parameter.name} here.')
                    parameter.value.copy_(parameter.stored_value.to(executor._device))

            make_operation_untrainable(operation=target_op, baking_function=executor.quantize_function)

            # empty cache.
            optimized_params.clear()
            fp32_outputs.clear()
            quant_inputs.clear()
            empty_cache()