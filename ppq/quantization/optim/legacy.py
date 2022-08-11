# Legacy Optimization Passes
from collections import defaultdict
from math import ceil
from typing import Callable, Dict, Iterable, List

import numpy as np
import torch
from numpy import ceil
from ppq.core import *
from ppq.core import QuantizationStates, TensorMeta, ppq_warning
from ppq.executor import BaseGraphExecutor, TorchExecutor
from ppq.IR import (BaseGraph, BaseGraph, Operation, Path,
                    QuantableOperation, SearchableGraph)
from ppq.IR.quantize import QuantableVariable
from ppq.IR.search import TraversalCommand
from ppq.quantization.algorithm.training import *
from ppq.quantization.measure import torch_mean_square_error
from ppq.quantization.observer import TensorObserverFactroy
from tqdm import tqdm

from .base import QuantizationOptimizationPass
from .training import TrainingBasedPass


class TimeDecay:
    """A helper class computing time decay."""
    def __init__(self, t_max: int, decay: float=0.2, beta_start: float=20, beta_end:float=2):
        self.t_max = t_max
        self.start_decay = decay * t_max
        self.start_b = beta_start
        self.end_b = beta_end

    def __call__(self, t):
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class AdaroundRegTerm(torch.nn.Module):
    """Adaround Reg Term is a part of Adaround optimization algorithm.
    This term represents the difference between a fp32 value and its quantized counter-part.
        We use a same implementation as proposed in Adaround paper.
    Args:
        torch ([type]): [description]
    """
    def __init__(self, max_iter: int = 20000,
                 zeta: float = 1.1, gamma:float = -0.1,
                 alpha: float = 0.01, beta: float = 20,
                 warm_ratio: float = 0.2):
        self.max_iter = max_iter
        self.zeta = zeta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.warm_ratio = warm_ratio
        self.temp_anneal = TimeDecay(self.max_iter, self.warm_ratio)
        super().__init__()

    def rectified_sigmoid(self, r: torch.Tensor) -> torch.Tensor:
        return ((self.zeta - self.gamma) * torch.sigmoid(r) + self.gamma).clamp(0, 1)

    def forward(self, r: torch.Tensor, iter: int) -> torch.Tensor:
        if iter < self.max_iter * self.warm_ratio:
            round_loss = 0
        else:
            self.beta = self.temp_anneal(iter)
            round_loss = self.alpha * (1 - torch.pow((self.rectified_sigmoid(r) - 0.5).abs() * 2, self.beta)).sum()
        return round_loss


class AdaRoundDelegator(TorchQuantizeDelegator):
    def __init__(
        self, var: QuantableVariable,
        config: TensorQuantizationConfig, 
        steps: int,
    ) -> None:
        self.reg                    = AdaroundRegTerm(max_iter=steps)
        self.config                 = config
        self.var                    = var
        self.is_parameter           = self.var.is_parameter
        self.rounding               = self.initiate_rounding(value=self.var.value, config=self.config, zeta=1.1, gamma=-0.1)

        if not self.var.is_parameter:
            raise TypeError(f'Can not create adaround delegator with variable {var.name}, '
                            'Adaround delegator works only with parameter.')
        if self.config.state == QuantizationStates.PASSIVE:
            raise TypeError(f'Can not create adaround delegator with variable {var.name}, '
                            'Adaround delegator can not work with passive parameter.')
        self.param_backup = None
        if self.is_parameter:
            self.param_backup = self.var.value.clone()

    @ staticmethod
    def initiate_rounding(value: torch.Tensor, config: TensorQuantizationConfig, zeta: float, gamma: float) -> torch.Tensor:
        with torch.no_grad():
            scale, offset = config.scale, config.offset
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                assert isinstance(config, ChannelwiseTensorQuantizationConfig)
                shape = [1 if axis != config.channel_axis else -1 for axis in range(value.ndim)]
                scale = scale.view(shape)

            rounding = (value / scale) - (value / scale).floor()
            rounding = - torch.log((zeta - gamma) / (rounding - gamma) - 1)
            rounding = torch.zeros_like(rounding).copy_(rounding)
            rounding.requires_grad = True
        return rounding

    def trainable_tensors(self) -> List[torch.Tensor]:
        tensors = [self.rounding]
        return tensors

    def finalize(self) -> None:
        weight, scale, offset = self.var.value, self.config.scale, self.config.offset
        if self.config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            assert isinstance(self.config, ChannelwiseTensorQuantizationConfig)
            shape = [1 if axis != self.config.channel_axis else -1 for axis in range(weight.ndim)]
            scale = scale.view(shape)
            offset = offset.view(shape)
        weight = (weight / scale).floor() + (self.rounding >= 0).float()
        weight = torch.clamp(weight + offset, self.config.quant_min, self.config.quant_max)
        weight = (weight - offset) * scale
        self.var.value = weight
    
    def withdraw(self) -> None:
        with torch.no_grad():
            self.var.value.copy_(self.param_backup)

    def __call__(self, tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
        scale = config.scale
        offset = config.offset
        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            assert isinstance(config, ChannelwiseTensorQuantizationConfig)
            shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
            scale = scale.view(shape)
            offset = offset.view(shape)
        tensor = (tensor / scale).floor() + self.reg.rectified_sigmoid(self.rounding)
        tensor = torch.clamp(tensor + offset, config.quant_min, config.quant_max)
        tensor = (tensor - offset) * scale
        return tensor

    def regularization_loss(self, step: int) -> torch.Tensor:
        return self.reg.forward(r=self.rounding, iter=step)


class ChannelSplitPass(QuantizationOptimizationPass):
    """ChannelSplitPass is designed for per-tenser quantization only, this
    implementation is based on the original paper:

      "zhao, Ritchie et al., Improving Neural Network Quantization without Retraining using Outlier Channel Splitting"

      Basically this pass shrinks ranges of outlier channels by first half-down the channel value
          then duplicate the whole channel, making it more friendly for per-tensor quantization
          while preserving the fp output same

      In this implementation, to avoid bringing in supplemental ops, for each user-given op, we find its counterpart op,
      split input/output channel of the user-given op and duplicate output/input channel of its counterpart

              split Conv1                                          split Conv2

      Conv1  --  Relu  --  Conv2                               Conv1  --  Relu  --  Conv2
    (C2,C1,H1,W1)      (C3,C2,H2,W2)                        (C2,C1,H1,W1)      (C3,C2,H2,W2)
          || split          || duplicate                          || duplicate       || split
          \/ duplicate      \/                                    \/                 \/ duplicate
    (C2+C,C1,H1,W1)    (C3,C2+C,H2,W2)                      (C2+C,C1,H1,W1)    (C3,C2+C,H2,W2)
    """
    def __init__(self,
                interested_layers: List[str],
                search_directions: List[str] = None,
                expand_ratio: float=0.1,
                split_ratio: float=0.5,
                grid_aware: bool=True
    ) -> None:
        """ChannelSplitPass, try this when other algorithms fail to improve
        your per-tensor quantization accuracy, interested_layers and
        corresponding search_directions should decided by user, user should
        make sure every split operation in interested_layers has a counterpart
        along the corresponding search direction.

        Args:
            interested_layers (List[str]): a list of strings representing ops you want to apply channel split.
            search_directions (List[str]): a list of strings representing search directions(up or down) of ops given in
                                           interested_layers by user.
            expand_ratio (float, optional): ratio of duplicate channels. Defaults to 0.1.
            split_ratio (float, optional): ratio of values in outlier channel which will half-down. Defaults to 0.5.
            grid_aware (bool, optional): whether to apply quantization aware split. Defaults to True.
        """
        self.interested_layers = interested_layers
        self.search_directions = search_directions

        if not self.search_directions or len(search_directions) != len(interested_layers):
            ppq_warning('You do not provide a valid search direction. '
                        'All layer will split with its upstream layers by default.')
            self.search_directions = ['up' for _ in self.interested_layers]

        self.expand_ratio = expand_ratio
        self.grid_aware = grid_aware
        self.split_ratio = split_ratio
        self.current_search_direction = None
        super().__init__(name='Channel Split Pass')

    def calculate_scale(self, split_op: QuantableOperation) -> torch.Tensor:
        config = split_op.config.input_quantization_config[1]
        observer = TensorObserverFactroy.build_observer(split_op.parameters[0], config)
        observer.observe(split_op.parameters[0].value)
        observer.render_quantization_config()
        return config.scale

    def flip(self, op: Operation) -> bool:
        return (self.current_search_direction == 'down') != (op.type == 'ConvTranspose' or (op.type == 'Gemm'\
            and op.attributes.get('transB', 0) == 0))

    def OCS_forward(self, split_op: Operation) -> List[int]:
        weight = split_op.parameters[0].value
        axes = list(range(weight.ndim))

        # update bias when the out dimension needs half-down and duplicate
        update_bias = (self.current_search_direction == 'down' and len(split_op.parameters) > 1)
        if update_bias:
            bias = split_op.parameters[1].value

        # permute weight so that we can always operate on the second axis
        if self.flip(split_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        num_channels = weight.shape[1]
        ocs_channels = ceil(num_channels * self.expand_ratio)
        in_channels_to_copy = []
        orig_idx_dict = {}
        if self.grid_aware:
            assert isinstance(split_op, QuantableOperation), (
                f'Operation {split_op.name} is not quantable, can not be splited via this function.')
            w_scale = self.calculate_scale(split_op)

        for c in range(ocs_channels):
            flatten_weight = weight.permute(1, 0, *axes[2:]).contiguous()
            flatten_weight = flatten_weight.reshape(flatten_weight.shape[0], -1)
            max_per_channel = torch.max(flatten_weight.abs(), 1)[0]
            idxs = torch.argsort(max_per_channel, descending=True)
            split_idx = idxs[0].item()
            ch_slice = weight[:, split_idx:(split_idx + 1)].clone()
            ch_slice_half = ch_slice / 2
            ch_slice_zero = torch.zeros_like(ch_slice)

            # for a top-down search, we need to directly half-down the weight and bias
            if self.current_search_direction == 'up':
                split_value = ch_slice.max() * self.split_ratio
            else:
                split_value = 0

            if (not self.grid_aware) or self.current_search_direction == 'down':
                ch_slice_1 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half, ch_slice)
                ch_slice_2 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half, ch_slice_zero)
            else:
                # assert per-tensor
                ch_slice_half /= w_scale
                ch_slice_1 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half-0.25, ch_slice / w_scale) * w_scale
                ch_slice_2 = torch.where(torch.abs(ch_slice) > split_value, ch_slice_half+0.25, ch_slice_zero) * w_scale
            weight[:, split_idx:(split_idx+1)] = ch_slice_1
            weight = torch.cat([weight, ch_slice_2], dim=1)

            if update_bias:
                bias_slice_half = bias[split_idx:(split_idx+1)] / 2
                bias[split_idx] = bias_slice_half
                bias = torch.cat([bias, bias_slice_half], dim=0)

            if split_idx < num_channels:
                in_channels_to_copy.append(split_idx)
                orig_idx_dict[num_channels+c] = split_idx
            else:
                in_channels_to_copy.append(orig_idx_dict[split_idx])
                orig_idx_dict[num_channels+c] = orig_idx_dict[split_idx]

        # permute back
        if self.flip(split_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        # update param
        split_op.parameters[0].value = weight
        if update_bias:
            split_op.parameters[1].value = bias
        return in_channels_to_copy

    def update_counterpart(self, counterpart_op: Operation, in_channels_to_copy: List[int]) -> None:
        weight = counterpart_op.parameters[0].value
        axes = list(range(weight.ndim))

        # permute weight so that we can always operate on the first axis
        if self.flip(counterpart_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        weight_split = torch.index_select(weight, dim=0, index=torch.tensor(in_channels_to_copy, dtype=torch.int64, device=weight.device))
        weight = torch.cat([weight, weight_split], dim=0)

        # update bias when the output dimension needs duplicate
        update_bias = (self.current_search_direction == 'up' and len(counterpart_op.parameters) > 1)
        if update_bias:
            bias = counterpart_op.parameters[1].value
            bias_split = torch.index_select(bias, dim=0, index=torch.tensor(in_channels_to_copy, dtype=torch.int64, device=bias.device))
            bias = torch.cat([bias, bias_split], dim=0)

        # flip back
        if self.flip(counterpart_op):
            weight = weight.permute(1, 0, *axes[2:]).contiguous()

        # update param
        counterpart_op.parameters[0].value = weight
        if update_bias:
            counterpart_op.parameters[1].value = bias

    def check(self, graph: BaseGraph, path:Path) -> bool:
        if self.current_search_direction == 'up':
            for op in path.tolist()[1:]:
                if len(graph.get_downstream_operations(op)) != 1:
                    return False
            upstream_op, downstream_op = path[-1], path[0]
        else:
            for op in path.tolist()[:-1]:
                if len(graph.get_downstream_operations(op)) != 1:
                    return False
            upstream_op, downstream_op = path[0], path[-1]
        # not support group conv yet
        if upstream_op.attributes.get('group', 1) != 1 or downstream_op.attributes.get('group', 1) != 1:
            return False
        # should have as least one weight parameter
        if upstream_op.type == 'Gemm' and len(upstream_op.parameters) < 1 or\
            downstream_op.type == 'Gemm' and len(downstream_op.parameters) < 1:
            return False

        # check if weight shapes of upstream and downstream computing ops match
        up_axis, down_axis = 0, 1
        if upstream_op.type == 'ConvTranspose' or (upstream_op.type == 'Gemm' and upstream_op.attributes.get('transB', 0) == 0):
            up_axis = 1
        if downstream_op.type == 'ConvTranspose' or (downstream_op.type == 'Gemm' and downstream_op.attributes.get('transB', 0) == 0):
            down_axis = 0

        if upstream_op.parameters[0].meta.shape[up_axis] != downstream_op.parameters[0].meta.shape[down_axis]:
            return False

        return True

    def modify_meta(self, path:Path, num_channels:int) -> None:
        # all the activations along the path and changed params
        # needs modifying their meta info, i.e., add up the
        # duplicated channels to the second dimension
        for op in path.tolist():
            for var in op.inputs:
                if var.is_parameter:
                    op.meta_data.input_metas[op.inputs.index(var)] = TensorMeta.parsing_from_torch_tensor(var.value)
        path_ = path.tolist()[1:] if self.current_search_direction == 'up' else path.tolist()[:-1]
        for op in path_:
            output_meta = op.meta_data.output_metas[0]
            shape = output_meta.shape
            shape[1] += num_channels

    def store_parameter(self, path: Path) -> None:
        for op in path:
            if isinstance(op, QuantableOperation):
                op.store_parameter_value()

    def initiate_path_state(self, path: Path) -> None:
        for op in path:
            if isinstance(op, QuantableOperation):
                for quant_config in op.config.input_quantization_config + op.config.output_quantization_config:
                    if quant_config.state == QuantizationStates.ACTIVATED:
                        quant_config.state = QuantizationStates.INITIAL
                    elif quant_config.state == QuantizationStates.PASSIVE:
                        quant_config.state = QuantizationStates.PASSIVE_INIT


    def optimize(self, graph: BaseGraph,
                 dataloader: Iterable, executor: BaseGraphExecutor,
                 **kwargs) -> None:

        search_engine = SearchableGraph(graph)

        for name, search_direction in zip(self.interested_layers, self.search_directions):
            if name not in graph.operations:
                ppq_warning(f'Can not find operation {name} in your graph, skip its split.')
                continue
            op = graph.operations[name]
            if not op.is_computing_op or not isinstance(op, QuantableOperation):
                ppq_warning(f'Operation {name} can not be splited via channel spilt function, '
                            'cause it is not quantable or it has no parameter.')
                continue

            self.current_search_direction = search_direction
            matching = search_engine.path_matching(
                sp_expr=lambda x: x.name == name,
                rp_expr=lambda x, y: True, # be careful when choosing interested_layers, we assume a reasonable path
                ep_expr=lambda x: x.is_computing_op,
                direction=search_direction
            )

            if len(matching) != 1:
                ppq_warning(f'Can not find a counterpart of operation {name}, '
                            'graph is too complex.')
                continue

            path = matching[0]
            if not self.check(graph, path):
                logger.warning(f'Not support such path due to op constraints for now')
                continue

            if search_direction == 'up':
                logger.info(f"Now processing path {'--'.join(reversed([op.name for op in path]))}")
            else:
                logger.info(f"Now processing path {'--'.join([op.name for op in path])}")

            split_op, counterpart_op = path[0], path[-1]

            copy_channels = self.OCS_forward(split_op)
            self.update_counterpart(counterpart_op, copy_channels)
            self.modify_meta(path, len(copy_channels))
            self.store_parameter(path)
            self.initiate_path_state(path)


class AdaroundPass(TrainingBasedPass):
    """Blockwise Reconstruction Pass, perform adaround block by block, if you
    specify interested_layers in the setting, then only block which contains
    any of operations specified in interested_layers will be optimized,
    otherwise all searched blocks will be optimized. A standard procedure is,
    first turn all training-based optimization passes off in your quantization
    setting and run a plain quantization, then use error analysis tool(provided
    by ppq) to analysis snr error or cosine similarities of every layer, choose
    names of those with significant snr or poor similarities as your
    interested_layers, then turn on this pass and do optimization. In case you
    have no idea which layers should be selected as interested_layers, simply
    leave it as blank and all blocks will be tuned. Note that you could control
    the maximum number of operations in a block by setting
    OPTIM_ADVOPT_GRAPH_MAXSIZE in ppq.core.common, and by default every block
    will be trained for 300 epochs, which takes certain long time. The
    optimization goal of every block is.
                Loss = LpNormLoss(y, y^) + lambda * rounding_loss(v)
    where y is the output of the current block running in fp32 mode, and y^ is the output of the current block running
    in quant mode, lambda is a hyperparameter adjusting scales of rounding loss, and v is the element-wise rounding
    parameter applied to weights of every computing op in the block.
    """
    def __init__(self, name: str = 'Block-wise Adaround Reconstruction',
        interested_layers: List[str] = [], is_scale_trainable: bool = False,
        steps: int = 8000, lr: float = 1e-3, gamma: float = 1.0,
        collecting_device: str ='cuda', block_size: int = 4
    ) -> None:
        super().__init__(name = name)
        self.interested_layers  = interested_layers
        self.lr                 = lr
        self.gamma              = gamma
        self.steps              = steps
        self.block_size         = block_size
        self.collecting_device  = collecting_device
        self.is_scale_trainable = is_scale_trainable
        self.loss_fn            = torch_mean_square_error


    def tune_block_weight_scale(
        self, block: TrainableBlock, steps: int=900, 
        loss_fn: Callable = torch_mean_square_error) -> None:
        # before we tune weight roundings and activation scales, we optimize weight scale by
        # minimizing MSE(W, W^), 900 epochs would be enough in this non-overfit setting. Note
        # that this is usually unnecessary in 8 bit quantization, but we do it it anyway and
        # the loss checking procedure makes sure we always obtain no worse results.
        for op in block.rps:
            if op.is_computing_op and isinstance(op, QuantableOperation):
                c, v = op.input_quant_config[1], op.inputs[1]
                delegator = LSQDelegator(config=c, var=v, is_parameter_trainable=False)
                params = delegator.trainable_tensors()
                if len(params) == 0: continue

                optimizer = torch.optim.Adam(params, lr=self.lr)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(steps / 2), int(steps * 2 / 3)])

                initial_loss = loss_fn(delegator(tensor=v.value, config=c), v.value)
                for _ in range(steps):
                    optimizer.zero_grad()
                    loss = loss_fn(delegator(tensor=v.value, config=c), v.value)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                post_loss = loss_fn(delegator(tensor=v.value, config=c), v.value)

                if post_loss > initial_loss:
                    delegator.withdraw()


    def finetune(self, steps: int, learning_rate: float, block: TrainableBlock, executor: TorchExecutor,
        qt_inputs: List[Dict[str, torch.Tensor]], fp_outputs: List[Dict[str, torch.Tensor]], 
        optimizer: torch.optim.Optimizer=None, scheduler: object=None) -> Tuple[float, float]:

        # step - 1: enable gradient for training.
        self.enable_block_gradient(block)

        # record pre training loss.
        pre_loss = self.compute_block_loss(
            block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs,
            executor=executor, loss_fn=self.loss_fn)

        # tune block weight scale
        self.tune_block_weight_scale(block=block)

        # collect trainable params
        trainable_params, delegators = [], {}
        for op in block.rps:
            if not isinstance(op, QuantableOperation): continue

            # register quant delegator
            for cfg, var in op.config_with_variable:
                if cfg.state not in {QuantizationStates.ACTIVATED, QuantizationStates.SLAVE}: continue
                if var.is_parameter and cfg.state not in {QuantizationStates.PASSIVE}:
                    delegator = AdaRoundDelegator(config=cfg, var=var, steps=steps)
                    trainable_params.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator
                elif self.is_scale_trainable: 
                    delegator = LSQDelegator(config=cfg, var=var, is_offset_trainable=False)
                    trainable_params.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator

        # check if empty.
        tensors = [tensor for tensor in trainable_params if tensor.requires_grad]
        if len(tensors) == 0:
            for cfg, delegator in delegators.items():
                executor.remove_quantize_delegate(config=cfg)
            return 0, 0

        # initilize optimizer.
        if optimizer is None:
            optimizer = torch.optim.Adam(tensors, lr=learning_rate)

        dataset_length = len(qt_inputs)
        if dataset_length == 0: raise ValueError('Dataset is empty.')

        # step 2 - training procedure
        for idx in tqdm(range(steps), desc='# Tuning Procedure '):
            qt_input, fp_output = qt_inputs[idx % dataset_length], fp_outputs[idx % dataset_length]

            # forward
            optimizer.zero_grad()
            feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}
            output_names = [name for name in fp_output]

            qt_output = executor.partial_graph_forward(
                operations=block.rps, feed_dict=feed_dict, 
                output_names=output_names)

            # compute loss
            loss = 0.0
            for idx, name in enumerate(output_names):
                loss += self.loss_fn(qt_output[idx], fp_output[name].to(executor._device))
            
            # collect reg terms
            for delegator in delegators.values():
                if isinstance(delegator, AdaRoundDelegator):
                    loss += delegator.regularization_loss(idx) * self.gamma

            # backward from loss
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()

        # step - 3: record post training loss
        post_loss = self.compute_block_loss(
            block=block, qt_inputs=qt_inputs, fp_outputs=fp_outputs,
            executor=executor, loss_fn=self.loss_fn)

        # check and withdraw
        for cfg, delegator in delegators.items():
            if post_loss > pre_loss: delegator.withdraw()
            else: delegator.finalize()
            executor.remove_quantize_delegate(config=cfg)

        # disable gradient for evaluation.
        self.disable_block_gradient(block)

        # clear cache
        torch.cuda.empty_cache()
        return pre_loss, post_loss


    def optimize(self,
                 graph: BaseGraph,
                 dataloader: Iterable,
                 executor: TorchExecutor,
                 collate_fn: Callable,
                 **kwargs
    ) -> None:
        blocks = self.split_graph_into_blocks(
            graph=graph, executing_order=executor._executing_order, 
            blocksize=self.block_size, interested_layers=self.interested_layers)

        # do per-block finetune
        for block_idx, block in enumerate(blocks):
            # collect data for training
            qt_inputs, fp_outputs = self.collect(
                graph=graph, block=block, executor=executor, 
                dataloader=dataloader, collate_fn=collate_fn, 
                collecting_device=self.collecting_device)

            print(f'# Block [{block_idx + 1} / {len(blocks)}]: '
                  f'[{block.sp.name} -> {block.ep.name}]')
            pre_loss, post_loss = self.finetune(
                steps=self.steps, learning_rate=self.lr, block=block, 
                qt_inputs=qt_inputs, fp_outputs=fp_outputs, executor=executor)
            print(f'# Tuning Finished  : ({pre_loss:.4f} -> {min(pre_loss, post_loss):.4f}) [Block Loss]')
            print('') # blank line


class PPLCudaAddConvReluMerge(QuantizationOptimizationPass):
    def __init__(self) -> None:
        super().__init__(name='PPL CUDA Conv(Relu) - Add - Relu Merge')

    def is_same_platform(self, operations: List[Operation]):
        platforms = [operation.platform for operation in operations]
        return all([platform == platforms[0] for platform in platforms])

    def optimize(self,
                 processor: BaseGraph,
                 dataloader: Iterable,
                 executor: BaseGraphExecutor,
                 **kwargs) -> None:

        def ep_expr(operation: Operation):
            if not isinstance(operation, QuantableOperation): return False
            if operation.type == 'Conv': return True
            if operation.type in PPLCUDA_ACTIVATIONS:
                upstream_ops = graph.get_upstream_operations(operation=operation)
                if len(upstream_ops) == 0 and upstream_ops[0].type == 'Conv': return True
                if upstream_ops[0] in merged: return True
            return False

        def retrospect(operation: QuantableOperation) -> QuantableOperation:
            if not isinstance(operation, QuantableOperation): return None
            if len(graph.get_upstream_operations(operation)) != 1: return None

            parent = graph.get_upstream_operations(operation)[0]
            if parent.type != 'Conv': return None
            if not isinstance(parent, QuantableOperation): return None
            return parent

        def merge_fn(operation: QuantableOperation):
            assert isinstance(operation, QuantableOperation) and operation.type == 'Add'
            # check if upstream ops can be merged
            up_ops = graph.get_upstream_operations(operation)
            if not self.is_same_platform(up_ops + [operation]): return

            # Conv - Add - Relu Merge
            config = operation.config.output_quantization_config[0]

            # Step - 1: merge add output to next activation.
            down_ops = graph.get_downstream_operations(operation)
            if (len(down_ops) == 1 and
                down_ops[0].type in PPLCUDA_ACTIVATIONS and
                isinstance(down_ops[0], QuantableOperation) and
                down_ops[0].platform == operation.platform):
                config.dominated_by = down_ops[0].config.output_quantization_config[0]

            # Step - 2: disable input conv's quantization(only one).
            up_ops = graph.get_upstream_operations(operation)
            assert len(up_ops) == 2, f'Opeartion {operation.name} should has exact 2 input operations.'

            target_operation = None
            for op in up_ops:
                if op.type == 'Conv':
                    target_operation = op
                elif op.type in PPLCUDA_ACTIVATIONS:
                    target_operation = retrospect(operation)
                if target_operation is not None:
                    break

            if target_operation is not None:
                target_operation.config.output_quantization_config[0].dominated_by = config

        graph, merged, unchanged = processor.graph, set(), False

        # merge conv - add iteratively, until there is no one left.
        while not unchanged:
            unchanged = True

            search_engine = SearchableGraph(processor)
            matchings = search_engine(TraversalCommand(
                sp_expr=lambda x: (x.type == 'Add' and
                                   isinstance(x, QuantableOperation) and
                                   x not in merged),
                rp_expr=lambda x, y: False,
                ep_expr=ep_expr,
                direction='up'))

            # count how many matched inputs does an add operation has.
            counter = defaultdict(lambda : 0)

            # path[0] is add operation.
            for path in matchings: counter[path[0]] += 1

            for operation, count in counter.items():
                if count == 2:
                    merge_fn(operation)
                    merged.add(operation)
                    unchanged = False
