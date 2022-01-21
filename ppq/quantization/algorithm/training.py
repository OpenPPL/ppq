# TODO move training logic to here.

import random
from random import randint
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
from ppq.core import (NUM_OF_CHECKPOINT_FETCHS, USING_CUDA_KERNEL,
                      ChannelwiseTensorQuantizationConfig, NetworkFramework,
                      QuantizationProperty, QuantizationStates,
                      TensorQuantizationConfig)
from ppq.IR import BaseGraph, Operation, Variable
from ppq.quantization.qfunction.linear import (torch_channelwise_quantize,
                                               torch_tensorwise_quantize)
from ppq.utils.fetch import batch_random_fetch
from torch.autograd import Function


def torch_channelwise_quantize_sieve(
    offset: torch.Tensor, tensor: torch.Tensor, limit: float,
    config: ChannelwiseTensorQuantizationConfig, threshold: float) -> List[torch.Tensor]:
    assert 0 <= threshold <= 1, f'Invalid threshold value {threshold}, suppose value between 0 ~ 1'
    shape = [1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)]
    
    offset = torch.clip(offset, min=-limit, max=limit)
    quantized = torch_channelwise_quantize(tensor + offset * config.scale.view(shape), config)
    quant_mask = torch.abs(tensor - quantized) < ((1 - threshold) * config.scale.view(shape))
    return quantized, quant_mask

def torch_tensorwise_quantize_sieve(
    offset: torch.Tensor, tensor: torch.Tensor, limit: float,
    config: TensorQuantizationConfig, threshold: float) -> List[torch.Tensor]:
    assert 0 <= threshold <= 1, f'Invalid threshold value {threshold}, suppose value between 0 ~ 1'
    
    offset = torch.clip(offset, min=-limit, max=limit)
    quantized = torch_tensorwise_quantize(tensor + offset * config.scale, config)
    quant_mask = torch.abs(tensor - quantized) < ((1 - threshold) * config.scale)
    return quantized, quant_mask

if not USING_CUDA_KERNEL:
    class AdvancedLinearQuantTorchImpl(Function):
        @ staticmethod
        def forward(ctx, offset: torch.Tensor, tensor: torch.Tensor, limit: float,
                    config: TensorQuantizationConfig, threshold: float) -> Any:
            if not QuantizationStates.is_activated(config.state): return tensor
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                quantized, mask = torch_channelwise_quantize_sieve(
                    offset=offset, tensor=tensor, limit=limit, 
                    config=config, threshold=threshold)
                ctx.save_for_backward(mask)
                return quantized

            elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
                quantized, mask = torch_tensorwise_quantize_sieve(
                    offset=offset, tensor=tensor, limit=limit, 
                    config=config, threshold=threshold)
                ctx.save_for_backward(mask)
                return quantized

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            [mask] = ctx.saved_tensors
            gradient = dy.masked_fill(mask, 0)
            return gradient, gradient, None, None, None

    LinearQuantSieve = AdvancedLinearQuantTorchImpl().apply

else: # if USING_CUDA_KERNEL:
    from ppq.core import CUDA
    class AdvancedLinearQuantCudaImpl(Function):
        @ staticmethod
        def forward(ctx, offset: torch.Tensor, tensor: torch.Tensor, limit: float,
                    config: TensorQuantizationConfig, threshold: float) -> Any:
            # if not QuantizationStates.is_activated(config.state): return tensor

            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                assert isinstance(config, ChannelwiseTensorQuantizationConfig)
                quantized, mask = CUDA.ChannelwiseLinearQuantSieve(
                    tensor=tensor,
                    fp_offset=offset,
                    scales=config.scale,
                    offsets=config.offset,
                    channel_axis=config.channel_axis,
                    minimum=config.quant_min,
                    maximum=config.quant_max,
                    rounding=config.rounding.value,
                    limit=limit,
                    threshold=threshold
                )
                ctx.config = config # 这是强行植入的属性！
                ctx.save_for_backward(mask)
                return quantized

            elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
                assert isinstance(config, TensorQuantizationConfig)
                quantized, mask = CUDA.TensorwiseLinearQuantSieve(
                    tensor=tensor,
                    fp_offset=offset,
                    scale=config.scale,
                    offset=config.offset,
                    minimum=config.quant_min,
                    maximum=config.quant_max,
                    rounding=config.rounding.value,
                    limit=limit,
                    threshold=threshold
                )
                ctx.config = config # 这是强行植入的属性！
                ctx.save_for_backward(mask)
                return quantized
            else:
                raise ValueError(
                    'You are using Linear quantization function, make sure your quantization config'
                    ' either have property "PER_CHANNEL" or property "PER_TENSOR"')

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            [mask], config = ctx.saved_tensors, ctx.config
            gradient = dy.masked_fill(mask, 0)
            
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                shape = [1 if axis != config.channel_axis else -1 for axis in range(mask.ndim)]
                scale = config.scale.view(shape)
            if config.policy.has_property(QuantizationProperty.PER_TENSOR):
                scale = config.scale

            return gradient / scale, gradient, None, None, None

    LinearQuantSieve = AdvancedLinearQuantCudaImpl().apply


class FinetuneCheckPoint:
    """
    Finetune Check Point stores training loss for variables.
        It bounds to a specific variable, collects and stores its fp32 value as a reference.
    
    ATTENTION: collecting fp32 value might cause GPU memory overflow, so we use a seed to
        collect only a part of fp32 value instead(randomly pick about 2000 values from given tensor).
    
    Finetune Check Point maintains a seed for data collecting, a best loss, and a reference values.
    """
    def __init__(self, variable: str, random_fetch: bool = True, seed: int=None, fetchs: int=NUM_OF_CHECKPOINT_FETCHS) -> None:
        if seed is None: seed = randint(0, 0xffffffff)
        self.monitor_var = variable
        self.best_loss   = float(1e9)
        self.seed        = seed
        self.references  = []
        self.outputs     = []
        self.fetchs      = fetchs
        self.random_fetch = random_fetch
    
    def push(self, tensor: torch.Tensor, is_reference: bool) -> None:
        if self.random_fetch:
            tensor = batch_random_fetch(tensor, seed=self.seed, fetchs_per_batch=self.fetchs)
        if is_reference: self.references.append(tensor)
        else: self.outputs.append(tensor)

    def pop(self) -> Tuple[torch.Tensor]:
        assert len(self.outputs) == len(self.references), ('Inconsistent samples detected.'
            f'Reference output gets {len(self.references)} samples, however output has {len(self.outputs)}.')
        
        return self.outputs, self.references
    
    def clear(self):
        self.outputs.clear()


class TrainableDelegate:
    def __init__(
        self, value: torch.Tensor, config: TensorQuantizationConfig, 
        limit: float, boost: float, binding: Variable) -> None:
        """
        Helper class helps you create trainable variable.

        Args:
            value (torch.Tensor): [description]
            config (TensorQuantizationConfig): [description]
            limit (float): [description]
            boost (float): [description]
            binding (Variable): [description]

        Raises:
            PermissionError: [description]
        """
        
        if config.state in {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}:
            raise PermissionError(f'Can not create TrainableDelegate with variable {binding.name},'
                                  ' cause its value has been baked.')

        self.raw       = value
        self.config    = config
        self.limit     = limit
        self.boost     = boost
        self.binding   = binding
        self.offset    = torch.zeros_like(value)

        self._state_backup = config.state
        self.raw.requires_grad    = True
        self.offset.requires_grad = True
        self.config.state         = QuantizationStates.DEACTIVATED

    def quantize(self, threshold: float):
        self.binding.value = LinearQuantSieve(
            self.offset * self.boost, self.raw, self.limit, 
            self.config, threshold)

    def withdraw(self):
        self.binding.value = self.raw

    def clear(self):
        self.raw.requires_grad    = False
        self.offset.requires_grad = False
        self.config.state         = self._state_backup
        self.raw._grad            = None
        self.offset._grad         = None
        self.offset               = None


class RandomMemDataset:
    """
        A very little helper class for randomly pick data samples from your dataset.
    """
    def __init__(self, data: Iterable) -> None:
        self._data = data
        self._num_of_batchs = len(data)

    def pop(self):
        idx = random.randint(0, self._num_of_batchs - 1)
        return self._data[idx]


class TrainableSubgraph(BaseGraph):
    def __init__(self,
                 inputs: List[Variable],
                 outputs: List[Variable],
                 operations:List[Operation]) -> None:
        super().__init__(name='PPQ Trainable SubGraph', built_from=NetworkFramework.NATIVE)
        

class TimeDecay:
    """
        A helper class computing time decay.
    """
    def __init__(self, t_max: int, decay: float=0.2, beta_start: float=20, beta_end:float=2):
        self.t_max = t_max
        self.start_decay = decay * t_max
        self.start_b = beta_start
        self.end_b = beta_end

    def __call__(self, t):
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))

class AdaroundRegTerm(torch.nn.Module):
    """
        Adaround Reg Term is a part of Adaround optimization algorithm.
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

    def rectified_sigmoid(self, round_mask):
        return ((self.zeta - self.gamma) * torch.sigmoid(round_mask) + self.gamma).clamp(0, 1)

    def forward(self, round_mask, iter):
        if iter < self.max_iter * self.warm_ratio:
            round_loss = 0
        else:
            self.beta = self.temp_anneal(iter)
            round_loss = self.alpha * (1 - torch.pow((self.rectified_sigmoid(round_mask) - 0.5).abs() * 2, self.beta)).sum()
        return round_loss

def Lp_norm(pred: torch.tensor, tgt: torch.tensor, p: float = 2.0):
    return (pred - tgt).abs().pow(p).sum(1).mean()
