# TODO move training logic to here.

from typing import Any, Callable, List

import numpy as np
import torch
from ppq.IR.base.graph import Operation
from ppq.IR.quantize import QuantableOperation
from ppq.core import (USING_CUDA_KERNEL, ChannelwiseTensorQuantizationConfig,
                      QuantizationProperty, QuantizationStates,
                      TensorQuantizationConfig)
from ppq.core.defs import ppq_warning
from ppq.quantization.qfunction.linear import (torch_channelwise_quantize,
                                               torch_tensorwise_quantize)
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
            if not QuantizationStates.is_activated(config.state): return tensor

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
                ctx.save_for_backward(mask)
                return quantized
            else:
                raise ValueError(
                    'You are using Linear quantization function, make sure your quantization config'
                    ' either have property "PER_CHANNEL" or property "PER_TENSOR"')

        @ staticmethod
        def backward(ctx, dy: torch.Tensor):
            [mask] = ctx.saved_tensors
            gradient = dy.masked_fill(mask, 0)
            return gradient, gradient, None, None, None

    LinearQuantSieve = AdvancedLinearQuantCudaImpl().apply

def make_operation_trainable(operation: Operation, random_initial: bool) -> List[torch.Tensor]:
    """
    Change state of a opeartion to be trainable.
        Will set all operation's parameter requires_grad = True.
        Will overwrite all quantization state of parameter to be activate(from baking)
        Will create an offset tensor for each trainable parameter.

    Args:
        opeartion (Operation): [description]
        
        random_initial (bool): whether to initialize your training variable will all 0,
            or random value from -1 ~ 1

    Returns:
        A list of trainable offset tensors.
    """
    offsets = []
    for parameter in operation.parameters:
        value = parameter.value
    
        if isinstance(operation, QuantableOperation):
            for cfg, var in operation.config_with_variable:
                if not var.is_parameter: continue
                if cfg.state not in {QuantizationStates.ACTIVATED, QuantizationStates.BAKED, 
                                     QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_BAKED}:
                    raise PermissionError(f'Can not initialize training state of opeartion {operation.name}, '
                                          f'cause parameter {var.name} has not been correctly quantized.')
                
                if var.is_parameter and cfg.state == QuantizationStates.BAKED:
                    ppq_warning(f'Variable {var.name} has already been baked. '
                                'However you are requesting a training procedure to finetune it.'
                                'This optim pass will overwrite its baking state anyway.')
                    
                    var.value.copy_(var.stored_value.to(var.value.device))
                    cfg.state = QuantizationStates.ACTIVATED
                
                if var.is_parameter and cfg.state == QuantizationStates.PASSIVE_BAKED:
                    ppq_warning(f'Variable {var.name} has already been baked. '
                                'However you are requesting a training procedure to finetune it.'
                                'This optim pass will overwrite its baking state anyway.')
                    
                    var.value.copy_(var.stored_value.to(var.value.device))
                    cfg.state = QuantizationStates.PASSIVE
    
        # only float tensor can have gradient.
        if isinstance(value, torch.Tensor) and value.dtype == torch.float:
            value.requires_grad = True
            if random_initial:
                offsets.append(torch.randn_like(value, requires_grad=True))
            else:
                offsets.append(torch.zeros_like(value, requires_grad=True))

    return offsets

def make_operation_untrainable(operation: Operation, baking_function: Callable):
    """
    Change state of a opeartion to be trainable.
        Will set all operation's parameter requires_grad = False.
        Will clear parameter.grad = None

    Args:
        opeartion (Operation): [description]

    Returns:
        None
    """
    for parameter in operation.parameters:
        value = parameter.value

        # only float tensor can have gradient.
        if isinstance(value, torch.Tensor) and value.dtype == torch.float:
            value.requires_grad = False
            value._grad = None

    if isinstance(operation, QuantableOperation):
        for cfg, var in operation.config_with_variable:
            if not var.is_parameter: continue
            if cfg.state not in {QuantizationStates.ACTIVATED, QuantizationStates.BAKED, 
                                    QuantizationStates.PASSIVE, QuantizationStates.PASSIVE_BAKED}:
                raise PermissionError(f'Can not initialize training state of opeartion {operation.name}, '
                                        f'cause parameter {var.name} has not been correctly quantized.')
        operation.baking_parameters(baking_function)

class FinetuneCheckPoint:
    """
    Finetune Check Point stores training loss for variables.
        It bounds to a specific variable, collects and stores its fp32 value as a reference.
    
    ATTENTION: collecting fp32 value might cause GPU memory overflow, so we use a speckle to
        collect only a part of fp32 value instead(speckle randomly pick about 1000 values from given tensor).
    
    Finetune Check Point maintains a speckle(index) for data collecting, a best loss, and a reference values.
    """
    def __init__(self, variable: str, best_loss: float,
        speckle:List[int], references:List[torch.Tensor]) -> None:
        self.monitor_var = variable
        self.best_loss = best_loss
        self.speckle = speckle
        self.references = references

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
