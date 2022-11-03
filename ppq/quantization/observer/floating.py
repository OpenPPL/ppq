import torch
from ppq.core import (QuantizationProperty, QuantizationStates,
                      TensorQuantizationConfig, OBSERVER_FLOATING_MSE_FETCHES)
from ppq.IR import Variable

from .base import BaseTensorObserver
from ppq.utils.fetch import channel_random_fetch, tensor_random_fetch
from ppq.quantization.qfunction import PPQuantFunction


class ConstantObserver(BaseTensorObserver):
    """
    This observer will directly return scale = 1, offset = 0
    """
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._value_shape = None
        self._value_device = None

    @ torch.no_grad()
    def observe(self, value: torch.Tensor):
        self._value_shape  = value.shape # Do nothing here.
        self._value_device = value.device

    def render_quantization_config(self):
        device = self._value_device
        if self._quant_cfg.policy.has_property(QuantizationProperty.FLOATING):
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                self._quant_cfg.scale  = torch.tensor([1.0], dtype=torch.float32, device=device).squeeze(0)
                self._quant_cfg.offset = torch.tensor([0.0], dtype=torch.float32, device=device).squeeze(0)
                self._quant_cfg.state = QuantizationStates.ACTIVATED

            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                num_of_elements = self._value_shape[self._quant_cfg.channel_axis]
                scales  = [1.0 for _ in range(num_of_elements)]
                offsets = [0.0 for _ in range(num_of_elements)]

                self._quant_cfg.scale  = torch.tensor(scales, dtype=torch.float32, device=device)
                self._quant_cfg.offset = torch.tensor(offsets, dtype=torch.float32, device=device)
                self._quant_cfg.state = QuantizationStates.ACTIVATED
        else:
            raise TypeError('This Observer is designed for floating quantization.')


class DirectMSEObserver(BaseTensorObserver):
    """
    Direct MSE Observer, this observer compute MSE Loss directly.
    In PPQ there is another version of MSE Observer called TorchMSEObserver,
        which uses histogram for accelerating the computation of MSE Loss.

    We prefer to implements a direct MSE Observer for floating quantization,
        cause floating quantization can have only a few candidates of scale, which
        greatly reduce the computation complexity of solving MSE Loss.
    """
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        
        if not quant_cfg.policy.has_property(QuantizationProperty.FLOATING):
            raise TypeError('MSE Floating Observer is designed for floating quantization.')
        
        if not quant_cfg.policy.has_property(QuantizationProperty.POWER_OF_2):
            raise TypeError('MSE Floating Observer is designed for power-of-2 quantization.')

        self._value_shape  = None
        self._value_device = None
        self._collector    = []
        self._fetches      = OBSERVER_FLOATING_MSE_FETCHES

    @ torch.no_grad()
    def observe(self, value: torch.Tensor):
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
            if self._watch_on.is_parameter:
                value = torch.transpose(value, dim0=0, dim1=self._quant_cfg.channel_axis)
                self._collector.append(torch.flatten(value, start_dim=1))

            else:
                self._collector.append(channel_random_fetch(value, fetchs_per_channel=self._fetches))
                self._value_device = value.device

        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            self._collector.append(tensor_random_fetch(value, num_of_fetches=self._fetches))

    def render_quantization_config(self):
        device = self._value_device
        scale_candidates = [.0078125, .03125, .125, 1.0, 4.0, 16.0, 64.0]
        
        if not self._collector:
            raise PermissionError('Observer collector is empty, you should invoke observe function'
                                  ' before render quantization config.')

        self._quant_cfg.state = QuantizationStates.ACTIVATED
        if self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):

            self._collector = torch.cat(self._collector, dim=0)
            num_of_channel = self._collector.shape[0]
            scales  = torch.ones(size=[num_of_channel], dtype=torch.float32, device=device)
            offsets = torch.zeros(size=[num_of_channel], dtype=torch.float32, device=device)
            
            losses = []
            for scale in scale_candidates:
                self._quant_cfg._scale  = scales * scale
                self._quant_cfg._offset = offsets
                
                qt = PPQuantFunction(self._collector, self._quant_cfg)
                fp = self._collector
                loss = torch.mean(torch.square(qt - fp), dim=-1, keepdim=True)
                losses.append(loss)
            
            scale_index = torch.argmin(torch.cat(losses, dim=-1), dim=-1)
            best_scales = []
            for index in scale_index.cpu():
                best_scales.append(scale_candidates[index])
            self._quant_cfg._scale = torch.tensor(best_scales).to(self._collector.device)

        elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
            self._collector = torch.cat(self._collector, dim=0)
            scales  = torch.ones(size=[1], dtype=torch.float32, device=device)
            offsets = torch.zeros(size=[1], dtype=torch.float32, device=device)

            losses = []
            for scale in scale_candidates:
                self._quant_cfg._scale  = scales * scale
                self._quant_cfg._offset = offsets
                
                qt = PPQuantFunction(self._collector, self._quant_cfg)
                fp = self._collector
                loss = torch.mean(torch.square(qt - fp))

                losses.append((loss.item(), scale))

            best_scale = sorted(losses)[0][1]
            self._quant_cfg._scale = scales * best_scale