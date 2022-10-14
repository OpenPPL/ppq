import torch
from ppq.core import (QuantizationProperty, QuantizationStates,
                      TensorQuantizationConfig)
from ppq.IR import Variable

from .base import BaseTensorObserver


class TemporaryFloatingObserver(BaseTensorObserver):
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
