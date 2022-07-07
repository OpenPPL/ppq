import torch
from ppq.core import (CUDA, PPQ_CONFIG, QuantizationProperty,
                      QuantizationStates, TensorQuantizationConfig,
                      ppq_warning)
from ppq.IR.base.graph import Variable

from .base import BaseTensorObserver


class TorchIsotoneObserver(BaseTensorObserver):
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._cache = []
    """For softmax or sigmoid activations, usually we just need
    argmax(softmax(x)) == argmax(softmax(quant(x))) which is argmax(x) ==
    argmax(quant(x))

    Inspired by this Property, we designed an order-preserving calibration method,
        which cares only about max(x) [or min(x)]

    To keep argmax(x) == argmax(quant(x)), we only need to
        distinguish the largest element and the second largert element with quantization

        let L1 represents the largest element of x,
        while L2 represents the second largest.

        For symmetric quantization:
        We want
            quant(L1, scale) > quant(L2, scale)
            clip(round(L1 / scale)) > clip(round(L2 / scale))
        Which means:
            1. L1 - L2 > 0.5 * scale
            2. round(L2 / scale) < clip_max - 1
    Args:
        BaseTensorObserver ([type]): [description]
    """
    def observe(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), 'IsotoneObserver can only deal with torch Tensor values'
        assert value.numel() > 0, (f'You are observing an empty tensor({self._watch_on.name}).')
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                # flatten value as [batch, num_of_elements]
                value = value.flatten(start_dim=1)
                value, _ = torch.topk(value, k=2, dim=-1, largest=True, sorted=True)
                self._cache.append(value)
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                raise TypeError('IsotoneObserver is not designed for channelwise quantization.')
            else:
                raise TypeError('Min-max Observer only work with per-tensor or per-channel quantize policy.')


    def render_quantization_config(self):
        device = self._cache[-1].device
        collected = torch.cat(self._cache, dim=0)
        collected = collected.cpu().numpy()
        s_maxs, s_mins = [], []
        for l1, l2 in collected:
            scale_min = l2 / (self._quant_cfg.quant_max - 1)
            scale_max = 2 * (l1 - l2)
            s_maxs.append(scale_max)
            s_mins.append(scale_min)

        best_satisfied, best_scales = 0, []
        for s_candidate in s_maxs + s_mins:
            satisfied = 0

            for s_max, s_min in zip(s_maxs, s_mins):
                if s_candidate <= s_max: satisfied += 1
                if s_candidate >= s_min: satisfied += 1

            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_scales    = [s_candidate]
            if satisfied == best_satisfied:
                best_scales.append(s_candidate)

        best_scale = sum(best_scales) / len(best_scales)
        self._quant_cfg.scale  = torch.tensor([best_scale], dtype=torch.float32, device=device).squeeze(0)
        self._quant_cfg.offset = torch.tensor([0], dtype=torch.float32, device=device).squeeze(0)
        self._quant_cfg.state = QuantizationStates.ACTIVATED
