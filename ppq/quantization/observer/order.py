import torch

from ppq.core import (OBSERVER_ISOTONE_OBSERVER_AXIS, QuantizationProperty,
                      QuantizationStates, TensorQuantizationConfig,
                      ppq_warning)
from ppq.IR.base.graph import Variable

from .base import BaseTensorObserver
from .range import minmax_to_scale_offset


class TorchIsotoneObserver(BaseTensorObserver):
    def __init__(self, watch_on: Variable, quant_cfg: TensorQuantizationConfig):
        super().__init__(watch_on, quant_cfg)
        self._cache = []
        if OBSERVER_ISOTONE_OBSERVER_AXIS not in quant_cfg.detail:
            ppq_warning('Initializing Torch Isotone Observer with implicit axis is not recommended.')
            self.axis = -1
        else: self.axis = quant_cfg.detail[OBSERVER_ISOTONE_OBSERVER_AXIS]

    """For softmax or sigmoid activations, usually we just need
    argmax(softmax(x)) == argmax(softmax(quant(x)))

    Inspired by this Property, Isotone Observer is designed to provide an order-preserving calibration method,
        which cares only about argmax(x) [or argmin(x)]

    To keep argmax(x) == argmax(quant(x)), we only need to
        distinguish the largest element and the second largert element with quantization

        let L1 represents the largest element of x,
        while L2 represents the second largest.

        For symmetric quantization:
        We want
            1. L1 - L2 > scale
            2. round(L2 / scale) <= (quant_max - .5)
    Args:
        BaseTensorObserver ([type]): [description]
    """
    def observe(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor), 'IsotoneObserver can only deal with torch Tensor values'
        assert value.numel() > 0, (f'You are observing an empty tensor({self._watch_on.name}).')
        if self._quant_cfg.state == QuantizationStates.INITIAL:
            if self._quant_cfg.policy.has_property(QuantizationProperty.PER_TENSOR):
                # flatten value as [-1, num_of_elements in isotone axis]
                if value.ndim > 1:
                    value = value.transpose(dim0=self.axis, dim1=-1)
                    value = value.flatten(start_dim=0, end_dim=-2)
                value, _ = torch.topk(value, k=2, dim=self.axis, largest=True, sorted=True)
                if value.ndim <= 1: value = value.unsqueeze(0)
                self._cache.append(value)
            elif self._quant_cfg.policy.has_property(QuantizationProperty.PER_CHANNEL):
                raise TypeError('Isotone Observer is not designed for channelwise quantization.')
            else:
                raise TypeError('Isotone Observer only work with per-tensor or per-channel quantize policy.')

    def render_quantization_config(self):
        device = self._cache[-1].device
        collected = torch.cat(self._cache, dim=0)
        collected = collected.cpu().numpy()
        s_candidates = []

        for l1, l2 in collected:
            if self._quant_cfg.policy.has_property(QuantizationProperty.SYMMETRICAL):
                l1, l2 = abs(l1), abs(l2)

            scale_min = max(l2 / (self._quant_cfg.quant_max - .51), 0)
            scale_max = 2 * (l1 - max(l2, 0))
            if scale_max > scale_min and l1 > 0:
                s_candidates.append((scale_min, 'min'))
                s_candidates.append((scale_max, 'max'))

        if len(s_candidates) <= 0:
            # fall back to min-max calibration
            scale, offset = minmax_to_scale_offset(min_val=0, max_val=l1, config=self._quant_cfg)
            self._quant_cfg.scale  = torch.tensor([scale], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.offset = torch.tensor([offset], dtype=torch.float32, device=device).squeeze(0)
            self._quant_cfg.state  = QuantizationStates.ACTIVATED
            ppq_warning(
                f'There is no way to classify variable {self._watch_on.name} under int8 quantization.\n'
                f'变量 {self._watch_on.name} 无法进行保序校准，在校准数据集上无法保证分类正确性，请检查数据。')
            return
    
        s_candidates, best_satisfied, satisfied = sorted(s_candidates), 0, 0
        for s_candidate, T in s_candidates:

            if T == 'min': satisfied += 1
            if T == 'max': satisfied -= 1

            if satisfied > best_satisfied:
                best_satisfied = satisfied
                best_scale     = s_candidate

        self._quant_cfg.scale  = torch.tensor([best_scale], dtype=torch.float32, device=device).squeeze(0)
        self._quant_cfg.offset = torch.tensor([0], dtype=torch.float32, device=device).squeeze(0)
        self._quant_cfg.state  = QuantizationStates.ACTIVATED
        self.s_candidates = s_candidates
