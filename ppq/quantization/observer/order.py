from .base import BaseTensorObserver
import torch


class OrderPreservingObserver(BaseTensorObserver):
    """
    For softmax or sigmoid activations, usually we just need
        argmax(softmax(x)) == argmax(softmax(quant(x)))
        which is argmax(x) == argmax(quant(x))
    
    Inspired by this Property, we designed a order preserving calibration method,
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
            2. round(L2 / scale) < clip_max
    Args:
        BaseTensorObserver ([type]): [description]
    """
    def observe(self, value: torch.Tensor):
        return super().observe(value)

    def render_quantization_config(self):
        return super().render_quantization_config()