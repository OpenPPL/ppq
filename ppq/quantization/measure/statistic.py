import torch

def torch_KL_divergence(hist: torch.Tensor, ref_hist: torch.Tensor, eps=1e-30) -> float:
    if hist.ndim != 1 or ref_hist.ndim != 1: raise ValueError(
        'Only 1 dimension tensor can compute KL divergence with another tensor. '\
        f'While your input hist has dimension {hist.ndim} and ref_hist has dimension {ref_hist.ndim}')
    if len(hist) != len(ref_hist): raise ValueError(
        'Can not compute KL divergence, len(hist) != len(ref_hist')

    # here we compute KL divergence at float64 precision, make sure your hist and ref_hist are stored at cpu.
    # otherwise it might be very slow.
    return torch.dot(hist.double(), torch.log10(hist.double() + eps) - torch.log10(ref_hist.double() + eps)).item()
