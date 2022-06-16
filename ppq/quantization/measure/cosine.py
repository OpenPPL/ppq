import torch
from numpy import dot, ndarray
from numpy.linalg import norm


def torch_cosine_similarity(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute mse loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1).float()
    y_real = y_real.flatten(start_dim=1).float()

    cosine_sim = torch.cosine_similarity(y_pred, y_real, dim=-1)

    if reduction == 'mean':
        return torch.mean(cosine_sim)
    elif reduction == 'sum':
        return torch.sum(cosine_sim)
    elif reduction == 'none':
        return cosine_sim
    else:
        raise ValueError(f'Unsupported reduction method.')


def numpy_cosine_similarity(
    x: ndarray, y: ndarray) -> ndarray:
    return dot(x, y) / (norm(x) * norm(y))


def torch_cosine_similarity_as_loss(
    y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    return 1 - torch_cosine_similarity(y_pred=y_pred, y_real=y_real, reduction=reduction)
