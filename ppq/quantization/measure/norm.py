import torch

def torch_mean_square_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute mse loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)
    
    diff = torch.pow(y_pred - y_real, 2).flatten(start_dim=1)
    mse  = torch.mean(diff, dim=-1)

    if reduction == 'mean':
        return torch.mean(mse)
    elif reduction == 'sum':
        return torch.sum(mse)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f'Unsupported reduction method.')

def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
            f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)
        
    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)
    # y_diff = torch.sum(torch.pow(y_pred - y_real, 2), dim=-1)
    # y_power = torch.sum(torch.pow(y_real, 2), dim=-1)
    
    noise_power  = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = noise_power / signal_power

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')
