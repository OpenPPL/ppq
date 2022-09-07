from decimal import ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from math import ceil, floor, log2
from typing import Union

import torch
from ppq.core import RoundingPolicy
from torch.autograd import Function

class PPQTensorRoundImpl(Function):
    @ staticmethod
    def forward(ctx, value: torch.Tensor, 
                policy:RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) -> torch.Tensor:
        """
            reference: https://en.wikipedia.org/wiki/Rounding

        Args:
            value (torch.Tensor): [description]
            policy (RoundingPolicy, optional): [description]. Defaults to RoundingPolicy.ROUND_HALF_EVEN.

        Raises:
            ValueError: [description]

        Returns:
            torch.Tensor: [description]
        """
        assert isinstance(value, torch.Tensor), 'tensor round only takes effect on torch tensor.'
        if policy == RoundingPolicy.ROUND_HALF_EVEN:
            # default rounding policy of torch is ROUND_TO_NEAR_EVEN
            # try this: print(torch.Tensor([1.5, 2.5, 3.5, 4.5]).round())
            # However it may generate unexpected results due to version difference.
            return value.round()
        elif policy == RoundingPolicy.ROUND_UP:
            return value.ceil()
        elif policy == RoundingPolicy.ROUND_HALF_TOWARDS_ZERO:
            return torch.sign(value) * torch.ceil(value.abs() - 0.5)
        elif policy == RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO:
            return torch.sign(value) * torch.floor(value.abs() + 0.5)
        elif policy == RoundingPolicy.ROUND_HALF_DOWN:
            return torch.ceil(value - 0.5)
        elif policy == RoundingPolicy.ROUND_HALF_UP:
            return torch.floor(value + 0.5)
        elif policy == RoundingPolicy.ROUND_TO_NEAR_INT:
            raise NotImplementedError(f'Torch Tensor can not use this rounding policy({policy}) try ROUND_HALF_EVEN instead.')
        else:
            raise ValueError('Unexpected rounding policy found.')

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None

def ppq_numerical_round(value: float,
    policy: RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) -> int:
    """
        reference: https://en.wikipedia.org/wiki/Rounding

        decimal definition:
            - decimal.ROUND_CEILING (towards Infinity)
            - decimal.ROUND_DOWN (towards zero)
            - decimal.ROUND_FLOOR (towards -Infinity)
            - decimal.ROUND_HALF_DOWN (to nearest with ties going towards zero)
            - decimal.ROUND_HALF_EVEN (to nearest with ties going to nearest even integer)
            - decimal.ROUND_HALF_UP (to nearest with ties going away from zero)
            - decimal.ROUND_UP (away from zero)
            - decimal.ROUND_05UP (away from zero if last digit after rounding towards zero would have been 0 or 5; otherwise towards zero)

    Args:
        value (float): [description]
        policy (RoundingPolicy, optional): [description]. Defaults to RoundingPolicy.ROUND_HALF_EVEN.

    Raises:
        ValueError: [description]

    Returns:
        int: [description]
    """
    assert isinstance(value, float), 'numerical round only takes effect on float number.'
    if policy == RoundingPolicy.ROUND_HALF_EVEN:
        return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_EVEN))
    elif policy == RoundingPolicy.ROUND_HALF_UP:
        if value > 0: return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_UP))
        else: return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_DOWN))
    elif policy == RoundingPolicy.ROUND_HALF_DOWN:
        if value > 0: return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_DOWN))
        else: return int(Decimal(value).quantize(exp=Decimal(1), rounding=ROUND_HALF_UP))
    elif policy == RoundingPolicy.ROUND_HALF_TOWARDS_ZERO:
        return ppq_numerical_round(value, RoundingPolicy.ROUND_HALF_DOWN)
    elif policy == RoundingPolicy.ROUND_HALF_FAR_FORM_ZERO:
        return ppq_numerical_round(value, RoundingPolicy.ROUND_HALF_UP)
    elif policy == RoundingPolicy.ROUND_TO_NEAR_INT:
        if value > 0: return floor(value + 0.5)
        else: return ceil(value - 0.5)
    elif policy == RoundingPolicy.ROUND_UP:
        return ceil(value)
    else:
        raise ValueError('Unexpected rounding policy found.')

def ppq_tensor_round(
    value: torch.Tensor,
    policy:RoundingPolicy=RoundingPolicy.ROUND_HALF_EVEN) -> torch.Tensor:
    """
        reference: https://en.wikipedia.org/wiki/Rounding

    Args:
        value (torch.Tensor): [description]
        policy (RoundingPolicy, optional): [description]. Defaults to RoundingPolicy.ROUND_HALF_EVEN.

    Raises:
        ValueError: [description]

    Returns:
        torch.Tensor: [description]
    """
    return PPQTensorRoundImpl.apply(value, policy)

def ppq_round_to_power_of_2(value: Union[float, int],
    policy: RoundingPolicy=RoundingPolicy.ROUND_UP) -> float:
    if value == 0: return 0
    sign = 1 if value >= 0 else -1
    assert isinstance(value, float) or isinstance(value, int), \
        'power-of-2 round only takes effect on float or int.'
    return sign * float(pow(2, ppq_numerical_round(log2(sign * value), policy=policy)))
