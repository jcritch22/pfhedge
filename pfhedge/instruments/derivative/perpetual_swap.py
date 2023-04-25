from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.str import _format_float

from ..primary.base import BasePrimary
from .base import BaseDerivative
from .base import PerpMixin
from pfhedge.nn.functional import perp_binary_payoff


class PerpetualSwap(BaseDerivative, PerpMixin):
    r"""Perpetual Swap.

    Args:
        underlier (:class:`BasePrimary`): The underlying instrument of the perpetual swap.
        funding_rate (float, default=0.0): The funding rate of the perpetual swap.
        premium_index (float, default=0.0): The premium index of the perpetual swap.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:
        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import PerpetualSwap
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = PerpetualSwap(BrownianStock(), funding_rate=0.001, premium_index=0.01)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> derivative.payoff()
        tensor([ 0.0117, -0.0061])
    """

    def __init__(
        self,
        underlier: BasePrimary,
        call: bool = True,
        strike: float = 1.0,
        funding_rate: float = 0.3,
        premium_index: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.call = call
        self.strike = strike
        self.funding_rate = funding_rate
        self.premium_index = premium_index

        if dtype is not None or device is not None:
            self.to(dtype=dtype, device=device)
            raise DeprecationWarning(
                "Specifying device and dtype when constructing a Derivative is deprecated."
                "Specify them in the constructor of the underlier instead."
            )

    def extra_repr(self) -> str:
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        params.append("funding_rate=" + _format_float(self.funding_rate))
        params.append("premium_index=" + _format_float(self.premium_index))
        return ", ".join(params)

    def payoff_fn(self) -> Tensor:
        return perp_binary_payoff(
            self.ul().spot, call=self.call, strike=self.strike)