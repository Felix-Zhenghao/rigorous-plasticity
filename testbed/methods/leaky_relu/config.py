from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LeakyReLUConfig:
    """Use LeakyReLU activations in an MLP or ResNet.

    negative_slope: Finite nonnegative multiplier a for negative inputs:
        f(x)=x when x>=0, otherwise a*x. Zero gives ReLU. Used when the model
        omits negative_slope; an explicit model value takes precedence over
        the default 0.01. A nondefault method value must match an explicit
        model value, or construction fails.
    """

    negative_slope: float = 0.01

    def __post_init__(self) -> None:
        if self.negative_slope < 0 or not math.isfinite(self.negative_slope):
            raise ValueError("negative_slope must be finite and nonnegative")


CONFIG_CLASS = LeakyReLUConfig
