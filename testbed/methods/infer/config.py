from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InFeRConfig:
    """Match auxiliary predictions to a frozen copy of their initial mapping.

    coefficient: Finite nonnegative lambda multiplying auxiliary squared
        error; the supervised prediction loss is unchanged. Zero disables
        auxiliary-loss gradients.
    num_heads: Positive number of scalar auxiliary outputs from one linear
        layer applied to the features entering the final output layer.
    target_scale: Finite multiplier on frozen initial auxiliary predictions.
        One preserves their original scale; zero targets zero; negative
        values reverse their sign.
    head_reduction: "sum" sums squared errors across auxiliary outputs then
        averages examples; "mean" averages outputs and examples together.
        At fixed coefficient, "sum" is num_heads times stronger than "mean".
    """

    coefficient: float
    num_heads: int
    target_scale: float
    head_reduction: str = "sum"

    def __post_init__(self) -> None:
        if self.coefficient < 0 or not math.isfinite(self.coefficient):
            raise ValueError("coefficient must be finite and nonnegative")
        if type(self.num_heads) is not int or self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not math.isfinite(self.target_scale):
            raise ValueError("target_scale must be finite")
        if self.head_reduction not in {"sum", "mean"}:
            raise ValueError("head_reduction must be sum or mean")


CONFIG_CLASS = InFeRConfig
