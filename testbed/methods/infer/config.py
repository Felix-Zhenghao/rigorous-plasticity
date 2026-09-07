import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InFeRConfig:
    coefficient: float
    num_heads: int
    target_scale: float
    head_reduction: str = "sum"

    def __post_init__(self):
        if self.coefficient < 0 or not math.isfinite(self.coefficient):
            raise ValueError("coefficient must be finite and nonnegative")
        if type(self.num_heads) is not int or self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not math.isfinite(self.target_scale):
            raise ValueError("target_scale must be finite")
        if self.head_reduction not in {"sum", "mean"}:
            raise ValueError("head_reduction must be sum or mean")


CONFIG_CLASS = InFeRConfig
