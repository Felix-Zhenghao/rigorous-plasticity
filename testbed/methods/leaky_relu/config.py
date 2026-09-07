import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LeakyReLUConfig:
    negative_slope: float = 0.01

    def __post_init__(self):
        if self.negative_slope < 0 or not math.isfinite(self.negative_slope):
            raise ValueError("negative_slope must be finite and nonnegative")


CONFIG_CLASS = LeakyReLUConfig
