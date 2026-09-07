import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralConfig:
    coefficient: float
    exponent: float = 2.0
    power_iterations: int = 1
    eps: float = 1e-12
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = True
    include_embeddings: bool = True

    def __post_init__(self):
        if self.coefficient < 0 or not math.isfinite(self.coefficient):
            raise ValueError("coefficient must be finite and nonnegative")
        if self.exponent <= 0 or not math.isfinite(self.exponent):
            raise ValueError("exponent must be finite and positive")
        if type(self.power_iterations) is not int or self.power_iterations <= 0:
            raise ValueError("power_iterations must be positive")
        if self.eps <= 0 or not math.isfinite(self.eps):
            raise ValueError("eps must be finite and positive")
        if isinstance(self.parameter_scope, str):
            if self.parameter_scope not in {"network", "hidden", "head"}:
                raise ValueError("invalid parameter_scope")
        elif not self.parameter_scope or len(set(self.parameter_scope)) != len(self.parameter_scope):
            raise ValueError("parameter_scope must contain distinct module names")


CONFIG_CLASS = SpectralConfig
