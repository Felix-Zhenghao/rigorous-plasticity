import math
from dataclasses import dataclass


@dataclass(frozen=True)
class L2Config:
    coefficient: float
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = False
    include_embeddings: bool = True

    def __post_init__(self):
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        if isinstance(self.parameter_scope, str):
            if self.parameter_scope not in {"network", "hidden", "head"}:
                raise ValueError("invalid parameter_scope")
        elif not self.parameter_scope or len(set(self.parameter_scope)) != len(self.parameter_scope):
            raise ValueError("explicit parameter_scope must contain distinct module names")


CONFIG_CLASS = L2Config
