import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureNormConfig:
    coefficient: float
    feature_sites: tuple[str, ...] = ("head_input",)
    reduction: str = "mean"

    def __post_init__(self):
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        if isinstance(self.feature_sites, str) or not self.feature_sites or len(set(self.feature_sites)) != len(self.feature_sites):
            raise ValueError("feature_sites must contain distinct site names")
        if self.reduction not in {"mean", "sum_per_example"}:
            raise ValueError("invalid feature reduction")


CONFIG_CLASS = FeatureNormConfig
