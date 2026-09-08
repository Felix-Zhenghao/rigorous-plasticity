from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureNormConfig:
    """Penalize squared intermediate activations, summed across selected sites.

    coefficient: Finite nonnegative multiplier of the total feature penalty.
        Zero removes the penalty's contribution to gradients.
    feature_sites: Nonempty, distinct feature names. MLP supports "hidden.i",
        "head_hidden.i" and "head_input"; indices are zero-based, and hidden
        sites are after the full hidden layer, including dropout. ResNet
        supports "stem" (after pooling), "stages.s.b.activation1",
        "stages.s.b.activation2", "pooled" (spatial mean), "head_hidden.i"
        and "head_input". Only existing indices are allowed. "head_input"
        is the input to the final output layer. Selecting overlapping sites
        charges each selected site's penalty separately.
    reduction: "mean" averages squared values over every tensor element;
        "sum_per_example" sums squared non-batch elements then averages over
        examples, so larger feature tensors receive a larger penalty.
    """

    coefficient: float
    feature_sites: tuple[str, ...] = ("head_input",)
    reduction: str = "mean"

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        if isinstance(self.feature_sites, str) or not self.feature_sites or len(set(self.feature_sites)) != len(self.feature_sites):
            raise ValueError("feature_sites must contain distinct site names")
        if self.reduction not in {"mean", "sum_per_example"}:
            raise ValueError("invalid feature reduction")


CONFIG_CLASS = FeatureNormConfig
