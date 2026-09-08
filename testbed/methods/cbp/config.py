from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CBPConfig:
    """Continuously replace mature units with the lowest estimated utility.

    MLP sites are all hidden/head_hidden layers; ResNet sites are the first
    convolution in each residual block; ViT sites are feedforward fc1 units.
    Replacements resample incoming weights, zero incoming biases and outgoing
    weights, and reset the unit's normalization parameters/statistics.

    replacement_rate: Fraction in [0, 1] of mature units to replace per site
        per update. Fractional counts accumulate across updates; zero disables
        replacement. Utility is mean(abs(feature)) * sum(abs(outgoing_weight)).
    ema_decay: Decay in [0, 1) for unit utility and mean-feature averages:
        average = decay * old + (1-decay) * observation. Selection corrects
        initialization bias using the unit's age; zero uses this step only.
    maturity_updates: Nonnegative minimum unit age in optimizer updates.
        A unit becomes eligible when age > maturity_updates; replacement
        resets its age to zero.
    bias_compensation: Add the removed units' estimated mean contribution
        to the downstream bias before zeroing outgoing weights. Supported
        only for MLP with pre-activation normalization, after_activation
        statistics, no dropout and existing downstream biases.
    statistic_site: "after_activation" measures activated units; "after_norm"
        measures normalization outputs, before or after the activation as
        configured by the model. ViT supports only "after_activation".
    optimizer_state: "keep" preserves optimizer history; "clear_moments"
        zeros parameter-shaped optimizer state only at modified entries,
        preserving step counters; "reset_all" discards all optimizer state.
        "reset_all" runs only when at least one unit is replaced.
    """

    replacement_rate: float = 1e-4
    ema_decay: float = 0.99
    maturity_updates: int = 20
    bias_compensation: bool = False
    statistic_site: str = "after_activation"
    optimizer_state: str = "clear_moments"

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('replacement_rate', 'ema_decay')):
            raise ValueError("method numeric parameters must be finite")
        if not 0 <= self.replacement_rate <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid CBP rate or ema_decay")
        if type(self.maturity_updates) is not int or self.maturity_updates < 0:
            raise ValueError("maturity_updates must be a nonnegative integer")
        if self.statistic_site not in {"after_activation", "after_norm"}:
            raise ValueError("unknown statistic_site")
        if self.optimizer_state not in {"keep", "clear_moments", "reset_all"}:
            raise ValueError("unknown optimizer_state")


CONFIG_CLASS = CBPConfig
