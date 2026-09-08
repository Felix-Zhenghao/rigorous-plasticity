from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReDoConfig:
    """Recycle units whose activity is small relative to their layer average.

    MLP sites are all hidden/head_hidden layers; ResNet sites are the first
    convolution in each residual block; ViT sites are feedforward fc1 units.
    Replacements zero outgoing weights and reset normalization parameters
    and the selected units' BatchNorm running statistics.

    threshold: Finite nonnegative cutoff: recycle when mean(abs(feature)) /
        (layer_mean_activity + eps) <= threshold. A larger value replaces
        more units; zero selects only exactly inactive units.
    reset_source: "fresh_init" resamples incoming weights using their original
        initialization rule and zeros incoming biases. "saved_init" restores
        each selected unit's saved initial weights and biases.
    statistics_window_updates: Positive number of recent training batches
        used to estimate activity, weighted by the number of feature values.
        One uses the latest batch; the rolling window continues across resets.
    statistic_site: "after_activation" measures activated units; "after_norm"
        measures normalization outputs, before or after activation according
        to model placement. ViT supports only "after_activation".
    optimizer_state: "keep" preserves optimizer history; "clear_moments"
        zeros parameter-shaped optimizer state only at modified entries,
        preserving step counters; "reset_all" discards all optimizer state.
        "reset_all" runs only when at least one unit is replaced.
    eps: Finite positive offset in the layer-mean activity denominator.
    at_updates: Sorted, distinct positive completed-update numbers at which
        to intervene, after the optimizer step. Use () with every_updates;
        otherwise set every_updates=None and provide at least one number.
    every_updates: Intervene after every N completed optimizer updates (N > 0),
        counting from the start of the run and across resumes. None disables
        the periodic schedule; exactly one schedule must be configured.
    """

    threshold: float = 0.1
    reset_source: str = "fresh_init"
    statistics_window_updates: int = 1
    statistic_site: str = "after_activation"
    optimizer_state: str = "clear_moments"
    eps: float = 1e-9
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('threshold', 'eps')):
            raise ValueError("method numeric parameters must be finite")
        if self.threshold < 0 or self.eps <= 0:
            raise ValueError("threshold must be nonnegative and eps positive")
        if self.reset_source not in {"fresh_init", "saved_init"}:
            raise ValueError("unknown reset_source")
        if type(self.statistics_window_updates) is not int or self.statistics_window_updates < 1:
            raise ValueError("statistics_window_updates must be positive")
        if self.statistic_site not in {"after_activation", "after_norm"}:
            raise ValueError("unknown statistic_site")
        if self.optimizer_state not in {"keep", "clear_moments", "reset_all"}:
            raise ValueError("unknown optimizer_state")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = ReDoConfig
