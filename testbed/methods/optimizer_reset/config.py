from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OptResetConfig:
    """Wrap a base learner and clear its optimizer history on a schedule.

    Resets happen after the base learner's step and maintenance; learned
    parameters, learning-rate progress and the base method's own state remain.

    at_updates: Sorted, distinct positive completed-update numbers at which
        to intervene, after the optimizer step. Use () with every_updates;
        otherwise set every_updates=None and provide at least one number.
    every_updates: Intervene after every N completed optimizer updates (N > 0),
        counting from the start of the run and across resumes. None disables
        the periodic schedule; exactly one schedule must be configured.
    base_method: Method to wrap on the chosen MLP/ResNet architecture:
        "backprop", "c_chain", "cbp", "feature_norm", "fire", "infer", "l2",
        "l2_init", "layer_norm", "leaky_relu", "nap", "redo", "shrink_perturb",
        "spectral" or "swr". Another "optimizer_reset" wrapper is forbidden.
    base_config: Dictionary of fields for base_method's configuration class;
        {} uses its defaults (required fields must still be supplied). The
        base method cannot also set optimizer_state="reset_all".
    """

    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None
    base_method: str = "backprop"
    base_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")
        if self.base_method == "optimizer_reset":
            raise ValueError("optimizer reset wrappers cannot be nested")
        if self.base_config.get("optimizer_state") == "reset_all":
            raise ValueError("a base method cannot also reset all optimizer state")


CONFIG_CLASS = OptResetConfig
