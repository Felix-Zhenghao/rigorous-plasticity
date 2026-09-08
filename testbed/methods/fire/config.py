from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FIREConfig:
    """Project selected weight matrices toward scaled orthogonal matrices.

    MLP/ResNet apply FIRE to all Linear/Conv2d weights; ViT applies it only to
    attention query/key weights. Convolutions project each spatial slice.

    iterations: Positive Newton-Schulz iteration count per matrix/slice.
        More iterations improve orthogonalization at greater compute cost.
    eps: Finite positive Frobenius-norm cutoff. Matrices/slices with norm <=
        eps are left unchanged to avoid division by zero during normalization.
    optimizer_state: "keep" preserves optimizer history; "reset_all"
        discards all optimizer state at every scheduled projection.
    at_updates: Sorted, distinct positive completed-update numbers at which
        to intervene, after the optimizer step. Use () with every_updates;
        otherwise set every_updates=None and provide at least one number.
    every_updates: Intervene after every N completed optimizer updates (N > 0),
        counting from the start of the run and across resumes. None disables
        the periodic schedule; exactly one schedule must be configured.
    """

    iterations: int = 10
    eps: float = 1e-12
    optimizer_state: str = "keep"
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('eps',)):
            raise ValueError("method numeric parameters must be finite")
        if type(self.iterations) is not int or self.iterations < 1 or self.eps <= 0:
            raise ValueError("FIRE requires positive iterations and eps")
        if self.optimizer_state not in {"keep", "reset_all"}:
            raise ValueError("FIRE optimizer_state must be keep or reset_all")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = FIREConfig
