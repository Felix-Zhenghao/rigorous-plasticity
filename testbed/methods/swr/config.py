import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SWRConfig:
    utility: str = "magnitude"
    selection: str = "fraction"
    fraction: float = 0.001
    threshold: float = 0.01
    replacement: str = "init_sample"
    ema_decay: float = 0.0
    optimizer_state: str = "keep"
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = True
    include_embeddings: bool = True
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, key)) for key in ('fraction', 'threshold', 'ema_decay')):
            raise ValueError("method numeric parameters must be finite")
        if self.utility not in {"magnitude", "gradient"} or self.selection not in {"fraction", "threshold"}:
            raise ValueError("invalid SWR utility or selection")
        if self.replacement not in {"init_sample", "init_mean"}:
            raise ValueError("invalid SWR replacement")
        if not 0 <= self.fraction <= 1 or self.threshold < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid SWR fraction, threshold, or ema_decay")
        if self.optimizer_state not in {"keep", "clear_moments", "reset_all"}:
            raise ValueError("unknown optimizer_state")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = SWRConfig
