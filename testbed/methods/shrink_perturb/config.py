import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShrinkPerturbConfig:
    retain: float = 0.999
    noise_scale: float = 0.001
    noise_source: str = "fresh_init"
    optimizer_state: str = "keep"
    reset_running_stats: bool = False
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = True
    include_embeddings: bool = True
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, key)) for key in ('retain', 'noise_scale')):
            raise ValueError("method numeric parameters must be finite")
        if not 0 <= self.retain <= 1 or self.noise_scale < 0:
            raise ValueError("retain must lie in [0, 1] and noise_scale must be nonnegative")
        if self.noise_source not in {"fresh_init", "gaussian", "saved_init"}:
            raise ValueError("unknown noise_source")
        if self.optimizer_state not in {"keep", "clear_moments", "reset_all"}:
            raise ValueError("unknown optimizer_state")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = ShrinkPerturbConfig
