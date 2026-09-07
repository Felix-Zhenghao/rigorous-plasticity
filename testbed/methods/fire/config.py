import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FIREConfig:
    iterations: int = 10
    eps: float = 1e-12
    optimizer_state: str = "keep"
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self):
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
