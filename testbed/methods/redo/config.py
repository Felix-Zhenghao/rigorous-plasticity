import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReDoConfig:
    threshold: float = 0.1
    reset_source: str = "fresh_init"
    statistics_window_updates: int = 1
    statistic_site: str = "after_activation"
    optimizer_state: str = "clear_moments"
    eps: float = 1e-9
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self):
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
