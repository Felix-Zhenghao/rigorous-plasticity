import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CBPConfig:
    replacement_rate: float = 1e-4
    ema_decay: float = 0.99
    maturity_updates: int = 20
    bias_compensation: bool = False
    statistic_site: str = "after_activation"
    optimizer_state: str = "clear_moments"

    def __post_init__(self):
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
