import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CChainConfig:
    buffer_size: int
    reference_batch_size: int
    coefficient: float = 0.1
    reference_lag_updates: int = 1
    distance: str = "auto"
    coefficient_mode: str = "fixed"
    target_loss_ratio: float = 0.01
    loss_window: int = 1000
    adapt_after_updates: int = 1000
    eps: float = 1e-8

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, key)) for key in ('coefficient', 'target_loss_ratio', 'eps')):
            raise ValueError("method numeric parameters must be finite")
        for key in ("buffer_size", "reference_batch_size", "reference_lag_updates", "loss_window"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.adapt_after_updates) is not int or self.adapt_after_updates < 0:
            raise ValueError("adapt_after_updates must be a nonnegative integer")
        if self.coefficient < 0 or self.target_loss_ratio < 0 or self.eps <= 0:
            raise ValueError("invalid C-CHAIN coefficient, ratio or eps")
        if self.distance not in {"auto", "prediction_ce", "logit_mse"}:
            raise ValueError("invalid reference distance")
        if self.coefficient_mode not in {"fixed", "loss_ratio"}:
            raise ValueError("invalid coefficient_mode")


CONFIG_CLASS = CChainConfig
