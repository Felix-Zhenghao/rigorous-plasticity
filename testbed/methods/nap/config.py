import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NaPConfig:
    radius: str = "initial"
    radius_multiplier: float = 1.0
    affine_policy: str = "init_decay"
    affine_decay: float = 0.0
    eps: float = 1e-12
    parameter_scope: str | tuple[str, ...] = "network"
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, key)) for key in ('radius_multiplier', 'affine_decay', 'eps')):
            raise ValueError("method numeric parameters must be finite")
        if not self.at_updates and self.every_updates is None:
            object.__setattr__(self, "every_updates", 1)
        if self.radius != "initial" or self.radius_multiplier <= 0 or self.eps <= 0:
            raise ValueError("NaP requires initial radii, a positive multiplier and eps")
        if self.affine_policy not in {"init_decay", "joint_project", "free"} or not 0 <= self.affine_decay <= 1:
            raise ValueError("invalid NaP affine policy or decay")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = NaPConfig
