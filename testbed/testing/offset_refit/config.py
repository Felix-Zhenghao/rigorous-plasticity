import math
from dataclasses import dataclass, field

from testbed.testing.common import validate_probe


@dataclass
class OffsetProbeConfig:
    updates: int
    batch_size: int = 32
    eval_every_updates: int = 10
    target_offsets: tuple = ("same", 0)
    repeats: int = 1
    load_optimizer_state: bool = False
    optimizer: dict = field(default_factory=lambda: {"name": "adam", "lr": 0.001})
    lr_schedule: str | dict = "constant"
    grad_clip_norm: float | None = None
    fresh_reference: bool = True

    def __post_init__(self):
        validate_probe(self)
        if not self.target_offsets or any(x != "same" and (type(x) not in (float, int) or not math.isfinite(x))
                                          for x in self.target_offsets):
            raise ValueError("target_offsets must contain numbers or 'same'")
