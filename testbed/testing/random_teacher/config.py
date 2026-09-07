from dataclasses import dataclass, field

from testbed.testing.common import validate_probe


@dataclass
class TeacherProbeConfig:
    updates: int
    batch_size: int = 32
    eval_every_updates: int = 10
    input_source: str = "unseen"
    n_samples: int = 1000
    split: str = "test"
    repeats: int = 1
    load_optimizer_state: bool = False
    optimizer: dict = field(default_factory=lambda: {"name": "adam", "lr": 0.001})
    lr_schedule: str | dict = "constant"
    grad_clip_norm: float | None = None
    fresh_reference: bool = True

    def __post_init__(self):
        validate_probe(self)
        if self.input_source not in {"seen", "unseen"} or self.split not in {"val", "test"}:
            raise ValueError("invalid probe input_source or split")
        if not isinstance(self.n_samples, int) or self.n_samples < 1:
            raise ValueError("n_samples must be positive")
