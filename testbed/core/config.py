"""Strict recipe loading; every launch saves the fully expanded values."""
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

from .optim import optimizer_dict, resolve_schedule
from .random import derive_seed


def strict_dataclass(cls, values):
    if isinstance(values, cls):
        return values
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


@dataclass
class TrainerConfig:
    batch_size: int = 32
    grad_clip_norm: float | None = None
    lr_schedule: str | dict = "constant"
    eval_every_updates: int | None = None
    eval_splits: tuple[str, ...] = ("val", "test")
    checkpoint_every_updates: int | None = None
    probe_at_updates: tuple[int, ...] = ()
    probes: tuple[str, ...] = ()
    fresh_reference_at_updates: tuple[int, ...] = ()
    fresh_reference_updates: int | None = None
    seen_input_capacity: int = 1000

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.grad_clip_norm is not None and (self.grad_clip_norm <= 0 or not math.isfinite(self.grad_clip_norm)):
            raise ValueError("grad_clip_norm must be positive")
        for name in ("eval_every_updates", "checkpoint_every_updates", "fresh_reference_updates"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        for name in ("probe_at_updates", "fresh_reference_at_updates"):
            values = getattr(self, name)
            if list(values) != sorted(set(values)) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} must be sorted distinct nonnegative updates")
        if self.fresh_reference_at_updates and self.fresh_reference_updates is None:
            raise ValueError("fresh references require a fitting budget")
        if bool(self.probe_at_updates) != bool(self.probes):
            raise ValueError("scheduled probes require both probes and probe_at_updates")
        if type(self.seen_input_capacity) is not int or self.seen_input_capacity < 0:
            raise ValueError("seen_input_capacity cannot be negative")
        if not self.eval_splits or set(self.eval_splits) - {"val", "test"}:
            raise ValueError("eval_splits must contain val and/or test")
        self.lr_schedule = resolve_schedule(self.lr_schedule)


@dataclass
class RunConfig:
    paradigm: str
    data: dict
    model: dict = field(default_factory=lambda: {"name": "mlp"})
    method: dict = field(default_factory=lambda: {"name": "backprop"})
    optimizer: dict = field(default_factory=lambda: {"name": "adam", "lr": 0.001})
    trainer: TrainerConfig | dict = field(default_factory=TrainerConfig)
    seed: int = 0
    data_seed: int | None = None
    model_seed: int | None = None
    method_seed: int | None = None
    probe_seed: int | None = None
    data_root: str = "./data"
    output_dir: str = "./runs/example"
    device: str = "cpu"
    num_workers: int = 0

    def __post_init__(self):
        self.trainer = strict_dataclass(TrainerConfig, self.trainer) if isinstance(self.trainer, dict) else self.trainer
        self.optimizer = optimizer_dict(self.optimizer)
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        for name in ("data_seed", "model_seed", "method_seed", "probe_seed"):
            if getattr(self, name) is None:
                setattr(self, name, derive_seed(self.seed, name))
        for name in ("seed", "data_seed", "model_seed", "method_seed", "probe_seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**63:
                raise ValueError(f"{name} must be an integer in [0, 2**63)")


def set_dotted(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into {key}")
    node[parts[-1]] = value


def load_yaml(path, overrides=()):
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    for override in overrides:
        if "=" not in override:
            raise ValueError("--set uses dotted.key=value")
        key, raw = override.split("=", 1)
        set_dotted(value, key, yaml.safe_load(raw))
    return value


def plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_yaml(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(plain(value), sort_keys=False))
