"""Strict recipe loading; every launch saves the fully expanded values."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .optim import optimizer_dict, resolve_schedule
from .random import derive_seed
from .types import DataclassInstance

ConfigT = TypeVar("ConfigT", bound=DataclassInstance)


def strict_dataclass(cls: type[ConfigT], values: ConfigT | dict[str, Any]) -> ConfigT:
    if isinstance(values, cls):
        return values
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


@dataclass
class TrainerConfig:
    """Training/reporting settings; all update indices count completed optimizer steps.

    batch_size: Maximum examples per optimizer step; a short chunk/pass yields
        a smaller batch instead of borrowing examples from the next one.
    grad_clip_norm: Maximum global L2 gradient norm before each optimizer step;
        None disables clipping. Includes any trainable auxiliary parameters.
    lr_schedule: "constant", or a mapping with name="constant"/"linear"/
        "cosine"/"step". See core.optim.resolve_schedule for every schedule key.
    eval_every_updates: Period between held-out and current-chunk evaluations;
        None leaves only initialization and completed-run final evaluation.
    eval_splits: Any nonempty selection of "val" and "test". Empty datasets are
        skipped; creating a validation split requires data.validation_fraction>0.
    checkpoint_every_updates: Save every N completed steps; None disables
        periodic saves. Initialization, stop, final, and probe saves still occur.
    probe_at_updates: Sorted, distinct step indices (0 allowed) at which to run
        every recipe in probes. Requires probes when nonempty.
    probes: Paths to random_teacher/offset_refit YAML recipes for scheduled
        assays. Missing recipe seed/device values inherit this run's settings.
    fresh_reference_at_updates: Sorted, distinct indices for paired aged/fresh
        fits on the next training updates; 0 is allowed. Primary state is preserved.
    fresh_reference_updates: Number of future training steps in each paired
        window; required when references are scheduled. The stream must have
        this many steps remaining, or the reference raises an error.
    seen_input_capacity: Maximum distinct source inputs cached for "seen" probes;
        keeps their first observed tensors. 0 disables storage, and seen probes
        fail if their requested sample count exceeds the saved bank.
    """

    batch_size: int = 32
    grad_clip_norm: float | None = None
    lr_schedule: str | dict[str, Any] = "constant"
    eval_every_updates: int | None = None
    eval_splits: tuple[str, ...] = ("val", "test")
    checkpoint_every_updates: int | None = None
    probe_at_updates: tuple[int, ...] = ()
    probes: tuple[str, ...] = ()
    fresh_reference_at_updates: tuple[int, ...] = ()
    fresh_reference_updates: int | None = None
    seen_input_capacity: int = 1000

    def __post_init__(self) -> None:
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
    """Top-level training recipe; nested mappings use their own documented configs.

    paradigm: "class_remap", "pixel_permutation", or "class_incremental";
        chooses the data transformation/availability rules and data config.
    data: Fields of that paradigm's config in testbed/training/<paradigm>/config.py.
    model: name="mlp"/"resnet_18"/"vit" plus fields from models/config.py;
        optional preset supplies defaults which explicit fields override.
    method: name from core.factory.METHOD_ARCHITECTURES plus its fields from
        methods/<name>/config.py. The selected architecture must be supported.
    optimizer: OptimizerConfig fields; weight_decay is independent of any
        method penalty, so enabling both applies both effects.
    trainer: TrainerConfig fields controlling minibatches, LR, reporting, and assays.
    seed: Master seed in [0, 2**63); seeds global RNGs and derives omitted sub-seeds.
    data_seed: Data membership, transformations, and consumption-order seed;
        None deterministically derives a separate stream from seed.
    model_seed: Network initialization seed; None derives it from seed.
    method_seed: Learner stochasticity/maintenance seed; None derives it from seed.
    probe_seed: Default seed for scheduled probe recipes; None derives it from seed.
    data_root: Dataset cache directory; real dataset downloads go here by default.
    output_dir: Destination for expanded config, metrics, checkpoints, and artifacts.
        A directory with training metrics requires resume or a new destination.
    device: PyTorch device, e.g. "cpu" or "cuda:0"; must exist in this environment.
    num_workers: DataLoader subprocess count; 0 loads in the main process.
        Logical augmentation seeds are preserved across worker counts.
    """

    paradigm: str
    data: dict[str, Any]
    model: dict[str, Any] = field(default_factory=lambda: {"name": "mlp"})
    method: dict[str, Any] = field(default_factory=lambda: {"name": "backprop"})
    optimizer: dict[str, Any] = field(default_factory=lambda: {"name": "adam", "lr": 0.001})
    trainer: TrainerConfig | dict[str, Any] = field(default_factory=TrainerConfig)
    seed: int = 0
    data_seed: int | None = None
    model_seed: int | None = None
    method_seed: int | None = None
    probe_seed: int | None = None
    data_root: str = "./data"
    output_dir: str = "./runs/example"
    device: str = "cpu"
    num_workers: int = 0

    def __post_init__(self) -> None:
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


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into {key}")
    node[parts[-1]] = value


def load_yaml(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
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


def plain(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_yaml(path: str | Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(plain(value), sort_keys=False))
