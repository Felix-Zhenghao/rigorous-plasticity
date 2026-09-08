"""Ordinary optimizer and update-based learning-rate construction."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OptimizerConfig:
    """PyTorch optimizer settings; unused optimizer-specific options must stay default.

    name: "sgd", "adam", or "adamw". AdamW decouples weight decay from the
        gradient; SGD and Adam add weight_decay * parameter to the gradient.
    lr: Nonnegative base learning rate; trainer.lr_schedule scales/replaces it.
    momentum: SGD velocity decay in [0, 1); 0 disables momentum.
    dampening: SGD fraction of the new gradient omitted from momentum updates,
        in [0, 1]. Has no effect without momentum; Nesterov requires 0.
    nesterov: Use SGD Nesterov momentum; requires momentum>0 and dampening=0.
    betas: Adam/AdamW decay rates for first and second moments, each in [0, 1).
        Larger values retain gradient history longer.
    eps: Positive Adam/AdamW denominator stabilizer, added after the square root.
    weight_decay: Nonnegative decay coefficient, applied to every optimized
        parameter, including biases, normalization, and auxiliary parameters.
        It adds to any configured maintenance/regularization method.
    amsgrad: Adam/AdamW variant using a running maximum of second moments.
    """

    name: str = "adam"
    lr: float = 0.001
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "betas", tuple(self.betas))
        if self.name not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unknown optimizer: {self.name}")
        if (not all(math.isfinite(v) for v in (self.lr, self.weight_decay, self.eps))
                or self.lr < 0 or self.weight_decay < 0 or self.eps <= 0):
            raise ValueError("invalid optimizer rate, decay, or epsilon")
        if not 0 <= self.momentum < 1 or not 0 <= self.dampening <= 1:
            raise ValueError("invalid momentum or dampening")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must be in [0, 1)")
        if self.name != "sgd" and (self.momentum or self.dampening or self.nesterov):
            raise ValueError("SGD options cannot be used with Adam")
        if self.name == "sgd" and (tuple(self.betas) != (0.9, 0.999) or self.eps != 1e-8 or self.amsgrad):
            raise ValueError("Adam options cannot be used with SGD")
        if self.nesterov and (self.momentum <= 0 or self.dampening != 0):
            raise ValueError("Nesterov requires momentum and zero dampening")


def optimizer_dict(config: OptimizerConfig | Mapping[str, Any]) -> dict[str, Any]:
    config = asdict(config) if is_dataclass(config) else dict(config)
    return asdict(OptimizerConfig(**config))


def make_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    config: OptimizerConfig | Mapping[str, Any],
) -> torch.optim.Optimizer:
    cfg = optimizer_dict(config)
    pairs = [(n, p) for n, p in named_parameters if p.requires_grad]
    if len({id(p) for _, p in pairs}) != len(pairs):
        raise ValueError("optimizer parameters must be registered exactly once")
    groups = [{"params": [p for _, p in pairs], "param_names": [n for n, _ in pairs]}]
    common = {k: cfg[k] for k in ("lr", "weight_decay")}
    if cfg["name"] == "sgd":
        return torch.optim.SGD(groups, **common, **{k: cfg[k] for k in ("momentum", "dampening", "nesterov")})
    cls = torch.optim.Adam if cfg["name"] == "adam" else torch.optim.AdamW
    return cls(groups, **common, **{k: cfg[k] for k in ("betas", "eps", "amsgrad")})


def resolve_schedule(schedule: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate an update-based LR mapping; a string supplies only its name.

    name: "constant" keeps optimizer.lr; "linear"/"cosine" interpolate from
        optimizer.lr to terminal_lr; "step" multiplies it by gamma per milestone.
    warmup_updates: Nonnegative number W of initial updates using
        optimizer.lr * (t + 1) / W, where t counts already completed updates.
        0 disables warmup. Warmup takes precedence over the named schedule.
    horizon: Required for linear/cosine; absolute completed-update index H at
        which terminal_lr is reached. H must exceed warmup_updates. Between
        warmup W and H, progress is (t - W)/(H - W); afterwards LR stays terminal.
    terminal_lr: Nonnegative final LR for linear/cosine; defaults to 0.
    milestones: Required for step; sorted, distinct positive completed-update
        indices. A milestone M affects the step taken after M completed updates.
        Milestones reached during warmup are counted once warmup ends.
    gamma: Step multiplier in (0, 1], default 0.1; after k reached milestones,
        LR is optimizer.lr * gamma**k. 1 leaves the LR unchanged.

    horizon/terminal_lr are rejected for constant/step; milestones/gamma are
    rejected for other schedules. None or an empty mapping means constant.
    """
    cfg = {"name": schedule} if isinstance(schedule, str) else dict(schedule or {"name": "constant"})
    allowed = {"name", "horizon", "terminal_lr", "warmup_updates", "milestones", "gamma"}
    if set(cfg) - allowed:
        raise ValueError(f"unknown LR schedule fields: {sorted(set(cfg) - allowed)}")
    name = cfg.setdefault("name", "constant")
    if name not in {"constant", "linear", "cosine", "step"}:
        raise ValueError(f"unknown LR schedule: {name}")
    cfg.setdefault("warmup_updates", 0)
    if not isinstance(cfg["warmup_updates"], int) or cfg["warmup_updates"] < 0:
        raise ValueError("warmup_updates must be nonnegative")
    if name in {"linear", "cosine"}:
        if type(cfg.get("horizon")) is not int or cfg["horizon"] <= cfg["warmup_updates"]:
            raise ValueError("finite schedules require horizon > warmup_updates")
        if cfg.setdefault("terminal_lr", 0.0) < 0 or not math.isfinite(cfg["terminal_lr"]):
            raise ValueError("terminal_lr must be nonnegative")
    elif "horizon" in cfg or "terminal_lr" in cfg:
        raise ValueError("horizon and terminal_lr apply only to linear/cosine schedules")
    if name == "step":
        points = cfg.get("milestones", [])
        if not points or points != sorted(set(points)) or any(not isinstance(p, int) or p < 1 for p in points):
            raise ValueError("step schedule needs sorted distinct positive milestones")
        if not 0 < cfg.setdefault("gamma", 0.1) <= 1:
            raise ValueError("step gamma must lie in (0, 1]")
    elif "milestones" in cfg or "gamma" in cfg:
        raise ValueError("milestones/gamma apply only to step schedules")
    return cfg


def learning_rate(
    config: OptimizerConfig | Mapping[str, Any],
    schedule: str | Mapping[str, Any] | None,
    completed_updates: int,
) -> float:
    base = config.lr if is_dataclass(config) else config["lr"]
    cfg = resolve_schedule(schedule)
    t = completed_updates
    warmup = cfg["warmup_updates"]
    if t < warmup:
        return base * (t + 1) / warmup
    if cfg["name"] == "constant":
        return base
    if cfg["name"] == "step":
        return base * cfg["gamma"] ** sum(t >= s for s in cfg["milestones"])
    fraction = min(max((t - warmup) / (cfg["horizon"] - warmup), 0), 1)
    factor = 1 - fraction if cfg["name"] == "linear" else (1 + math.cos(math.pi * fraction)) / 2
    return cfg["terminal_lr"] + (base - cfg["terminal_lr"]) * factor


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig | Mapping[str, Any],
    schedule: str | Mapping[str, Any] | None,
    completed_updates: int,
) -> float:
    lr = learning_rate(config, schedule, completed_updates)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
