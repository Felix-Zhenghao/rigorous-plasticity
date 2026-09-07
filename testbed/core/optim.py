"""Ordinary optimizer and update-based learning-rate construction."""
import math
from dataclasses import asdict, dataclass, is_dataclass

import torch


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False

    def __post_init__(self):
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


def optimizer_dict(config):
    config = asdict(config) if is_dataclass(config) else dict(config)
    return asdict(OptimizerConfig(**config))


def make_optimizer(named_parameters, config):
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


def resolve_schedule(schedule):
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


def learning_rate(config, schedule, completed_updates):
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


def set_learning_rate(optimizer, config, schedule, completed_updates):
    lr = learning_rate(config, schedule, completed_updates)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
