"""Portable checkpoint manifests and isolated reconstruction helpers."""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import RunConfig, plain, strict_dataclass
from .optim import optimizer_dict, set_learning_rate
from .random import rng_state, set_rng_state
from .types import Batch, Learner, ProblemSpec

if TYPE_CHECKING:
    from testbed.data.datasets import DatasetInput

    from .consumption import Consumer


def atomic_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return state


class SeenInputBank:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.inputs, self.ids = [], []
        self._membership = set()

    def add(self, batch: Batch) -> None:
        if len(self.ids) >= self.capacity:
            return
        x, _, ids = batch
        for tensor, source_id in zip(x, ids):
            source_id = int(source_id)
            if source_id not in self._membership:
                self.inputs.append(tensor.detach().cpu().clone())
                self.ids.append(source_id)
                self._membership.add(source_id)
            if len(self.ids) >= self.capacity:
                break

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "inputs": torch.stack(self.inputs) if self.inputs else None,
                "ids": torch.tensor(self.ids, dtype=torch.long)}

    def load_state_dict(self, state: dict[str, Any], prefix: int | None = None) -> None:
        self.capacity = state["capacity"]
        n = len(state["ids"]) if prefix is None else prefix
        if n > len(state["ids"]):
            raise ValueError("seen-input bank is shorter than checkpoint's valid prefix")
        self.ids = state["ids"][:n].tolist()
        self.inputs = [] if n == 0 else list(state["inputs"][:n].unbind())
        self._membership = set(self.ids)


def snapshot(
    config: RunConfig,
    learner: Learner,
    consumer: Consumer,
    bank: SeenInputBank,
    *,
    bank_path: str | Path,
    recorder_state: dict[str, Any] | None = None,
    report_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bank_state = bank.state_dict()
    if Path(bank_path).exists():
        existing = torch.load(bank_path, map_location="cpu", weights_only=False)
        prefix = min(len(existing["ids"]), len(bank.ids))
        if not torch.equal(existing["ids"][:prefix], bank_state["ids"][:prefix]):
            raise ValueError("seen-input bank disagrees with the existing run")
        if len(existing["ids"]) > len(bank.ids):
            bank_state = existing
    atomic_save(bank_state, bank_path)
    return {"format_version": 1, "config": plain(config), "problem": asdict(consumer.paradigm.problem),
            "architecture": config.model["name"],
            "model_config": deepcopy(learner.resolved_model_config),
            "method_config": deepcopy(learner.resolved_method_config),
            "initialization_seed": config.model_seed, "source_update": learner.completed_updates,
            "learner_state": deepcopy(learner.state_dict()), "consumer_state": consumer.state_dict(),
            "paradigm_state": deepcopy(consumer.paradigm.state_dict()), "run_rng": rng_state(),
            "seen_inputs": {"path": str(Path(bank_path).resolve()), "prefix": len(bank.ids)},
            "recorder_state": recorder_state, "report_state": report_state or {}}


def build_learner(config: RunConfig, problem: ProblemSpec, *, start_update: int = 0) -> Learner:
    from .factory import make_model
    learner = make_model(config.method["name"], config.model["name"], problem=problem,
                         model_config={k: v for k, v in config.model.items() if k != "name"},
                         method_config={k: v for k, v in config.method.items() if k != "name"},
                         optimizer_config=config.optimizer, lr_schedule=config.trainer.lr_schedule,
                         grad_clip_norm=config.trainer.grad_clip_norm, device=config.device,
                         seed=config.model_seed, method_seed=config.method_seed,
                         start_update=start_update)
    return learner


def restore_run(
    state: dict[str, Any],
    *,
    device: str | None = None,
    output_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    num_workers: int | None = None,
    datasets: DatasetInput = None,
) -> tuple[RunConfig, Learner, Consumer, SeenInputBank]:
    from testbed.training import make_paradigm

    from .consumption import Consumer
    cfg = deepcopy(state["config"])
    if device is not None:
        cfg["device"] = device
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    if data_root is not None:
        cfg["data_root"] = str(data_root)
    if num_workers is not None:
        cfg["num_workers"] = num_workers
    config = strict_dataclass(RunConfig, cfg)
    paradigm = make_paradigm(config.paradigm, config.data, data_root=config.data_root,
                             seed=config.data_seed, datasets=datasets)
    if paradigm.problem != ProblemSpec(**state["problem"]):
        raise ValueError("restored data changed the fixed ProblemSpec")
    learner = build_learner(config, paradigm.problem)
    learner.load_state_dict(deepcopy(state["learner_state"]))
    consumer = Consumer(paradigm, config.trainer.batch_size, num_workers=config.num_workers, seed=config.data_seed)
    consumer.load_state_dict(state["consumer_state"])
    bank = SeenInputBank(config.trainer.seen_input_capacity)
    saved = torch.load(state["seen_inputs"]["path"], map_location="cpu", weights_only=False)
    bank.load_state_dict(saved, state["seen_inputs"]["prefix"])
    set_rng_state(state["run_rng"])
    return config, learner, consumer, bank


def load_network_optimizer(learner: Learner, checkpoint: dict[str, Any]) -> None:
    """Restore only network statistics, rejecting incompatible probe settings."""
    saved_cfg = optimizer_dict(checkpoint["config"]["optimizer"])
    probe_cfg = optimizer_dict(learner.optimizer_config)
    if any(saved_cfg[k] != probe_cfg[k] for k in saved_cfg if k != "lr"):
        raise ValueError("probe optimizer must match saved type and non-LR settings")
    saved = checkpoint["learner_state"].get("optimizer")
    if saved is None:
        raise ValueError("checkpoint has no optimizer state")
    entries = {}
    for group in saved["param_groups"]:
        names = group.get("param_names")
        if names is None or len(names) != len(group["params"]):
            raise ValueError("optimizer checkpoint must contain parameter names")
        for name, index in zip(names, group["params"]):
            if name in entries:
                raise ValueError("duplicate optimizer parameter name")
            entries[name] = saved["state"].get(index, {})
    current = learner.optimizer.state_dict()
    parameters = dict(learner.network.named_parameters())
    for group in current["param_groups"]:
        for name, index in zip(group["param_names"], group["params"]):
            if name not in entries or not name.startswith("network."):
                raise ValueError(f"missing network optimizer state: {name}")
            parameter = parameters[name.removeprefix("network.")]
            stats = deepcopy(entries[name])
            for key, value in stats.items():
                if isinstance(value, torch.Tensor) and key != "step" and value.shape != parameter.shape:
                    raise ValueError(f"optimizer shape mismatch for {name}.{key}")
            current["state"][index] = stats
    learner.optimizer.load_state_dict(current)
    set_learning_rate(learner.optimizer, probe_cfg, learner.lr_schedule, 0)
