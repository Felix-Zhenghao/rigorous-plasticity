"""One isolated fitting path for standalone, scheduled, and suite probes."""
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from testbed.core.checkpoint import atomic_save, load_checkpoint, load_network_optimizer
from testbed.core.config import plain
from testbed.core.consumption import Consumer
from testbed.core.metrics import JsonlWriter, evaluate, write_json
from testbed.core.optim import optimizer_dict, resolve_schedule
from testbed.core.random import RNGStream, isolated_rng
from testbed.core.types import Consumption, DataBlock, ProblemSpec


@dataclass
class ProbeResult:
    output_dir: str
    source_update: int
    comparisons: list[dict]


def validate_probe(config):
    for key in ("updates", "batch_size", "eval_every_updates", "repeats"):
        if type(getattr(config, key)) is not int or getattr(config, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.grad_clip_norm is not None and (config.grad_clip_norm <= 0 or not math.isfinite(config.grad_clip_norm)):
        raise ValueError("grad_clip_norm must be positive")
    config.optimizer = optimizer_dict(config.optimizer)
    config.lr_schedule = resolve_schedule(config.lr_schedule)


def source_state(checkpoint):
    return load_checkpoint(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint


def fingerprint(*values):
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"{")
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
            digest.update(b"}")
        elif isinstance(value, (list, tuple)):
            digest.update(b"[")
            for item in value:
                update(item)
            digest.update(b"]")
        else:
            digest.update(json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode())
    for value in values:
        update(value)
    return digest.hexdigest()[:20]


class FixedFitting:
    def __init__(self, problem, data, updates):
        self.problem, self.data, self.updates = problem, data, updates
        self.used = False

    def get_data(self):
        if self.used:
            return None
        self.used = True
        return DataBlock(self.data, Consumption(len(self.data), epochs=None, updates=self.updates))

    def get_eval_data(self, split):
        return self.data

    def state_dict(self):
        return {"used": self.used}

    def load_state_dict(self, state):
        self.used = state["used"]


def probe_network(checkpoint, config, *, fresh, device, seed):
    from testbed.core.factory import make_network
    from testbed.methods.backprop.learner import BackpropLearner
    problem = ProblemSpec(**checkpoint["problem"])
    network = make_network(checkpoint["architecture"], problem=problem,
                           model_config=checkpoint["model_config"], device=device,
                           seed=checkpoint["initialization_seed"])
    if not fresh:
        network.load_state_dict(checkpoint["learner_state"]["network"], strict=True)
    learner = BackpropLearner(network=network, problem=problem, optimizer_config=config.optimizer,
                              lr_schedule=config.lr_schedule, grad_clip_norm=config.grad_clip_norm,
                              start_update=0)
    learner.rng = RNGStream(seed, device)
    if config.load_optimizer_state and not fresh:
        load_network_optimizer(learner, checkpoint)
    return learner


def fit_branch(checkpoint, config, dataset, *, fresh, device, seed, writer, assay):
    problem = ProblemSpec(**checkpoint["problem"])
    learner = probe_network(checkpoint, config, fresh=fresh, device=device, seed=seed)
    consumer = Consumer(FixedFitting(problem, dataset, config.updates), config.batch_size, seed=seed)
    curve = []

    def report():
        scores = evaluate(learner, dataset, problem, batch_size=config.batch_size)
        record = {"source_update": checkpoint["source_update"], "probe_update": learner.completed_updates,
                  "branch": "fresh" if fresh else "aged", "assay_fingerprint": assay,
                  "load_optimizer_state": config.load_optimizer_state and not fresh, **scores}
        writer.write(record)
        curve.append(record)

    report()
    for batch in consumer:
        learner.train_step(batch)
        consumer.commit()
        if learner.completed_updates % config.eval_every_updates == 0 or learner.completed_updates == config.updates:
            report()
    area = sum((b["probe_update"] - a["probe_update"]) * (a["loss"] + b["loss"]) / 2
               for a, b in zip(curve, curve[1:]))
    return {"initial": curve[0], "terminal": curve[-1], "loss_auc": area / config.updates}


def fit_pair(checkpoint, config, inputs, targets, ids, *, output_dir, seed, device, metadata):
    problem = ProblemSpec(**checkpoint["problem"])
    assay = fingerprint(inputs, targets, ids, checkpoint["model_config"], checkpoint["problem"], plain(config), seed)
    source = fingerprint(checkpoint["config"]["output_dir"], checkpoint["source_update"],
                         checkpoint["learner_state"]["network"],
                         checkpoint["learner_state"]["optimizer"] if config.load_optimizer_state else None)
    destination = Path(output_dir) / f"update_{checkpoint['source_update']}_{source}" / assay
    if (destination / "result.json").exists():
        return json.loads((destination / "result.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    artifact = {"inputs": inputs, "targets": targets, "example_ids": ids,
                "input_target_fingerprint": fingerprint(inputs, targets, ids), **metadata}
    atomic_save(artifact, destination / "fitting_data.pt")
    write_json(destination / "manifest.json", {"assay_fingerprint": assay,
               "source_update": checkpoint["source_update"], "problem": checkpoint["problem"],
               "architecture": checkpoint["architecture"], "model_config": checkpoint["model_config"],
               "probe": plain(config), "seed": seed, **metadata})
    dataset = TensorDataset(inputs, targets, ids)
    summaries = {}
    for fresh in ([False, True] if config.fresh_reference else [False]):
        label = "fresh" if fresh else "aged"
        path = destination / label / "metrics.jsonl"
        if path.exists():
            path.unlink()  # Only an incomplete job can reach this point.
        summaries[label] = fit_branch(checkpoint, config, dataset, fresh=fresh, device=device, seed=seed,
                                      writer=JsonlWriter(path), assay=assay)
    result = {"assay_fingerprint": assay, "source_update": checkpoint["source_update"],
              "source_fingerprint": source,
              "updates": config.updates, "branches": summaries, **metadata}
    if config.fresh_reference:
        result["loss_gap"] = summaries["aged"]["terminal"]["loss"] - summaries["fresh"]["terminal"]["loss"]
        if problem.loss_kind == "cross_entropy":
            result["accuracy_gap"] = summaries["fresh"]["terminal"]["accuracy"] - summaries["aged"]["terminal"]["accuracy"]
    write_json(destination / "result.json", result)
    return result


def unseen_inputs(checkpoint, config, *, seed, data_root=None, datasets=None):
    from testbed.training import make_paradigm
    run = checkpoint["config"]
    paradigm = make_paradigm(run["paradigm"], run["data"], data_root=data_root or run["data_root"],
                             seed=run["data_seed"], datasets=datasets)
    paradigm.load_state_dict(deepcopy(checkpoint["paradigm_state"]))
    dataset = paradigm.get_unseen_data(config.split)
    if len(dataset) < config.n_samples:
        raise ValueError(f"unseen input split has {len(dataset)} examples; need {config.n_samples}")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:config.n_samples]
    with isolated_rng(seed):
        samples = [dataset[int(i)] for i in indices]
    return torch.stack([x for x, _, _ in samples]), torch.tensor([int(i) for _, _, i in samples])


def seen_inputs(checkpoint, n_samples, *, seed):
    saved = checkpoint["seen_inputs"]
    bank = torch.load(saved["path"], map_location="cpu", weights_only=False)
    count = saved["prefix"]
    if count < n_samples or len(bank["ids"]) < count:
        raise ValueError(f"checkpoint has {count} valid seen inputs; need {n_samples}")
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:n_samples]
    return bank["inputs"][indices].clone(), bank["ids"][indices].clone()
