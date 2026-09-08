"""Finite, sequential suites with saved job manifests and paired probe seeds."""
from __future__ import annotations

import glob
import itertools
import json
import statistics
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from testbed.core.checkpoint import atomic_save, load_checkpoint
from testbed.core.config import load_yaml, set_dotted
from testbed.core.metrics import write_json
from testbed.core.types import ProblemSpec
from testbed.testing.common import fingerprint


def resolve_path(path: str | Path, config_dir: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() or path.exists() else Path(config_dir) / path


def _strict(recipe: dict[str, Any], allowed: Iterable[str]) -> None:
    extra = set(recipe) - set(allowed) - {"kind", "output_dir"}
    if extra:
        raise ValueError(f"unknown suite fields: {sorted(extra)}")


def checkpoint_jobs(recipe: dict[str, Any], config_dir: str | Path) -> list[dict[str, Any]]:
    _strict(recipe, {"probe_config", "checkpoints", "repeat_seeds", "device"})
    probe = load_yaml(resolve_path(recipe["probe_config"], config_dir))
    paths = set()
    for pattern in recipe["checkpoints"]:
        matches = glob.glob(str(resolve_path(pattern, config_dir)))
        if not matches:
            raise ValueError(f"checkpoint pattern matched nothing: {pattern}")
        paths.update(str(Path(p).resolve()) for p in matches)
    sources = []
    for path in paths:
        state = load_checkpoint(path)
        sources.append((str(Path(state["config"]["output_dir"]).resolve()), state["source_update"], path))
    jobs = []
    for run, update, path in sorted(sources):
        for seed in recipe.get("repeat_seeds", [0]):
            config = deepcopy(probe)
            config["seed"] = seed
            if "device" in recipe:
                config["device"] = recipe["device"]
            key = f"{Path(run).name}-{fingerprint(run)[:8]}/update_{update}-{fingerprint(path)[:8]}"
            jobs.append({"checkpoint": path, "source_run": run, "source_update": update,
                         "seed": seed, "config": config, "destination": key})
    if not jobs:
        raise ValueError("checkpoint suite contains no jobs")
    return jobs


def experiment_jobs(recipe: dict[str, Any], config_dir: str | Path) -> list[dict[str, Any]]:
    _strict(recipe, {"recipes", "grid", "seeds", "probes", "device"})
    grid = recipe.get("grid", {})
    if any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("grid values must be nonempty finite lists")
    jobs = []
    for source in recipe["recipes"]:
        path = resolve_path(source, config_dir)
        original = load_yaml(path)
        for values in itertools.product(*grid.values()):
            settings = dict(zip(grid, values))
            for seed in recipe.get("seeds", [0]):
                config = deepcopy(original)
                for key, value in settings.items():
                    set_dotted(config, key, value)
                config["seed"] = seed
                if "device" in recipe:
                    config["device"] = recipe["device"]
                scientific = {k: v for k, v in config.items() if k not in {"seed", "output_dir", "device"}}
                group = f"{path.stem}-{fingerprint(str(path.resolve()), scientific)[:8]}"
                config["output_dir"] = str(Path(recipe["output_dir"]) / group / f"seed_{seed}")
                jobs.append({"recipe": str(path), "group": group, "family": config["paradigm"],
                             "settings": settings, "seed": seed, "config": config})
    if not jobs:
        raise ValueError("experiment suite contains no jobs")
    return jobs


def aggregate(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {}
    for job in jobs:
        output = Path(job["config"]["output_dir"])
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        latest = {}
        for record in records:
            if record["kind"] == "evaluation":
                latest[record["split"]] = record
        for split, record in latest.items():
            key = (job["family"], job["group"], split)
            groups.setdefault(key, []).append(record)
    result = []
    for (family, group, split), records in groups.items():
        entry = {"family": family, "group": group, "split": split, "seeds": len(records)}
        for metric in ("loss", "accuracy", "macro_accuracy"):
            values = [r[metric] for r in records if metric in r]
            if values:
                entry[metric] = {"mean": statistics.mean(values),
                                 "std": statistics.stdev(values) if len(values) > 1 else 0.0}
        result.append(entry)
    return result


def transfer_branches(recipe: dict[str, Any], config_dir: str | Path) -> None:
    from testbed.core.trainer import train
    from testbed.training import make_paradigm
    from testbed.training.class_incremental.paradigm import retarget_state
    _strict(recipe, {"source_checkpoint", "source_recipe", "source_update", "target_sizes", "device", "probes"})
    if ("source_checkpoint" in recipe) == ("source_recipe" in recipe):
        raise ValueError("specify exactly one source_checkpoint or source_recipe")
    update = recipe["source_update"]
    if not isinstance(update, int) or update < 1:
        raise ValueError("source_update must be a positive absolute update")
    if "source_recipe" in recipe:
        config = load_yaml(resolve_path(recipe["source_recipe"], config_dir))
        config["output_dir"] = str(Path(recipe["output_dir"]) / "source")
        if "device" in recipe:
            config["device"] = recipe["device"]
        source = train(config, stop_after_updates=update)
    else:
        source = load_checkpoint(resolve_path(recipe["source_checkpoint"], config_dir))
    if source["source_update"] != update:
        raise ValueError("source checkpoint does not match source_update")
    original = source["config"]
    if original["paradigm"] != "class_incremental" or original["data"]["progression"] != "transfer":
        raise ValueError("target branches require a fixed-head transfer source recipe")
    cs = source["consumer_state"]
    if cs["paradigm_state"]["active_task"] != 0 or cs["before_block"] is None:
        raise ValueError("source_update must end source consumption before target training")
    from testbed.core.consumption import Consumer
    old = make_paradigm("class_incremental", original["data"], data_root=original["data_root"], seed=original["data_seed"])
    check = Consumer(old, original["trainer"]["batch_size"], seed=original["data_seed"])
    check.load_state_dict(cs)
    if check.cursor.start < len(check.block.data):
        raise ValueError("source_update must complete the source data budget")
    jobs = []
    for size in recipe["target_sizes"]:
        state = deepcopy(source)
        cfg = state["config"]
        cfg["data"]["stage_sizes"][1] = size
        cfg["output_dir"] = str(Path(recipe["output_dir"]) / f"target_{size}")
        if "device" in recipe:
            cfg["device"] = recipe["device"]
        target = make_paradigm("class_incremental", cfg["data"], data_root=cfg["data_root"], seed=cfg["data_seed"])
        if as_problem(target.problem) != state["problem"]:
            raise ValueError("target branch changed the saved ProblemSpec")
        state["paradigm_state"] = retarget_state(state["paradigm_state"], target)
        for key in ("paradigm_state", "before_block"):
            state["consumer_state"][key] = retarget_state(state["consumer_state"][key], target)
        jobs.append({"target_size": size, "output_dir": cfg["output_dir"], "source_update": update})
        destination = Path(cfg["output_dir"])
        atomic_save(state, destination / "source.pt")
    write_json(Path(recipe["output_dir"]) / "jobs.json", jobs)
    for job in jobs:
        result = train(resume=Path(job["output_dir"]) / "source.pt")
        _probes(recipe.get("probes", []), config_dir, result, Path(job["output_dir"]) / "probes")


def as_problem(problem: ProblemSpec) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(problem)


def _probes(
    paths: Sequence[str | Path],
    config_dir: str | Path,
    checkpoint: dict[str, Any],
    output: str | Path,
) -> None:
    from testbed.cli import run_test
    for path in paths:
        run_test(load_yaml(resolve_path(path, config_dir)), checkpoint, output)


def run_suite(recipe: dict[str, Any], *, config_dir: str | Path = ".") -> str:
    """Run one sequential suite; only fields for its kind are accepted.

    kind: "experiments" (default), "checkpoint_probes", or "transfer_branches".
    output_dir: Required root for the job manifest and all generated run results.
    device: Optional override for training recipes in experiments/transfer_branches,
        or probe recipes in checkpoint_probes. Post-training probes in the other
        suite kinds use their own recipe's device (default "cpu").

    experiments fields:
        recipes: Required list of training YAML paths.
        grid: Dotted config keys mapped to nonempty lists of candidate values;
            every Cartesian-product combination is run. Omit for no overrides.
        seeds: Independent master training seeds per combination; default [0].
            Explicit sub-seeds inside a recipe stay fixed across these runs.
        probes: Probe YAML paths to run on each final checkpoint; default [].
            Their own seeds apply, or the source checkpoint's probe_seed if absent.

    checkpoint_probes fields:
        probe_config: Required path to one probe YAML recipe.
        checkpoints: Required list of checkpoint paths/globs, e.g. update_*.pt.
            Every pattern must match; duplicate paths are evaluated only once.
        repeat_seeds: Probe seeds used for each checkpoint; default [0].
            These replace the recipe seed; probe.repeats still runs within each seed.

    transfer_branches fields:
        source_recipe: Training YAML with progression="transfer"; run through
            source_update. Set exactly one of this and source_checkpoint.
        source_checkpoint: Existing transfer checkpoint at source_update.
        source_update: Required positive absolute step index immediately after
            the complete source budget, before any target training.
        target_sizes: Required list of target pool sizes: integer counts,
            fractions in (0, 1], or "all". Each branch inherits the same complete
            learner state and replaces only the target stage's available pool.
        probes: Probe YAML paths evaluated after each target branch; default [].

    Recipe/checkpoint paths resolve from the working directory when they exist
    there, otherwise relative to config_dir. output_dir is relative to the working
    directory. Existing final experiment checkpoints are reused.
    """
    from testbed.cli import run_test
    from testbed.core.trainer import train
    output = Path(recipe["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    kind = recipe.get("kind", "experiments")
    if kind == "checkpoint_probes":
        jobs = checkpoint_jobs(recipe, config_dir)
        write_json(output / "jobs.json", jobs)
        for job in jobs:
            run_test(job["config"], job["checkpoint"], output / job["destination"])
    elif kind == "experiments":
        jobs = experiment_jobs(recipe, config_dir)
        write_json(output / "jobs.json", jobs)
        for job in jobs:
            final = Path(job["config"]["output_dir"]) / "checkpoints" / "final.pt"
            result = load_checkpoint(final) if final.exists() else train(job["config"])
            _probes(recipe.get("probes", []), config_dir, result, Path(job["config"]["output_dir"]) / "probes")
        write_json(output / "summary.json", aggregate(jobs))
    elif kind == "transfer_branches":
        transfer_branches(recipe, config_dir)
    else:
        raise ValueError(f"unknown suite kind: {kind}")
    return str(output)
