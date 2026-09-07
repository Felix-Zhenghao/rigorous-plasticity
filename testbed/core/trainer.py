"""Shared orchestration; only learners perform optimization or maintenance."""
import hashlib
import importlib.metadata
import json
import platform
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .checkpoint import SeenInputBank, atomic_save, build_learner, load_checkpoint, restore_run, snapshot
from .config import RunConfig, load_yaml, plain, save_yaml, strict_dataclass
from .consumption import Consumer
from .metrics import Recorder, Score, evaluate, write_json
from .random import isolated_rng, seed_all


def _matches_saved(given, saved):
    if isinstance(given, dict) and isinstance(saved, dict):
        return all(key in saved and _matches_saved(value, saved[key]) for key, value in given.items())
    return plain(given) == plain(saved)


def training_window(config, learner, consumer, updates, *, output_path, datasets=None):
    from testbed.training import make_paradigm
    start = learner.completed_updates
    with isolated_rng():
        values = {}
        for name in ("aged", "fresh"):
            model = build_learner(config, consumer.paradigm.problem, start_update=start)
            if name == "aged":
                model.load_state_dict(deepcopy(learner.state_dict()))
            paradigm = make_paradigm(config.paradigm, config.data, data_root=config.data_root,
                                     seed=config.data_seed, datasets=datasets)
            branch = Consumer(paradigm, config.trainer.batch_size, num_workers=config.num_workers, seed=config.data_seed)
            branch.load_state_dict(consumer.state_dict())
            score = Score(paradigm.problem)
            for _ in range(updates):
                try:
                    batch = next(branch)
                except StopIteration as exc:
                    raise ValueError(f"fresh reference at update {start} needs {updates} future updates") from exc
                result = model.train_step(batch)
                score.add(result.predictions, batch[1], result.supervised_loss)
                branch.commit()
            values[name] = score.result()
            del model, branch
        result = {"kind": "fresh_reference", "completed_updates": start, "window_updates": updates,
                  "aged_lifetime_updates": start + updates, "scores": values,
                  "loss_gap": values["aged"]["loss"] - values["fresh"]["loss"]}
        if paradigm.problem.loss_kind == "cross_entropy":
            result["accuracy_gap"] = values["fresh"]["accuracy"] - values["aged"]["accuracy"]
        write_json(output_path, result)
        return result


def train(config=None, *, resume=None, datasets=None, stop_after_updates=None):
    from testbed.training import make_paradigm
    if resume is not None:
        saved = load_checkpoint(resume) if isinstance(resume, (str, Path)) else resume
        overrides = plain(config) if config is not None else {}
        if isinstance(overrides.get("trainer", {}).get("lr_schedule"), str):
            from .optim import resolve_schedule
            overrides["trainer"]["lr_schedule"] = resolve_schedule(overrides["trainer"]["lr_schedule"])
        config, learner, consumer, bank = restore_run(saved, device=overrides.get("device"),
                                                    output_dir=overrides.get("output_dir"),
                                                    data_root=overrides.get("data_root"),
                                                    num_workers=overrides.get("num_workers"), datasets=datasets)
        if overrides:
            permitted = {"device", "output_dir", "data_root", "num_workers"}
            if any(k not in permitted and not _matches_saved(v, saved["config"].get(k)) for k, v in overrides.items()):
                raise ValueError("resume preserves scientific configuration; use a declared branch for data changes")
        report_state = deepcopy(saved.get("report_state", {}))
    else:
        config = strict_dataclass(RunConfig, config) if isinstance(config, dict) else config
        if not isinstance(config, RunConfig):
            raise ValueError("train requires a run configuration")
        seed_all(config.seed)
        paradigm = make_paradigm(config.paradigm, config.data, data_root=config.data_root,
                                 seed=config.data_seed, datasets=datasets)
        learner = build_learner(config, paradigm.problem)
        consumer = Consumer(paradigm, config.trainer.batch_size, num_workers=config.num_workers, seed=config.data_seed)
        bank = SeenInputBank(config.trainer.seen_input_capacity)
        report_state = {}
    if stop_after_updates is not None and stop_after_updates < learner.completed_updates:
        raise ValueError("stop_after_updates precedes the restored update")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    if resume is None and metrics_path.exists():
        raise FileExistsError(f"run already exists: {output}; choose another output_dir or --resume")
    if resume is not None and metrics_path.exists():
        records = [line for line in metrics_path.read_text().splitlines()
                   if json.loads(line).get("completed_updates", 0) <= learner.completed_updates]
        metrics_path.write_text("\n".join(records) + ("\n" if records else ""))
    # Persist method-constrained architecture settings, not just the user's shorthand.
    config.model = {"name": config.model["name"], **learner.resolved_model_config}
    config.method = {"name": config.method["name"], **learner.resolved_method_config}
    if hasattr(consumer.paradigm, "config"):
        config.data = plain(consumer.paradigm.config)
    save_yaml(output / "config.yaml", config)
    recorder = Recorder(metrics_path, consumer.paradigm.problem)
    if resume is not None and saved.get("recorder_state"):
        recorder.load_state_dict(saved["recorder_state"])
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parents[1].rglob("*.py")):
        code.update(str(path.relative_to(Path(__file__).parents[1])).encode())
        code.update(path.read_bytes())
    manifest = {"format_version": 1, "python": platform.python_version(),
                "code_sha256": code.hexdigest(), "data_state_artifact": "artifacts/data_state.pt",
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "PyYAML")},
                "problem": asdict(consumer.paradigm.problem), "device": config.device,
                "parameters": sum(p.numel() for p in learner.network.parameters()),
                "selected_parameters": getattr(learner, "selected_parameters", []),
                "selected_sites": getattr(learner, "selected_sites", []),
                "initialization_rules": getattr(learner.network, "initialization_rules", {}),
                **getattr(learner, "cost_metrics", {}),
                "initialization_seed": config.model_seed,
                "adaptations": "See docs/ and testbed-impl-plan.md; recipes do not claim exact paper reproduction."}
    write_json(output / "manifest.json", manifest)
    start_time = time.perf_counter()
    bank_path = output / "artifacts" / "seen_inputs.pt"

    def save(name):
        state = snapshot(config, learner, consumer, bank, bank_path=bank_path,
                         recorder_state=recorder.state_dict(), report_state=report_state)
        path = output / "checkpoints" / f"{name}.pt"
        atomic_save(state, path)
        return state, path

    def report(*, final=False):
        update = learner.completed_updates
        interval = config.trainer.eval_every_updates
        due = update == 0 or final or (interval is not None and update % interval == 0)
        if due and update not in report_state.setdefault("evaluated", []):
            for split in config.trainer.eval_splits:
                data = consumer.paradigm.get_eval_data(split)
                scores = evaluate(learner, data, consumer.paradigm.problem, batch_size=max(32, config.trainer.batch_size))
                if scores is not None:
                    recorder.writer.write({"kind": "evaluation", "split": split,
                                           "completed_updates": update, **consumer.counters, **scores})
            fitting = consumer.fitting_data()
            if fitting is not None:
                scores = evaluate(learner, fitting, consumer.paradigm.problem,
                                  batch_size=max(32, config.trainer.batch_size))
                recorder.writer.write({"kind": "current_fitting", "completed_updates": update,
                                       **consumer.counters, **scores})
            if recorder.online.count:
                recorder.writer.write({"kind": "online", "completed_updates": update,
                                       **consumer.counters, **recorder.online.result()})
            report_state["evaluated"].append(update)
        if update in config.trainer.fresh_reference_at_updates and update not in report_state.setdefault("references", []):
            result = training_window(config, learner, consumer, config.trainer.fresh_reference_updates,
                                     output_path=output / "references" / f"update_{update}.json", datasets=datasets)
            recorder.writer.write(result)
            report_state["references"].append(update)
        if update in config.trainer.probe_at_updates and update not in report_state.setdefault("probed", []):
            from testbed.cli import run_test
            state, _ = save(f"update_{update}")
            for path in config.trainer.probes:
                recipe = load_yaml(path)
                recipe.setdefault("seed", config.probe_seed)
                recipe.setdefault("device", config.device)
                run_test(recipe, state, output / "probes" / f"update_{update}", datasets=datasets)
            report_state["probed"].append(update)
            save(f"update_{update}")

    report()
    if resume is None:
        save("init")
    while stop_after_updates is None or learner.completed_updates < stop_after_updates:
        try:
            batch = next(consumer)
        except StopIteration:
            break
        result = learner.train_step(batch)
        consumer.commit()
        bank.add(batch)
        recorder.record(result, learner.completed_updates, consumer.counters, batch=batch,
                        lr=learner.optimizer.param_groups[0]["lr"], elapsed=time.perf_counter() - start_time)
        report()
        interval = config.trainer.checkpoint_every_updates
        if interval and learner.completed_updates % interval == 0:
            save(f"update_{learner.completed_updates}")
    if consumer.done:
        report(final=True)
        state, path = save("final")
    else:
        state, path = save(f"update_{learner.completed_updates}")
    atomic_save(consumer.paradigm.state_dict(), output / "artifacts" / "data_state.pt")
    write_json(output / "summary.json", {"completed_updates": learner.completed_updates,
               **consumer.counters, "complete": consumer.done, "checkpoint": str(path),
               "online": recorder.online.result() if recorder.online.count else None})
    return state
