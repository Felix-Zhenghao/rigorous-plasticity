from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from testbed.core.config import load_yaml, strict_dataclass
from testbed.core.random import isolated_rng

if TYPE_CHECKING:
    from testbed.data.datasets import DatasetInput
    from testbed.testing.common import ProbeResult


def run_test(
    recipe: dict[str, Any],
    checkpoint: str | Path | dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    datasets: DatasetInput = None,
) -> ProbeResult:
    """Run a standalone or scheduled probe from a strict recipe mapping.

    paradigm: "random_teacher" or "offset_refit"; chooses the probe config.
    probe: TeacherProbeConfig or OffsetProbeConfig fields, respectively.
    seed: Probe RNG seed; omitted values use the checkpoint's probe_seed.
    device: PyTorch device for fitting; defaults to "cpu" for standalone tests.
    data_root: Optional replacement dataset cache root; otherwise use the source run.
    output_dir: Root for results; default is runs/probes/<source>/update_<N>.
        The caller's output_dir takes precedence, and paradigm/seed subdirectories
        are appended. Source and assay fingerprints identify each fitting result.
    """
    from testbed.testing.common import source_state
    from testbed.testing.offset_refit import OffsetProbeConfig
    from testbed.testing.offset_refit import run as offset
    from testbed.testing.random_teacher import TeacherProbeConfig
    from testbed.testing.random_teacher import run as teacher
    registry = {"random_teacher": (TeacherProbeConfig, teacher), "offset_refit": (OffsetProbeConfig, offset)}
    allowed = {"paradigm", "probe", "seed", "device", "data_root", "output_dir"}
    if set(recipe) - allowed:
        raise ValueError(f"unknown probe recipe fields: {sorted(set(recipe) - allowed)}")
    if recipe.get("paradigm") not in registry:
        raise ValueError("test paradigm must be random_teacher or offset_refit")
    cls, run = registry[recipe["paradigm"]]
    config = strict_dataclass(cls, recipe.get("probe", {}))
    state = source_state(checkpoint)
    seed = recipe.get("seed", state["config"]["probe_seed"])
    if output_dir is None:
        source_run = Path(state["config"]["output_dir"]).name
        output_dir = recipe.get("output_dir", f"runs/probes/{source_run}/update_{state['source_update']}")
    destination = Path(output_dir) / recipe["paradigm"] / f"seed_{seed}"
    with isolated_rng():
        return run(config, state, destination, seed=seed, device=recipe.get("device", "cpu"),
                   data_root=recipe.get("data_root"), datasets=datasets)


def main(
    argv: Sequence[str] | None = None,
    *,
    default_command: str | None = None,
    expected_paradigm: str | None = None,
) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if default_command:
        argv.insert(0, default_command)
    parser = argparse.ArgumentParser(description="Supervised loss-of-plasticity testbed")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "test", "suite"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=name != "train")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "train":
            command.add_argument("--resume", type=Path)
            command.add_argument("--stop-after-updates", type=int)
        if name == "test":
            command.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = load_yaml(args.config, args.set) if args.config else None
    if expected_paradigm and recipe and recipe.get("paradigm") != expected_paradigm:
        parser.error(f"this launcher requires paradigm={expected_paradigm}")
    if args.command == "train":
        from testbed.core.trainer import train
        if recipe is None and args.resume is None:
            parser.error("train requires --config or --resume")
        if recipe is None and args.set:
            import yaml

            from testbed.core.config import set_dotted
            recipe = {}
            for override in args.set:
                key, value = override.split("=", 1)
                set_dotted(recipe, key, yaml.safe_load(value))
        result = train(recipe, resume=args.resume, stop_after_updates=args.stop_after_updates)
        print(f"Saved {result['source_update']} completed updates to {result['config']['output_dir']}")
    elif args.command == "test":
        result = run_test(recipe, args.checkpoint)
        print(f"Saved {len(result.comparisons)} probe comparisons to {result.output_dir}")
    else:
        from testbed.suite import run_suite
        result = run_suite(recipe, config_dir=args.config.parent)
        print(f"Saved suite results to {result}")
