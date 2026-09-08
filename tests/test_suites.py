from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from test_integration_probes import recipe

from testbed.core.checkpoint import load_checkpoint
from testbed.core.trainer import train
from testbed.suite import run_suite


def test_sequential_checkpoint_suite_matches_standalone_and_keeps_jobs(tmp_path: Path) -> None:
    config = recipe(tmp_path / "source")
    train(config)
    probe = tmp_path / "probe.yaml"
    probe.write_text(yaml.safe_dump(dict(paradigm="random_teacher", probe=dict(updates=1, n_samples=4, batch_size=4))))
    suite = dict(kind="checkpoint_probes", probe_config=str(probe),
                 checkpoints=[str(tmp_path / "source/checkpoints/update_*.pt")],
                 repeat_seeds=[5], output_dir=str(tmp_path / "suite"))
    run_suite(suite)
    jobs = json.loads((tmp_path / "suite/jobs.json").read_text())
    assert [j["source_update"] for j in jobs] == sorted(j["source_update"] for j in jobs)
    assert len(list((tmp_path / "suite").glob("**/fitting_data.pt"))) == len(jobs)
    before = list((tmp_path / "suite").glob("**/fitting_data.pt"))
    run_suite(suite)
    assert len(list((tmp_path / "suite").glob("**/fitting_data.pt"))) == len(before)


def test_experiment_grid_aggregates_independent_seeds(tmp_path: Path) -> None:
    config = recipe(tmp_path / "unused")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    run_suite(dict(kind="experiments", recipes=[str(path)], grid={"method.name": ["backprop", "layer_norm"]},
                   seeds=[0, 1], output_dir=str(tmp_path / "suite")))
    summary = json.loads((tmp_path / "suite/summary.json").read_text())
    assert len(summary) == 2
    assert all(s["seeds"] == 2 and s["family"] == "class_remap" for s in summary)


def test_target_size_branches_start_from_identical_source_state(tmp_path: Path) -> None:
    config = recipe(tmp_path / "unused")
    config["paradigm"] = "class_incremental"
    config["data"] = dict(dataset="synthetic", target_dataset="synthetic", progression="transfer",
                           stage_sizes=[12, 12], task_samples="pool", chunk_size="task", epochs=None,
                           updates=[3, 2], validation_fraction=0,
                           data_options=dict(n_train=24, n_test=8, num_classes=3, input_shape=[1, 4, 4]),
                           target_data_options=dict(n_train=24, n_test=8, num_classes=3, input_shape=[1, 4, 4]))
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(config))
    run_suite(dict(kind="transfer_branches", source_recipe=str(path), source_update=3,
                   target_sizes=[6, 12], output_dir=str(tmp_path / "suite")))
    sources = [load_checkpoint(tmp_path / f"suite/target_{size}/source.pt") for size in (6, 12)]
    for name, value in sources[0]["learner_state"]["network"].items():
        assert torch.equal(value, sources[1]["learner_state"]["network"][name])
    for size in (6, 12):
        final = load_checkpoint(tmp_path / f"suite/target_{size}/checkpoints/final.pt")
        assert final["source_update"] == 5
        assert len(final["paradigm_state"]["stage_pools"][1]) == size
