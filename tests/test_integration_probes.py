from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from test_core import assert_state_equal

from testbed.core.checkpoint import load_checkpoint
from testbed.core.trainer import train
from testbed.testing.common import probe_network, seen_inputs
from testbed.testing.offset_refit import OffsetProbeConfig
from testbed.testing.offset_refit import run as offset_probe
from testbed.testing.random_teacher import TeacherProbeConfig
from testbed.testing.random_teacher import run as teacher_probe


def recipe(
    output: str | Path, *, regression: bool = False, method: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(dataset="synthetic", num_tasks=2, task_samples=12, chunk_size=6, epochs=2,
                validation_fraction=0, data_options=dict(n_train=24, n_test=12, num_classes=3, input_shape=[1, 4, 4]))
    if regression:
        data.update(num_tasks=1, first_mapping="identity", target_mode="fixed_regression",
                    target_mean=8, pool_size=12, task_samples="pool", chunk_size="task")
    return dict(paradigm="class_remap", data=data,
                model=dict(name="mlp", hidden_sizes=[12], dropout=.1),
                method=method or dict(name="backprop"), optimizer=dict(name="adam", lr=.01),
                trainer=dict(batch_size=4, eval_every_updates=4, checkpoint_every_updates=4,
                             seen_input_capacity=12), seed=3, output_dir=str(output))


def test_exact_training_resume(tmp_path: Path) -> None:
    full = train(recipe(tmp_path / "full"))
    config = recipe(tmp_path / "resumed")
    partial = train(config, stop_after_updates=5)
    assert partial["source_update"] == 5
    resumed = train(resume=tmp_path / "resumed/checkpoints/update_5.pt")
    assert_state_equal(full["learner_state"], resumed["learner_state"])
    assert_state_equal(full["consumer_state"], resumed["consumer_state"])


def test_teacher_classification_fixed_targets_and_optimizer_policy(tmp_path: Path) -> None:
    source = train(recipe(tmp_path / "source"))
    original = deepcopy(source)
    config = TeacherProbeConfig(updates=3, n_samples=6, batch_size=4, eval_every_updates=2,
                                optimizer=dict(name="adam", lr=.001))
    result = teacher_probe(config, source, tmp_path / "probe", seed=7)
    artifact = torch.load(next((tmp_path / "probe").glob("**/fitting_data.pt")), weights_only=False)
    assert artifact["targets"].shape == (6,)
    assert set(artifact["targets"].tolist()) <= set(source["problem"]["output_ids"])
    assert "accuracy" in result.comparisons[0]["branches"]["aged"]["terminal"]
    assert_state_equal(source, original)
    learner = probe_network(source, config, fresh=False, device="cpu", seed=7)
    assert not learner.optimizer.state
    expected = learner.predict(artifact["inputs"])
    carrying = TeacherProbeConfig(updates=3, n_samples=6, load_optimizer_state=True)
    carried = probe_network(source, carrying, fresh=False, device="cpu", seed=7)
    assert torch.equal(expected, carried.predict(artifact["inputs"]))
    assert carried.optimizer.state
    assert carried.optimizer.param_groups[0]["lr"] == .001
    assert carried.completed_updates == 0


def test_probe_loads_network_moments_excludes_infer_auxiliary(tmp_path: Path) -> None:
    source = train(recipe(tmp_path / "source", method=dict(name="infer", coefficient=.01, num_heads=2, target_scale=2)))
    config = TeacherProbeConfig(updates=1, n_samples=4, load_optimizer_state=True)
    learner = probe_network(source, config, fresh=False, device="cpu", seed=0)
    assert len(learner.optimizer.state) == len(list(learner.network.parameters()))
    assert all(name.startswith("network.") for group in learner.optimizer.param_groups for name in group["param_names"])
    source["config"]["optimizer"]["weight_decay"] = .3
    with pytest.raises(ValueError, match="non-LR"):
        probe_network(source, config, fresh=False, device="cpu", seed=0)


def test_seen_prefix_and_insufficient_initialization_bank(tmp_path: Path) -> None:
    train(recipe(tmp_path / "source"))
    initial = load_checkpoint(tmp_path / "source/checkpoints/init.pt")
    early = load_checkpoint(tmp_path / "source/checkpoints/update_4.pt")
    with pytest.raises(ValueError, match="valid seen"):
        seen_inputs(initial, 1, seed=0)
    x, ids = seen_inputs(early, 6, seed=0)
    bank = torch.load(early["seen_inputs"]["path"], weights_only=False)
    assert set(ids.tolist()) <= set(bank["ids"][:early["seen_inputs"]["prefix"]].tolist())


def test_scheduled_probe_and_fresh_window_do_not_change_primary_training(tmp_path: Path) -> None:
    plain = train(recipe(tmp_path / "plain"))
    probe_path = tmp_path / "probe.yaml"
    probe_path.write_text(yaml.safe_dump(dict(paradigm="random_teacher", probe=dict(updates=2, n_samples=4, batch_size=4))))
    config = recipe(tmp_path / "with_probes")
    config["trainer"].update(probe_at_updates=[4], probes=[str(probe_path)],
                             fresh_reference_at_updates=[4], fresh_reference_updates=2)
    probed = train(config)
    assert_state_equal(plain["learner_state"], probed["learner_state"])
    assert_state_equal(plain["consumer_state"], probed["consumer_state"])


def test_offset_exact_inputs_independent_offsets_and_regression_teacher(tmp_path: Path) -> None:
    source = train(recipe(tmp_path / "source", regression=True))
    fixed = source["paradigm_state"]["fixed_regression"]
    config = OffsetProbeConfig(updates=2, batch_size=4, eval_every_updates=1)
    offset_probe(config, source, tmp_path / "offset", seed=9)
    artifacts = [torch.load(p, weights_only=False) for p in (tmp_path / "offset").glob("**/fitting_data.pt")]
    assert len(artifacts) == 2
    for artifact in artifacts:
        assert torch.equal(artifact["inputs"], fixed["inputs"])
        assert abs(float(artifact["targets"].mean()) - artifact["target_offset"]) < 1e-6
    centered = [a["targets"] - a["target_offset"] for a in artifacts]
    assert torch.allclose(centered[0], centered[1], atol=1e-6)
    teacher = teacher_probe(TeacherProbeConfig(updates=2, n_samples=6), source, tmp_path / "teacher", seed=9)
    assert "accuracy" not in teacher.comparisons[0]["branches"]["aged"]["terminal"]
    del source["paradigm_state"]["fixed_regression"]
    with pytest.raises(ValueError, match="exact-X"):
        offset_probe(config, source, tmp_path / "missing")


def test_same_fixed_data_probes_independent_of_launch_order(tmp_path: Path) -> None:
    source = train(recipe(tmp_path / "source"))
    config = TeacherProbeConfig(updates=2, n_samples=5, batch_size=3)
    a = teacher_probe(config, source, tmp_path / "a", seed=10)
    teacher_probe(config, source, tmp_path / "between", seed=999)
    b = teacher_probe(config, source, tmp_path / "b", seed=10)
    assert a.comparisons == b.comparisons


def test_probe_cache_distinguishes_checkpoint_age_in_same_directory(tmp_path: Path) -> None:
    source = train(recipe(tmp_path / "source"))
    early = load_checkpoint(tmp_path / "source/checkpoints/update_4.pt")
    config = TeacherProbeConfig(updates=1, n_samples=4, batch_size=4)
    a = teacher_probe(config, early, tmp_path / "probes", seed=2)
    b = teacher_probe(config, source, tmp_path / "probes", seed=2)
    assert a.comparisons[0]["source_update"] == 4
    assert b.comparisons[0]["source_update"] == source["source_update"]
    assert a.comparisons[0]["assay_fingerprint"] == b.comparisons[0]["assay_fingerprint"]
    assert a.comparisons[0]["source_fingerprint"] != b.comparisons[0]["source_fingerprint"]
