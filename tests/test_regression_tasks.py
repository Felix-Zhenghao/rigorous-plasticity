from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from test_core import assert_state_equal
from test_data import bundle

from testbed.training.class_remap import ClassRemap


def regression(**overrides: Any) -> ClassRemap:
    config = dict(dataset="toy", num_tasks=3, task_samples="pool", chunk_size="task",
                  first_mapping="identity", target_mode="fixed_regression", pool_size=16,
                  validation_fraction=0.2, teacher={"name": "mlp", "hidden_sizes": [8], "dropout": 0.5})
    config.update(overrides)
    return ClassRemap(config, datasets=bundle(), seed=17)


@pytest.mark.parametrize("family", ["teacher", "sine_teacher"])
def test_teacher_changes_only_between_tasks_on_the_same_inputs(family: str) -> None:
    rng = torch.random.get_rng_state()
    p = regression(target_family=family)
    initial_eval = p.get_eval_data("test")
    views, evaluations, targets, seeds = [], [], [], []
    for _ in range(p.config.num_tasks):
        data = p.get_data().data
        evaluation = p.get_eval_data("test")
        assert p.get_eval_data("test") is evaluation
        assert torch.equal(data[0][1], data[0][1])
        if views:
            assert torch.equal(data.inputs, views[-1].inputs)
            assert torch.equal(data.ids, views[-1].ids)
            assert not torch.equal(data.targets, views[-1].targets)
            assert not torch.equal(evaluation.targets, evaluations[-1].targets)
        else:
            assert evaluation is initial_eval
        views.append(data)
        evaluations.append(evaluation)
        targets.append(data.targets.clone())
        seeds.append(p.fixed_regression["generator"]["seed"])
    assert len(set(seeds)) == p.config.num_tasks
    assert all(torch.equal(view.targets, saved) for view, saved in zip(views, targets))
    assert p.get_data() is None
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("center", [False, True])
def test_target_schedules_and_heldout_centering_follow_each_tasks_teacher(center: bool) -> None:
    means, scales, frequencies = [0, 8, -2], [0, 2, 0.5], [0, 3, -7]
    p = regression(target_family="sine_teacher", target_mean=means, target_scale=scales,
                   omega=frequencies, center_targets=center)
    raw = regression(center_targets=False)
    for mean, scale, omega in zip(means, scales, frequencies):
        data, raw_data = p.get_data().data, raw.get_data().data
        base = torch.sin(omega * raw_data.targets)
        training_center = base.mean() if center else 0
        assert torch.allclose(data.targets, mean + scale * (base - training_center), atol=1e-6)
        if center:
            assert float(data.targets.mean()) == pytest.approx(mean, abs=1e-6)
        artifact = p.fixed_regression
        assert artifact["target_mean"] == mean
        assert artifact["generator"]["target_scale"] == scale
        assert artifact["generator"]["omega"] == omega
        for split in ("val", "test"):
            raw_eval = raw.get_eval_data(split)
            expected = mean + scale * (torch.sin(omega * raw_eval.targets) - training_center)
            assert torch.allclose(p.get_eval_data(split).targets, expected, atol=1e-6)


@pytest.mark.parametrize("family", ["teacher", "sine_teacher"])
def test_regression_restore_recovers_active_evaluation_and_next_task(family: str) -> None:
    options = dict(target_family=family, target_mean=[0, 4, 8], target_scale=[1, 0, 2], validation_fraction=0)
    p = regression(**options)
    p.get_data()
    p.get_data()
    saved = deepcopy(p.state_dict())
    expected_eval = p.get_eval_data("test")
    expected = p.get_data().data
    restored = regression(**options)
    restored.get_eval_data("test")
    restored.load_state_dict(saved)
    assert torch.equal(restored.get_eval_data("test").targets, expected_eval.targets)
    assert restored.get_eval_data("val").targets.shape == (0, 1)
    actual = restored.get_data().data
    assert torch.equal(actual.inputs, expected.inputs)
    assert torch.equal(actual.ids, expected.ids)
    assert torch.equal(actual.targets, expected.targets)
    assert_state_equal(p.state_dict(), restored.state_dict())


@pytest.mark.parametrize("name", ["target_mean", "target_scale", "omega"])
@pytest.mark.parametrize("values", [[], [1, 2], [1, 2, 3, 4]])
def test_target_schedules_require_one_value_per_task(name: str, values: list[float]) -> None:
    with pytest.raises(ValueError, match=f"{name} must contain exactly 3"):
        regression(target_family="sine_teacher", **{name: values})


@pytest.mark.parametrize("options", [
    {"target_mean": float("inf")}, {"target_mean": [0, float("nan"), 0]},
    {"target_scale": -1}, {"target_scale": [1, 1, -1]},
    {"omega": float("nan")}, {"omega": [1, float("inf"), 2]},
])
def test_target_parameters_must_be_finite_with_nonnegative_scale(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="finite|nonnegative"):
        regression(target_family="sine_teacher", **options)


@pytest.mark.parametrize("family", ["teacher", "sine_teacher"])
def test_regression_requires_a_teacher(family: str) -> None:
    with pytest.raises(ValueError, match="requires a teacher"):
        regression(target_family=family, teacher=None)


def test_removed_target_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_family must be teacher or sine_teacher"):
        regression(target_family="iid_normal")


@pytest.mark.parametrize("options", [{"task_samples": ["pool", 8, "pool"]}, {"chunk_size": ["task", 8, "task"]}])
def test_every_regression_task_requires_the_complete_fixed_pool(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="complete fixed pool"):
        regression(**options)
