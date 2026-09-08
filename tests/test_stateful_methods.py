"""State semantics and small source-operation traces for maintenance methods."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from testbed.core.factory import METHOD_ARCHITECTURES, make_model
from testbed.core.types import Batch, ProblemSpec
from testbed.methods.backprop.learner import BackpropLearner
from testbed.methods.fire.learner import newton_schulz
from testbed.methods.optimizer_reset.learner import OptimizerResetLearner

CONFIGS = {
    "c_chain": {"buffer_size": 8, "reference_batch_size": 3},
    "shrink_perturb": {"every_updates": 2},
    "cbp": {"replacement_rate": 0.3, "maturity_updates": 1},
    "redo": {"every_updates": 2, "threshold": 0.8},
    "swr": {"every_updates": 2, "fraction": 0.2, "ema_decay": 0.5},
    "fire": {"every_updates": 2},
    "nap": {},
    "optimizer_reset": {"every_updates": 2},
}
MODELS = {
    "mlp": {"hidden_sizes": [4, 4], "dropout": 0.2},
    "resnet_18": {"base_channels": 2},
    "vit": {"patch_size": 4, "depth": 1, "embed_dim": 8, "num_heads": 2,
            "mlp_dim": 12, "feedforward_dropout": 0.2},
}


def learner(
    method: str,
    architecture: str = "mlp",
    *,
    config: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    loss: str = "cross_entropy",
    optimizer: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BackpropLearner | OptimizerResetLearner:
    shape = (4,) if architecture == "mlp" else (3, 8, 8)
    problem = ProblemSpec(shape, loss, (0, 1) if loss == "cross_entropy" else (0,))
    return make_model(method, architecture, problem=problem, model_config=model or MODELS[architecture],
                      method_config=CONFIGS[method] if config is None else config,
                      optimizer_config=optimizer or {"name": "adam", "lr": 0.002, "amsgrad": True},
                      seed=19, method_seed=37, **kwargs)


def batch(architecture: str = "mlp", index: int = 0) -> Batch:
    generator = torch.Generator().manual_seed(index + 101)
    shape = (4,) if architecture == "mlp" else (3, 8, 8)
    return torch.randn(2, *shape, generator=generator), torch.tensor([0, 1]), torch.arange(2) + 2 * index


def assert_equal(left: object, right: object) -> None:
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method,architecture", [(m, a) for m in CONFIGS for a in METHOD_ARCHITECTURES[m]])
def test_resume_and_prediction_are_exact(method: str, architecture: str) -> None:
    aged = learner(method, architecture)
    aged.train_step(batch(architecture, 0))
    aged.train_step(batch(architecture, 1))
    saved = aged.state_dict()
    modes = [module.training for module in aged.network.modules()]
    aged.predict(batch(architecture, 2)[0])
    assert modes == [module.training for module in aged.network.modules()]
    assert_equal(saved, aged.state_dict())
    restored = learner(method, architecture)
    restored.load_state_dict(saved)
    for i in (2, 3):
        a, b = aged.train_step(batch(architecture, i)), restored.train_step(batch(architecture, i))
        torch.testing.assert_close(a.predictions, b.predictions, rtol=0, atol=0)
    assert aged.completed_updates == restored.completed_updates == 4
    assert_equal(aged.state_dict(), restored.state_dict())


def test_fire_matches_author_three_iteration_trace_and_handles_zero() -> None:
    # FIRE 3f73d78, vision/interventions/fire.py:newton_schulz, FP32 output.
    matrix = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    expected = torch.tensor([[0.0172563195, 0.2792161107], [0.2910345197, 0.4387994409],
                             [0.5648127794, 0.5983827114]])
    actual, skipped = newton_schulz(matrix, iterations=3)
    torch.testing.assert_close(actual, expected)
    transposed, _ = newton_schulz(matrix.T, iterations=3)
    torch.testing.assert_close(transposed, expected.T)
    zero, skipped = newton_schulz(torch.zeros(2, 3))
    assert skipped == 1 and torch.count_nonzero(zero) == 0
    projected, _ = newton_schulz(matrix, iterations=10)
    assert (projected.T @ projected - torch.eye(2)).norm() < (matrix.T @ matrix - torch.eye(2)).norm()


def test_fire_spatial_slices_scaling_and_vit_qk_scope() -> None:
    conv = learner("fire", "resnet_18", config={"every_updates": 1, "iterations": 3})
    weight = conv.network.stem_conv.weight
    before = weight.detach().clone()
    identity = id(weight)
    conv.intervene()
    for i in range(3):
        for j in range(3):
            expected, _ = newton_schulz(before[:, :, i, j], iterations=3)
            torch.testing.assert_close(weight[:, :, i, j], expected * math.sqrt(2 / 3) / 9)
    assert id(weight) == identity
    vit = learner("fire", "vit", config={"every_updates": 1})
    before = deepcopy(vit.network.state_dict())
    vit.intervene()
    for name, parameter in vit.network.state_dict().items():
        if name not in vit.selected_weights:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert set(vit.selected_weights) == {"blocks.0.attention.q.weight", "blocks.0.attention.k.weight"}


def test_nap_radii_affine_decay_and_architecture() -> None:
    model = learner("nap", model={"hidden_sizes": [4], "ln_gain": "residual"},
                    config={"affine_decay": 0.25, "at_updates": [2]})
    assert model.config.every_updates is None
    assert model.network.hidden[0].linear.bias is None
    norm = model.network.hidden[0].norm
    with torch.no_grad():
        for parameter in model.selected_weights.values():
            parameter.mul_(3)
        norm.weight.fill_(1)
        norm.bias.fill_(2)
    model.intervene()
    for name, parameter in model.selected_weights.items():
        torch.testing.assert_close(parameter.norm(), model.radii[name])
    torch.testing.assert_close(norm.gain, torch.full((4,), 1.75))
    torch.testing.assert_close(norm.bias, torch.full((4,), 1.5))
    residual = learner("nap", "resnet_18")
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in residual.network.modules())
    assert all(isinstance(block.sum_norm, nn.LayerNorm) for stage in residual.network.stages for block in stage)


def test_nap_joint_projection_uses_width_and_skips_zero() -> None:
    model = learner("nap", model={"hidden_sizes": [4]}, config={"affine_policy": "joint_project"})
    norm = model.network.hidden[0].norm
    with torch.no_grad():
        norm.weight.fill_(3)
        norm.bias.fill_(4)
        model.network.head.weight.zero_()
    assert model.intervene() == 1
    torch.testing.assert_close(norm.gain.square().sum() + norm.bias.square().sum(), torch.tensor(4.))
    assert model.network.head.weight.count_nonzero() == 0
    with pytest.raises(ValueError, match="homogeneous"):
        learner("nap", model={"hidden_sizes": [4], "activation": "gelu"}, config={"affine_policy": "joint_project"})


def test_shrink_perturb_saved_blend_and_absolute_timing() -> None:
    config = {"at_updates": [4], "retain": 0.75, "noise_scale": 0.25, "noise_source": "saved_init"}
    model = learner("shrink_perturb", config=config, start_update=3,
                    optimizer={"name": "sgd", "lr": 0.0}, model={"hidden_sizes": [4]})
    before = {name: p.detach().clone() for name, p in model.selected_weights.items()}
    with torch.no_grad():
        for parameter in model.selected_weights.values():
            parameter.add_(2)
    result = model.train_step(batch())
    assert result.metrics["intervention"] == 1
    for name, parameter in model.selected_weights.items():
        torch.testing.assert_close(parameter, before[name] + 1.5)
    state = model.state_dict()
    model.load_state_dict(state)
    assert model.train_step(batch(index=1)).metrics["intervention"] == 0


def test_swr_scores_post_optimizer_weights_with_retained_gradients() -> None:
    model = learner("swr", loss="mse", model={"hidden_sizes": [], "head_bias": False},
                    config={"every_updates": 1, "utility": "gradient", "fraction": 0.25, "replacement": "init_mean"},
                    optimizer={"name": "sgd", "lr": 1.0})
    with torch.no_grad():
        model.network.head.weight.copy_(torch.tensor([[1., 2., 4., 5.]]))
    # grad = [.9, .2, .3, .4]; post-update utility chooses coordinate 0,
    # while pre-update utility would choose coordinate 1.
    x = torch.tensor([[.9, .2, .3, .4]])
    target = model.predict(x).detach() - .5
    model.train_step((x, target, torch.tensor([0])))
    torch.testing.assert_close(model.network.head.weight, torch.tensor([[0., 1.8, 3.7, 4.6]]))


@pytest.mark.parametrize("gain,expected", [("standard", 1.0), ("residual", 0.0)])
def test_swr_initializer_mean_includes_norm_and_clears_whole_ema(gain: str, expected: float) -> None:
    model = learner("swr", model={"hidden_sizes": [4], "norm": "layer", "ln_gain": gain},
                    config={"every_updates": 1, "fraction": 1., "replacement": "init_mean", "ema_decay": .5})
    with torch.no_grad():
        model.network.hidden[0].norm.weight.fill_(3)
    model.maintain(True)
    assert torch.all(model.network.hidden[0].norm.weight == expected)
    assert all(utility.count_nonzero() == 0 for utility in model.utilities.values())


def test_cbp_maturity_and_fractional_accumulation() -> None:
    model = learner("cbp", model={"hidden_sizes": [2]}, config={"maturity_updates": 1,
                    "replacement_rate": .25, "ema_decay": .5})
    features = {"hidden.0": torch.tensor([[1., 4.], [1., 4.]])}
    with torch.no_grad():
        model.network.head.weight.fill_(1)
    assert model.update_statistics(features) == 0
    assert model.update_statistics(features) == 0
    assert model.fractional_counts["hidden.0"] == .5
    assert model.update_statistics(features) == 1
    assert model.ages["hidden.0"].tolist() == [0, 3]
    assert model.network.head.weight[:, 0].count_nonzero() == 0
    # Plan's sum-weight utility is source GnT contribution utility times fan-out.
    torch.testing.assert_close(model.utilities["hidden.0"][1], torch.tensor(7.))


@pytest.mark.parametrize("method", ["cbp", "redo"])
def test_overlapping_recycling_masks_clear_only_affected_moments(method: str) -> None:
    model = learner(method, model={"hidden_sizes": [3, 3], "norm": "layer", "ln_gain": "residual"})
    for parameter in model.parameters:
        parameter.grad = torch.ones_like(parameter)
    model.optimizer.step()
    moment = model.optimizer.state[model.network.hidden[1].linear.weight]
    before = moment["exp_avg"].clone()
    step = moment["step"].clone()
    selections = {"hidden.0": torch.tensor([1]), "hidden.1": torch.tensor([0])}
    model.replace_units(selections)
    middle = model.network.hidden[1].linear.weight
    assert middle[:, 1].count_nonzero() == 0
    assert moment["exp_avg"][:, 1].count_nonzero() == 0
    assert moment["exp_avg"][0].count_nonzero() == 0
    torch.testing.assert_close(moment["exp_avg"][1:, 2], before[1:, 2])
    torch.testing.assert_close(moment["step"], step)
    assert model.network.hidden[0].norm.weight[1] == 0


def test_redo_all_zero_scores_and_bounded_sample_weighted_window() -> None:
    model = learner("redo", model={"hidden_sizes": [2]}, config={"every_updates": 1, "threshold": 0.,
                    "statistics_window_updates": 2, "reset_source": "saved_init"})
    assert model.update_statistics({"hidden.0": torch.zeros(1, 2)}, True) == 2
    model.update_statistics({"hidden.0": torch.tensor([[4., 0.]])}, False)
    assert model.update_statistics({"hidden.0": torch.zeros(3, 2)}, True) == 1
    assert len(model.windows["hidden.0"]) == 2


def test_redo_resnet_batchnorm_channels_and_counter() -> None:
    model = learner("redo", "resnet_18")
    incoming, norm, outgoing = model.sites["stages.0.0.conv1"]
    with torch.no_grad():
        norm.running_mean.fill_(3)
        norm.running_var.fill_(4)
        norm.num_batches_tracked.fill_(9)
    model.replace_units({"stages.0.0.conv1": torch.tensor([0])})
    assert norm.running_mean.tolist() == [0., 3.]
    assert norm.running_var.tolist() == [1., 4.]
    assert norm.num_batches_tracked.item() == 9
    assert outgoing.weight[:, 0].count_nonzero() == 0


def test_chain_lag_disjoint_lru_and_nonzero_matching_gradient() -> None:
    model = learner("c_chain", model={"hidden_sizes": []}, loss="mse",
                    config={"buffer_size": 2, "reference_batch_size": 2})
    first = (torch.ones(1, 4), torch.ones(1, 1), torch.tensor([10]))
    assert model.train_step(first).metrics["warmup"] == 1
    model.optimizer.zero_grad(set_to_none=True)
    reference_loss = model.reference_loss({11})
    reference_loss.backward()
    assert sum(p.grad.abs().sum() for p in model.network.parameters() if p.grad is not None) > 0
    assert model.last_reference_ids == (10,)
    model.train_step((torch.full((1, 4), 2.), torch.ones(1, 1), torch.tensor([11])))
    model.train_step((torch.full((1, 4), 3.), torch.ones(1, 1), torch.tensor([10])))
    assert model.last_reference_ids == (11,)
    model.train_step((torch.full((1, 4), 4.), torch.ones(1, 1), torch.tensor([12])))
    assert list(model.recency) == [10, 12]
    torch.testing.assert_close(model.inputs[model.recency[10]], torch.full((4,), 3.))
    assert len(model.history) == 2


def test_chain_one_optimizer_step_and_later_coefficient_adaptation() -> None:
    model = learner("c_chain", config={"buffer_size": 8, "reference_batch_size": 3,
                    "coefficient_mode": "loss_ratio", "adapt_after_updates": 1, "coefficient": .2})
    count = 0
    original = model.optimizer.step
    def step() -> torch.Tensor | float | None:
        nonlocal count
        count += 1
        return original()
    model.optimizer.step = step
    model.train_step(batch(index=0))
    second = model.train_step(batch(index=1))
    next_coefficient = model.coefficient
    third = model.train_step(batch(index=2))
    assert count == 3
    assert second.metrics["coefficient"] == .2
    assert third.metrics["coefficient"] == next_coefficient
    assert next_coefficient != .2


def test_optimizer_reset_composition_preserves_anchors_and_parameter_objects() -> None:
    model = learner("optimizer_reset", config={"at_updates": [2], "base_method": "l2_init",
                    "base_config": {"coefficient": .1}})
    identities = [id(p) for p in model.parameters]
    model.train_step(batch(index=0))
    first = model.state_dict()
    assert first["optimizer"]["state"]
    result = model.train_step(batch(index=1))
    assert result.metrics["optimizer_reset"] == 1 and not model.optimizer.state
    assert identities == [id(p) for p in model.parameters]
    assert first["l2_init"].keys() == model.state_dict()["l2_init"].keys()
    assert_equal(first["l2_init"], model.state_dict()["l2_init"])
    model.train_step(batch(index=2))
    assert model.optimizer.state and model.resets == 1


@pytest.mark.parametrize("method", ["fire", "redo", "swr", "shrink_perturb", "optimizer_reset"])
def test_interventions_reject_missing_ambiguous_or_nonpositive_schedule(method: str) -> None:
    for config in ({}, {"every_updates": 0}, {"at_updates": [0]}, {"at_updates": [2, 1]},
                   {"every_updates": 2, "at_updates": [2]}):
        with pytest.raises(ValueError):
            learner(method, config=config)


@pytest.mark.parametrize("method,config", [
    ("shrink_perturb", {"every_updates": 1, "retain": 1., "noise_scale": 0.}),
    ("swr", {"every_updates": 1, "fraction": 0.}),
    ("cbp", {"replacement_rate": 0.}),
])
def test_disabled_replacements_preserve_backprop_dropout_trace(method: str, config: dict[str, Any]) -> None:
    control = make_model("backprop", "mlp", problem=ProblemSpec((4,), "cross_entropy", (0, 1)),
        model_config=MODELS["mlp"], optimizer_config={"name": "adam", "lr": .002, "amsgrad": True},
        seed=19, method_seed=37)
    maintained = learner(method, config=config)
    for index in range(3):
        a, b = control.train_step(batch(index=index)), maintained.train_step(batch(index=index))
        torch.testing.assert_close(a.predictions, b.predictions, rtol=0, atol=0)
    assert_equal(control.network.state_dict(), maintained.network.state_dict())
