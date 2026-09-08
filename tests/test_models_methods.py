from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn

from testbed.core.factory import METHOD_ARCHITECTURES, make_model, make_network
from testbed.core.types import Batch, ProblemSpec
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models.initialization import sample_initial
from testbed.models.layers import LayerNorm

METHODS = {
    "backprop": {}, "l2": {"coefficient": 0.01}, "l2_init": {"coefficient": 0.01},
    "feature_norm": {"coefficient": 0.01},
    "infer": {"coefficient": 0.01, "num_heads": 3, "target_scale": 2.0},
    "spectral": {"coefficient": 0.01}, "layer_norm": {}, "leaky_relu": {},
}
ARCHITECTURES = {
    "mlp": {"hidden_sizes": [8, 8]}, "resnet_18": {"base_channels": 2},
    "vit": {"depth": 1, "embed_dim": 8, "num_heads": 2, "mlp_dim": 12},
}


def learner(
    method: str = "backprop",
    architecture: str = "mlp",
    *,
    model: dict[str, Any] | None = None,
    method_config: dict[str, Any] | None = None,
    loss: str = "cross_entropy",
    **kwargs: Any,
) -> BackpropLearner:
    return make_model(method, architecture, problem=ProblemSpec((1, 8, 8), loss, (2, 7, 11)),
                      model_config=ARCHITECTURES[architecture] | (model or {}),
                      method_config=METHODS[method] | (method_config or {}),
                      optimizer_config={"name": "adam", "lr": 0.001}, **kwargs)


def batch() -> Batch:
    generator = torch.Generator().manual_seed(9)
    return torch.randn(2, 1, 8, 8, generator=generator), torch.tensor([2, 11]), torch.tensor([3, 8])


@pytest.mark.parametrize("method,architecture", [(m, a) for m in METHODS for a in METHOD_ARCHITECTURES[m]])
def test_supported_pairs_train_and_resume(method: str, architecture: str) -> None:
    current = learner(method, architecture, seed=3, method_seed=17)
    data = batch()
    result = current.train_step(data)
    assert result.predictions.shape == (2, 3)
    assert torch.isfinite(result.predictions).all()
    assert current.completed_updates == 1
    state = current.state_dict()
    restored = learner(method, architecture, seed=77, method_seed=88)
    restored.load_state_dict(state)
    torch.testing.assert_close(current.predict(data[0]), restored.predict(data[0]), rtol=0, atol=0)
    expected = current.train_step(data)
    actual = restored.train_step(data)
    torch.testing.assert_close(expected.predictions, actual.predictions, rtol=0, atol=0)
    for name, value in current.network.state_dict().items():
        torch.testing.assert_close(value, restored.network.state_dict()[name], rtol=0, atol=0)


def test_l2_half_factor_and_parameter_roles() -> None:
    current = learner("l2", model={"norm": "layer"}, method_config={"coefficient": 0.4})
    assert "hidden.0.linear.bias" in current.selected
    assert "hidden.0.norm.weight" not in current.selected
    expected = 0.2 * sum(p.square().sum() for p in current.selected.values())
    torch.testing.assert_close(current.penalty(), expected)
    current.penalty().backward()
    for parameter in current.selected.values():
        torch.testing.assert_close(parameter.grad, 0.4 * parameter)


def test_l2_init_keeps_distinct_frozen_anchors() -> None:
    current = learner("l2_init", method_config={"coefficient": 0.3})
    assert current.penalty().item() == 0
    anchors = deepcopy(current.anchors)
    with torch.no_grad():
        for parameter in current.selected.values():
            parameter.add_(0.2)
    current.penalty().backward()
    for name, parameter in current.selected.items():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.06))
        torch.testing.assert_close(current.anchors[name], anchors[name], rtol=0, atol=0)
        assert not current.anchors[name].requires_grad


def test_feature_penalty_reductions_and_readout_independence() -> None:
    current = learner("feature_norm", method_config={"coefficient": 0.7})
    predictions, features = current.forward_features(current.network, batch()[0], ("head_input",))
    penalty = current.penalty(features)
    torch.testing.assert_close(penalty, 0.7 * features["head_input"].square().mean())
    penalty.backward()
    assert current.network.head.weight.grad is None
    assert current.network.hidden[0].linear.weight.grad.abs().sum() > 0
    summed = learner("feature_norm", method_config={"coefficient": 0.7, "reduction": "sum_per_example"})
    torch.testing.assert_close(summed.penalty(features), penalty * features["head_input"].shape[1])


def test_infer_auxiliary_parameters_train_and_references_freeze() -> None:
    current = learner("infer", model={"head_hidden_sizes": [5]}, seed=4)
    initial = deepcopy(current.frozen_network.state_dict())
    initial_heads = deepcopy(current.frozen_auxiliary.state_dict())
    names = current.optimizer.param_groups[0]["param_names"]
    assert {"auxiliary.weight", "auxiliary.bias"} <= set(names)
    assert len(names) == len(set(names))
    assert current.cost_metrics["auxiliary_trainable_parameters"] == 18
    assert current.cost_metrics["frozen_state_bytes"] > 0
    before = current.auxiliary.weight.detach().clone()
    current.train_step(batch())
    assert not torch.equal(before, current.auxiliary.weight)
    for name, value in current.frozen_network.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    for name, value in current.frozen_auxiliary.state_dict().items():
        torch.testing.assert_close(value, initial_heads[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in current.frozen_network.parameters())


@pytest.mark.parametrize("architecture", ["mlp", "resnet_18"])
def test_feature_site_tracks_hidden_readout_and_matches_network(architecture: str) -> None:
    current = learner("feature_norm", architecture, model={"head_hidden_sizes": [5, 4]})
    current.network.eval()
    predictions, features = current.forward_features(current.network, batch()[0], ("head_input", "head_hidden.1"))
    assert features["head_input"].shape == (2, 4)
    torch.testing.assert_close(features["head_input"], features["head_hidden.1"], rtol=0, atol=0)
    torch.testing.assert_close(predictions, current.network(batch()[0]), rtol=0, atol=0)


def test_leaky_relu_respects_explicit_model_slope_and_rejects_conflicts() -> None:
    current = learner("leaky_relu", model={"negative_slope": 0.2})
    assert current.network.hidden[0].activation.negative_slope == 0.2
    with pytest.raises(ValueError, match="disagree"):
        learner("leaky_relu", model={"negative_slope": 0.2}, method_config={"negative_slope": 0.3})


def test_spectral_gradient_matches_exact_svd_and_includes_bias() -> None:
    problem = ProblemSpec((2,), "mse", (0, 1))
    current = make_model("spectral", "mlp", problem=problem, model_config={"hidden_sizes": []},
                         method_config={"coefficient": 0.2, "power_iterations": 50},
                         optimizer_config={"name": "sgd", "lr": 0.01})
    with torch.no_grad():
        current.network.head.weight.copy_(torch.diag(torch.tensor([2.0, 0.5])))
        current.network.head.bias.copy_(torch.tensor([0.3, -0.4]))
    approximate = current.penalty()
    approximate.backward()
    weight = current.network.head.weight.detach().clone().requires_grad_()
    bias = current.network.head.bias.detach().clone().requires_grad_()
    exact = 0.2 * ((torch.linalg.svdvals(weight)[0] ** 2 - 1) ** 2 + bias.square().sum() ** 2)
    exact.backward()
    torch.testing.assert_close(approximate, exact)
    torch.testing.assert_close(current.network.head.weight.grad, weight.grad)
    torch.testing.assert_close(current.network.head.bias.grad, bias.grad)


def test_spectral_effective_residual_gains_and_embeddings() -> None:
    current = learner("spectral", "vit", model={"ln_gain": "residual"})
    assert "pos_embedding" in current.matrices
    assert "cls_token" in current.vectors
    assert set(current.selected_parameters) == {name for name, _ in current.network.named_parameters()}
    for module in current.gains.values():
        torch.testing.assert_close(module.gain, torch.ones_like(module.weight))
        assert not module.weight.any()


@pytest.mark.parametrize("axes", ["channels", "all_features"])
def test_resnet_layer_norm_replaces_all_batch_norm(axes: str) -> None:
    current = learner("layer_norm", "resnet_18", model={"ln_axes": axes})
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in current.network.modules())
    assert sum(isinstance(module, LayerNorm) for module in current.network.modules()) == 20
    current.train_step(batch())
    inference = make_network("resnet_18", problem=current.problem, model_config=current.resolved_model_config)
    inference.load_state_dict(current.state_dict()["network"])
    inference.eval()
    torch.testing.assert_close(current.predict(batch()[0]), inference(batch()[0]))


def test_leaky_relu_covers_stem_and_both_block_activations() -> None:
    current = learner("leaky_relu", "resnet_18", method_config={"negative_slope": 0.2})
    activations = [module for module in current.network.modules() if isinstance(module, nn.LeakyReLU)]
    assert len(activations) == 17
    assert all(module.negative_slope == 0.2 for module in activations)
    assert not any(isinstance(module, nn.ReLU) for module in current.network.modules())


def test_predict_restores_modes_buffers_and_dropout_rng() -> None:
    current = learner(model={"dropout": 0.5}, seed=8)
    current.network.hidden[0].dropout.eval()
    modes = [module.training for module in current.network.modules()]
    before = current.state_dict()
    current.predict(batch()[0])
    assert modes == [module.training for module in current.network.modules()]
    torch.testing.assert_close(current.state_dict()["rng"]["torch"], before["rng"]["torch"], rtol=0, atol=0)
    restored = learner(model={"dropout": 0.5}, seed=99)
    restored.load_state_dict(before)
    current.train_step(batch())
    restored.train_step(batch())
    for name, value in current.network.state_dict().items():
        torch.testing.assert_close(value, restored.network.state_dict()[name], rtol=0, atol=0)


def test_sliced_initialization_retains_original_fan() -> None:
    current = learner()
    name = "hidden.0.linear.weight"
    rule = current.network.initialization_rules[name]
    assert rule["fan_in"] == 64
    samples = sample_initial(current.network, name, shape=(100000,), generator=torch.Generator().manual_seed(4))
    assert samples.abs().max() <= rule["high"]
    assert abs(samples.std().item() - (2 / 64) ** 0.5) < 0.002


@pytest.mark.parametrize("architecture,model", [("mlp", {"nonsense": 2}), ("mlp", {"hidden_sizes": [0]}),
    ("mlp", {"norm": "none", "norm_position": "post_activation"}),
    ("resnet_18", {"norm": "batch", "ln_axes": "all_features"}),
    ("vit", {"embed_dim": 10, "num_heads": 3}), ("vit", {"head_dim": 3})])
def test_invalid_or_ignored_architecture_fields_fail(architecture: str, model: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        learner(architecture=architecture, model=model)


def test_ordinary_learner_fits_stationary_regression() -> None:
    problem = ProblemSpec((2,), "mse", (0,))
    current = make_model("backprop", "mlp", problem=problem, model_config={"hidden_sizes": []},
                         optimizer_config={"name": "sgd", "lr": 0.1})
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    targets = x[:, :1] * 2 - x[:, 1:] + 0.5
    for _ in range(60):
        current.train_step((x, targets, torch.arange(4)))
    assert nn.functional.mse_loss(current.predict(x), targets).item() < 1e-9
