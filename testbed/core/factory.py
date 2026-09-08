"""Resolve one method/architecture pair; all updates stay inside its learner."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from testbed.core.random import isolated_rng
from testbed.models.config import resolve_config
from testbed.models.initialization import initialize

from .optim import OptimizerConfig
from .types import DataclassInstance, Learner, ProblemSpec

if TYPE_CHECKING:
    from testbed.models import Network
    from testbed.models.config import ModelConfig

METHOD_ARCHITECTURES = {
    "backprop": ("mlp", "resnet_18", "vit"),
    "l2": ("mlp", "resnet_18", "vit"),
    "l2_init": ("mlp", "resnet_18", "vit"),
    "infer": ("mlp", "resnet_18"),
    "feature_norm": ("mlp", "resnet_18"),
    "spectral": ("mlp", "resnet_18", "vit"),
    "shrink_perturb": ("mlp", "resnet_18", "vit"),
    "cbp": ("mlp", "resnet_18", "vit"),
    "redo": ("mlp", "resnet_18", "vit"),
    "swr": ("mlp", "resnet_18", "vit"),
    "fire": ("mlp", "resnet_18", "vit"),
    "layer_norm": ("mlp", "resnet_18", "vit"),
    "leaky_relu": ("mlp", "resnet_18"),
    "nap": ("mlp", "resnet_18"),
    "optimizer_reset": ("mlp", "resnet_18"),
    "c_chain": ("mlp", "resnet_18"),
}


def make_network(
    architecture: str,
    *,
    problem: ProblemSpec,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Network:
    config = resolve_config(architecture, model_config)
    with isolated_rng(seed):
        network = import_module(f"testbed.models.{architecture}").build(problem, config)
        initialize(network, config)
    network.resolved_config = asdict(config)
    network.architecture = architecture
    return network.to(device)


def make_model(
    method: str,
    architecture: str,
    *,
    problem: ProblemSpec,
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    method_config: DataclassInstance | Mapping[str, Any] | None = None,
    optimizer_config: OptimizerConfig | Mapping[str, Any] | None = None,
    lr_schedule: str | Mapping[str, Any] = "constant",
    grad_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    start_update: int = 0,
    method_seed: int | None = None,
) -> Learner:
    if method not in METHOD_ARCHITECTURES or architecture not in METHOD_ARCHITECTURES[method]:
        raise ValueError(f"unsupported method/architecture pair: {method}/{architecture}")
    values = asdict(method_config) if is_dataclass(method_config) else dict(method_config or {})
    if values.pop("name", method) != method:
        raise ValueError("method name and config disagree")
    config_class = import_module(f"testbed.methods.{method}.config").CONFIG_CLASS
    try:
        resolved_method = config_class(**values)
    except TypeError as error:
        raise ValueError(f"invalid {method} config: {error}") from error
    learner = import_module(f"testbed.methods.{method}.{architecture}").build(
        problem=problem, model_config=model_config or {}, method_config=resolved_method,
        optimizer_config=optimizer_config or {"name": "adam", "lr": 0.001},
        lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm, device=device,
        seed=seed, start_update=start_update, method_seed=method_seed)
    learner.resolved_model_config = learner.network.resolved_config
    learner.resolved_method_config = asdict(resolved_method)
    return learner
