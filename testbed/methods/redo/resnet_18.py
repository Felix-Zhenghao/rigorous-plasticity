from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from testbed.core.factory import make_network
from testbed.core.optim import OptimizerConfig
from testbed.core.types import ProblemSpec
from testbed.methods._typing import RecyclingSite
from testbed.models.config import ResNet18Config
from testbed.models.resnet_18 import ResNet18

from .config import CONFIG_CLASS, ReDoConfig
from .learner import ReDoLearner


def build(
    *,
    problem: ProblemSpec,
    model_config: ResNet18Config | Mapping[str, Any],
    method_config: ReDoConfig | Mapping[str, Any],
    optimizer_config: OptimizerConfig | Mapping[str, Any],
    lr_schedule: str | Mapping[str, Any] | None = "constant",
    grad_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    start_update: int = 0,
    method_seed: int | None = None,
) -> ReDoLearner:
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    sites = bind_sites(network, config)
    return ReDoLearner(network=network, config=config, sites=sites, feature_forward=forward_features, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)


def bind_sites(network: ResNet18, config: ReDoConfig) -> dict[str, RecyclingSite]:
    if getattr(config, "bias_compensation", False):
        raise ValueError("ResNet recycling does not support bias compensation")
    return {f"stages.{i}.{j}.conv1": (block.conv1, block.norm1, block.conv2)
            for i, stage in enumerate(network.stages) for j, block in enumerate(stage)}


def forward_features(network: ResNet18, x: Tensor, statistic_site: str) -> tuple[Tensor, dict[str, Tensor]]:
    features = {}
    x = network.stem_pool(network.stem_activation(network.stem_norm(network.stem_conv(x))))
    for i, stage in enumerate(network.stages):
        for j, block in enumerate(stage):
            residual = block.shortcut(x)
            normalized = block.norm1(block.conv1(x))
            activated = block.activation1(normalized)
            features[f"stages.{i}.{j}.conv1"] = (activated if statistic_site == "after_activation" else normalized).detach()
            x = block.activation2(block.sum_norm(block.norm2(block.conv2(activated)) + residual))
    x = x.mean(dim=(2, 3))
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), features
