from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from testbed.core.factory import make_network
from testbed.core.optim import OptimizerConfig
from testbed.core.types import ProblemSpec
from testbed.methods._typing import RecyclingSite
from testbed.models.config import ViTConfig
from testbed.models.vit import ViT

from .config import CONFIG_CLASS, CBPConfig
from .learner import CBPLearner


def build(
    *,
    problem: ProblemSpec,
    model_config: ViTConfig | Mapping[str, Any],
    method_config: CBPConfig | Mapping[str, Any],
    optimizer_config: OptimizerConfig | Mapping[str, Any],
    lr_schedule: str | Mapping[str, Any] | None = "constant",
    grad_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    start_update: int = 0,
    method_seed: int | None = None,
) -> CBPLearner:
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("vit", problem=problem, model_config=model_config, device=device, seed=seed)
    sites = bind_sites(network, config)
    return CBPLearner(network=network, config=config, sites=sites, feature_forward=forward_features, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)


def bind_sites(network: ViT, config: CBPConfig) -> dict[str, RecyclingSite]:
    if config.statistic_site != "after_activation":
        raise ValueError("ViT feedforward recycling uses post-GeLU features; there is no hidden-width normalization")
    if getattr(config, "bias_compensation", False):
        raise ValueError("ViT recycling does not support bias compensation")
    return {f"blocks.{i}.fc1": (block.fc1, nn.Identity(), block.fc2) for i, block in enumerate(network.blocks)}


def forward_features(network: ViT, x: Tensor, statistic_site: str) -> tuple[Tensor, dict[str, Tensor]]:
    features = {}
    x = network.embed_tokens(x)
    for i, block in enumerate(network.blocks):
        x = x + block.attention(block.norm1(x))
        activated = block.activation(block.fc1(block.norm2(x)))
        features[f"blocks.{i}.fc1"] = activated.detach().movedim(-1, 1)
        x = x + block.dropout(block.fc2(block.dropout(activated)))
    x = network.norm(x)
    x = x[:, 0] if network.pool == "cls" else x[:, 1:].mean(dim=1)
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), features
