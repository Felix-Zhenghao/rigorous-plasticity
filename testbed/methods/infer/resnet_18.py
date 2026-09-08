from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from testbed.core.factory import make_network
from testbed.core.types import ProblemSpec
from testbed.models.config import ResNet18Config
from testbed.models.resnet_18 import ResNet18

from .config import InFeRConfig
from .learner import InFeRLearner


def forward_features(network: ResNet18, x: Tensor) -> tuple[Tensor, Tensor]:
    x = network.stem_pool(network.stem_activation(network.stem_norm(network.stem_conv(x))))
    for stage in network.stages:
        x = stage(x)
    x = x.mean(dim=(2, 3))
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), x


def build(
    *,
    problem: ProblemSpec,
    model_config: ResNet18Config | Mapping[str, Any],
    method_config: InFeRConfig,
    device: str | torch.device = "cpu",
    seed: int = 0,
    method_seed: int | None = None,
    **kwargs: Any,
) -> InFeRLearner:
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    return InFeRLearner(config=method_config, forward_features=forward_features,
                        network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
