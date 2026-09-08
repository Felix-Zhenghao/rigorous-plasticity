from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from testbed.core.factory import make_network
from testbed.core.types import ProblemSpec
from testbed.models.config import MLPConfig
from testbed.models.mlp import MLP

from .config import InFeRConfig
from .learner import InFeRLearner


def forward_features(network: MLP, x: Tensor) -> tuple[Tensor, Tensor]:
    x = x.flatten(1)
    for layer in network.hidden:
        x = layer(x)
    for layer in network.head_hidden:
        x = layer(x)
    return network.head(x), x


def build(
    *,
    problem: ProblemSpec,
    model_config: MLPConfig | Mapping[str, Any],
    method_config: InFeRConfig,
    device: str | torch.device = "cpu",
    seed: int = 0,
    method_seed: int | None = None,
    **kwargs: Any,
) -> InFeRLearner:
    network = make_network("mlp", problem=problem, model_config=model_config, device=device, seed=seed)
    return InFeRLearner(config=method_config, forward_features=forward_features,
                        network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
