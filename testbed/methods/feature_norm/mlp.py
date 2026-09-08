from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from testbed.core.factory import make_network
from testbed.core.types import ProblemSpec
from testbed.models.config import MLPConfig
from testbed.models.mlp import MLP

from .config import FeatureNormConfig
from .learner import FeatureNormLearner


def forward_features(network: MLP, x: Tensor, sites: Sequence[str]) -> tuple[Tensor, dict[str, Tensor]]:
    features = {}
    x = x.flatten(1)
    for group in ("hidden", "head_hidden"):
        for index, layer in enumerate(getattr(network, group)):
            x = layer(x)
            name = f"{group}.{index}"
            if name in sites:
                features[name] = x
    if "head_input" in sites:
        features["head_input"] = x
    return network.head(x), features


def build(
    *,
    problem: ProblemSpec,
    model_config: MLPConfig | Mapping[str, Any],
    method_config: FeatureNormConfig,
    device: str | torch.device = "cpu",
    seed: int = 0,
    method_seed: int | None = None,
    **kwargs: Any,
) -> FeatureNormLearner:
    network = make_network("mlp", problem=problem, model_config=model_config, device=device, seed=seed)
    available = {"head_input"} | {f"{group}.{i}" for group in ("hidden", "head_hidden") for i in range(len(getattr(network, group)))}
    if not set(method_config.feature_sites) <= available:
        raise ValueError(f"unknown feature sites; available: {sorted(available)}")
    return FeatureNormLearner(config=method_config, forward_features=forward_features,
                              network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
