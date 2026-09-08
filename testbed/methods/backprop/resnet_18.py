from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from testbed.core.factory import make_network
from testbed.core.types import ProblemSpec
from testbed.models.config import ResNet18Config

from .config import BackpropConfig
from .learner import BackpropLearner


def build(
    *,
    problem: ProblemSpec,
    model_config: ResNet18Config | Mapping[str, Any],
    method_config: BackpropConfig,
    device: str | torch.device = "cpu",
    seed: int = 0,
    method_seed: int | None = None,
    **kwargs: Any,
) -> BackpropLearner:
    network = make_network("resnet_18", problem=problem, model_config=model_config, device=device, seed=seed)
    return BackpropLearner(network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
