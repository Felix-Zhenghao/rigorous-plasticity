from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from testbed.core.factory import make_network
from testbed.core.types import ProblemSpec
from testbed.models.config import ViTConfig

from .config import L2Config
from .learner import L2Learner


def build(
    *,
    problem: ProblemSpec,
    model_config: ViTConfig | Mapping[str, Any],
    method_config: L2Config,
    device: str | torch.device = "cpu",
    seed: int = 0,
    method_seed: int | None = None,
    **kwargs: Any,
) -> L2Learner:
    network = make_network("vit", problem=problem, model_config=model_config, device=device, seed=seed)
    return L2Learner(config=method_config, network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
