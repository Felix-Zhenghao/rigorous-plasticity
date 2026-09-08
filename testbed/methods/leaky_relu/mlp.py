from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import torch

from testbed.core.factory import make_network
from testbed.core.types import ProblemSpec
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models.config import MLPConfig

from .config import LeakyReLUConfig


def build(
    *,
    problem: ProblemSpec,
    model_config: MLPConfig | Mapping[str, Any],
    method_config: LeakyReLUConfig,
    device: str | torch.device = "cpu",
    seed: int = 0,
    method_seed: int | None = None,
    **kwargs: Any,
) -> BackpropLearner:
    values = asdict(model_config) if is_dataclass(model_config) else dict(model_config)
    values["activation"] = "leaky_relu"
    if "negative_slope" in values and method_config.negative_slope != 0.01 and values["negative_slope"] != method_config.negative_slope:
        raise ValueError("model and method negative_slope disagree")
    values.setdefault("negative_slope", method_config.negative_slope)
    network = make_network("mlp", problem=problem, model_config=values, device=device, seed=seed)
    return BackpropLearner(network=network, problem=problem, seed=seed if method_seed is None else method_seed, **kwargs)
