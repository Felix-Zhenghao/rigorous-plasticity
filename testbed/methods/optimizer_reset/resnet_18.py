from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from testbed.core.factory import make_model
from testbed.core.optim import OptimizerConfig
from testbed.core.types import ProblemSpec
from testbed.models.config import ResNet18Config

from .config import OptResetConfig
from .learner import OptimizerResetLearner


def build(
    *,
    problem: ProblemSpec,
    model_config: ResNet18Config | Mapping[str, Any],
    method_config: OptResetConfig | Mapping[str, Any],
    optimizer_config: OptimizerConfig | Mapping[str, Any],
    lr_schedule: str | Mapping[str, Any] | None = "constant",
    grad_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    start_update: int = 0,
    method_seed: int | None = None,
) -> OptimizerResetLearner:
    config = method_config if isinstance(method_config, OptResetConfig) else OptResetConfig(**method_config)
    base = make_model(config.base_method, "resnet_18", problem=problem, model_config=model_config,
                      method_config=config.base_config, optimizer_config=optimizer_config, lr_schedule=lr_schedule,
                      grad_clip_norm=grad_clip_norm, device=device, seed=seed, method_seed=method_seed, start_update=start_update)
    return OptimizerResetLearner(base, config)
