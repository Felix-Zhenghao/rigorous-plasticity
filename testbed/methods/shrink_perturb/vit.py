from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from testbed.core.factory import make_network
from testbed.core.optim import OptimizerConfig
from testbed.core.types import ProblemSpec
from testbed.models.config import ViTConfig

from .config import CONFIG_CLASS, ShrinkPerturbConfig
from .learner import ShrinkPerturbLearner


def build(
    *,
    problem: ProblemSpec,
    model_config: ViTConfig | Mapping[str, Any],
    method_config: ShrinkPerturbConfig | Mapping[str, Any],
    optimizer_config: OptimizerConfig | Mapping[str, Any],
    lr_schedule: str | Mapping[str, Any] | None = "constant",
    grad_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    start_update: int = 0,
    method_seed: int | None = None,
) -> ShrinkPerturbLearner:
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    network = make_network("vit", problem=problem, model_config=model_config, device=device, seed=seed)
    return ShrinkPerturbLearner(network=network, config=config, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)
