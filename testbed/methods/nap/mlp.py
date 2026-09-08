from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch import nn

from testbed.core.factory import make_network
from testbed.core.optim import OptimizerConfig
from testbed.core.types import ProblemSpec
from testbed.models.config import MLPConfig

from .config import CONFIG_CLASS, NaPConfig
from .learner import NaPLearner


def build(
    *,
    problem: ProblemSpec,
    model_config: MLPConfig | Mapping[str, Any],
    method_config: NaPConfig | Mapping[str, Any],
    optimizer_config: OptimizerConfig | Mapping[str, Any],
    lr_schedule: str | Mapping[str, Any] | None = "constant",
    grad_clip_norm: float | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    start_update: int = 0,
    method_seed: int | None = None,
) -> NaPLearner:
    config = method_config if isinstance(method_config, CONFIG_CLASS) else CONFIG_CLASS(**method_config)
    model_config = asdict(model_config) if is_dataclass(model_config) else dict(model_config)
    model_config.update(norm="layer", bias=False)
    network = make_network("mlp", problem=problem, model_config=model_config, device=device, seed=seed)
    scope = config.parameter_scope
    modules = dict(network.named_modules())
    if isinstance(scope, str) and scope not in {"network", "hidden", "head"}:
        raise ValueError("unknown NaP parameter_scope")
    if not isinstance(scope, str) and (not isinstance(scope, (list, tuple)) or not scope or any(name not in modules for name in scope)):
        raise ValueError("NaP scope contains an unknown module")
    selected = {f"{name}.weight": module.weight for name, module in modules.items()
                if isinstance(module, (nn.Linear, nn.Conv2d)) and
                (scope == "network" or scope == "head" and module is network.head
                 or scope == "hidden" and module is not network.head
                 or not isinstance(scope, str) and any(name == s or name.startswith(s + ".") for s in scope))}
    if not selected:
        raise ValueError("NaP scope selects no weights")
    return NaPLearner(network=network, config=config, selected_weights=selected, problem=problem,
                         optimizer_config=optimizer_config, lr_schedule=lr_schedule, grad_clip_norm=grad_clip_norm,
                         seed=seed if method_seed is None else method_seed, start_update=start_update)
