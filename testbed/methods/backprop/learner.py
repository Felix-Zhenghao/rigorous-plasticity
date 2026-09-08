from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import (
    OptimizerConfig,
    make_optimizer,
    optimizer_dict,
    resolve_schedule,
    set_learning_rate,
)
from testbed.core.random import RNGStream
from testbed.core.types import Batch, ProblemSpec, StepResult
from testbed.models import Network


class BackpropLearner:
    """Plain backpropagation; subclasses may reuse storage and prediction, never hooks."""

    def __init__(
        self,
        *,
        network: Network,
        problem: ProblemSpec,
        optimizer_config: OptimizerConfig | Mapping[str, Any],
        lr_schedule: str | Mapping[str, Any] | None = "constant",
        grad_clip_norm: float | None = None,
        start_update: int = 0,
        seed: int = 0,
        additional_parameters: Iterable[tuple[str, nn.Parameter]] = (),
    ) -> None:
        if type(start_update) is not int or start_update < 0:
            raise ValueError("start_update must be a nonnegative integer")
        if grad_clip_norm is not None and (grad_clip_norm <= 0 or not math.isfinite(grad_clip_norm)):
            raise ValueError("grad_clip_norm must be finite and positive")
        self.network, self.problem = network, problem
        self.optimizer_config = optimizer_dict(optimizer_config)
        self.lr_schedule = resolve_schedule(lr_schedule)
        self.grad_clip_norm = grad_clip_norm
        self.completed_updates = start_update
        self.device = next(network.parameters()).device
        pairs = [(f"network.{name}", value) for name, value in network.named_parameters()]
        pairs.extend(additional_parameters)
        self.optimizer = make_optimizer(pairs, self.optimizer_config)
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.rng = RNGStream(seed, self.device)
        self.resolved_model_config = getattr(network, "resolved_config", {})
        self.resolved_method_config = {}
        self.selected_parameters = []

    @property
    def cost_metrics(self) -> dict[str, int]:
        main = sum(parameter.numel() for parameter in self.network.parameters() if parameter.requires_grad)
        return {"main_trainable_parameters": main,
                "auxiliary_trainable_parameters": sum(parameter.numel() for parameter in self.parameters) - main,
                "frozen_state_bytes": 0, "reference_buffer_bytes": 0}

    def train_step(self, batch: Batch) -> StepResult:
        x, targets, _ = batch
        with self.rng:
            lr = set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.network(x.to(self.device))
            loss = supervised_loss(predictions, targets.to(self.device), self.problem)
            loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
        return StepResult(predictions.detach(), loss.detach(), metrics={"lr": lr})

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        modes = {module: module.training for module in self.network.modules()}
        try:
            self.network.eval()
            return self.network(x.to(self.device))
        finally:
            for module, mode in modes.items():
                module.training = mode

    def state_dict(self) -> dict[str, Any]:
        return deepcopy({"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
                         "optimizer_type": self.optimizer_config["name"], "completed_updates": self.completed_updates,
                         "optimizer_config": self.optimizer_config, "lr_schedule": self.lr_schedule,
                         "rng": self.rng.state_dict(), "resolved_model_config": self.resolved_model_config,
                         "resolved_method_config": self.resolved_method_config,
                         "selected_parameters": self.selected_parameters,
                         "initialization_rules": getattr(self.network, "initialization_rules", {})})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["optimizer_type"] != self.optimizer_config["name"]:
            raise ValueError("checkpoint optimizer type differs")
        groups = state["optimizer"]["param_groups"]
        current = self.optimizer.param_groups
        if len(groups) != len(current) or any(a["param_names"] != b["param_names"] for a, b in zip(groups, current)):
            raise ValueError("checkpoint optimizer parameter names differ")
        for saved_group, group in zip(groups, current):
            for parameter_id, parameter in zip(saved_group["params"], group["params"]):
                for key, value in state["optimizer"]["state"].get(parameter_id, {}).items():
                    if torch.is_tensor(value) and key != "step" and value.shape != parameter.shape:
                        raise ValueError("checkpoint optimizer parameter shape differs")
        self.network.load_state_dict(state["network"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.completed_updates = state["completed_updates"]
        self.rng.load_state_dict(deepcopy(state["rng"]))
