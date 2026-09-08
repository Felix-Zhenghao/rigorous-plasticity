from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from math import sqrt
from typing import Any

import torch
from torch import Tensor, nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import Batch, StepResult
from testbed.methods.backprop.learner import BackpropLearner

from .config import FIREConfig


@torch.no_grad()
def newton_schulz(matrix: Tensor, iterations: int = 10, eps: float = 1e-12) -> tuple[Tensor, int]:
    """Author FIRE iteration, with FP32 arithmetic and a zero-norm guard."""
    x = matrix.float()
    norm = x.norm()
    if norm <= eps:
        return matrix.clone(), 1
    transpose = x.shape[1] > x.shape[0]
    x = x.T if transpose else x
    x = x / norm
    for _ in range(iterations):
        x = 1.5 * x - 0.5 * x @ (x.T @ x)
    return (x.T if transpose else x).to(matrix.dtype), 0


class FIRELearner(BackpropLearner):
    def __init__(
        self, *, config: FIREConfig, selected_weights: Mapping[str, nn.Parameter], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config, self.selected_weights = config, selected_weights
        self.selected_parameters = list(self.selected_weights)

    @torch.no_grad()
    def intervene(self) -> int:
        skipped = 0
        for parameter in self.selected_weights.values():
            scale = sqrt(parameter.shape[0] / parameter.shape[1])
            if parameter.ndim == 4:
                height, width = parameter.shape[2:]
                for i in range(height):
                    for j in range(width):
                        projected, zero = newton_schulz(parameter[:, :, i, j], self.config.iterations, self.config.eps)
                        if not zero:
                            parameter[:, :, i, j].copy_(projected * (scale / (height * width)))
                        skipped += zero
            else:
                projected, zero = newton_schulz(parameter, self.config.iterations, self.config.eps)
                if not zero:
                    parameter.copy_(projected * scale)
                skipped += zero
        if self.config.optimizer_state == "reset_all":
            self.optimizer.state.clear()
        return skipped

    def train_step(self, batch: Batch) -> StepResult:
        with self.rng:
            x, targets, _ = batch
            device = next(self.network.parameters()).device
            x, targets = x.to(device), targets.to(device)
            set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.network(x)
            loss = supervised_loss(predictions, targets, self.problem)
            loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
            started = time.perf_counter()
            due = self.completed_updates in self.config.at_updates or (self.config.every_updates is not None and self.completed_updates % self.config.every_updates == 0)
            skipped = self.intervene() if due else 0
            return StepResult(predictions.detach(), loss.detach(), metrics={"intervention": float(due),
                              "projection_skipped": float(skipped), "maintenance_seconds": time.perf_counter() - started})

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["fire"] = {"selected_names": list(self.selected_weights)}
        return deepcopy(state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["fire"]["selected_names"] != list(self.selected_weights):
            raise ValueError("FIRE parameter scope differs from checkpoint")
        super().load_state_dict(state)
