from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.random import derive_seed, isolated_rng
from testbed.core.types import Batch, StepResult
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models import Network

from .config import InFeRConfig


class InFeRLearner(BackpropLearner):
    def __init__(
        self,
        *,
        config: InFeRConfig,
        forward_features: Callable[..., tuple[Tensor, Tensor]],
        network: Network,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.config, self.forward_features = config, forward_features
        device = next(network.parameters()).device
        with isolated_rng(derive_seed(seed, "infer_heads")):
            self.auxiliary = nn.Linear(network.head_input_dim, config.num_heads).to(device)
        self.frozen_network = deepcopy(network).eval().requires_grad_(False)
        self.frozen_network.head = nn.Identity()
        self.frozen_auxiliary = deepcopy(self.auxiliary).eval().requires_grad_(False)
        pairs = [(f"auxiliary.{name}", parameter) for name, parameter in self.auxiliary.named_parameters()]
        super().__init__(network=network, seed=seed, additional_parameters=pairs, **kwargs)
        self.selected_sites = ["head_input"]

    @property
    def cost_metrics(self) -> dict[str, int]:
        frozen_bytes = sum(value.numel() * value.element_size()
                           for module in (self.frozen_network, self.frozen_auxiliary)
                           for value in module.state_dict().values())
        return super().cost_metrics | {"frozen_state_bytes": frozen_bytes}

    def train_step(self, batch: Batch) -> StepResult:
        x, targets, _ = batch
        x = x.to(self.device)
        with self.rng:
            lr = set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.auxiliary.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions, features = self.forward_features(self.network, x)
            loss = supervised_loss(predictions, targets.to(self.device), self.problem)
            with torch.no_grad():
                reference = self.config.target_scale * self.frozen_auxiliary(self.frozen_network(x))
            errors = (self.auxiliary(features) - reference).square()
            extra = self.config.coefficient * (errors.sum(dim=1).mean() if self.config.head_reduction == "sum" else errors.mean())
            (loss + extra).backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
        return StepResult(predictions.detach(), loss.detach(), extra.detach(), {"lr": lr, "extra_forwards": 1})

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict() | {"infer": deepcopy({"auxiliary": self.auxiliary.state_dict(),
                "frozen_network": self.frozen_network.state_dict(), "frozen_auxiliary": self.frozen_auxiliary.state_dict()}),
                "selected_sites": self.selected_sites}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        for name in ("auxiliary", "frozen_network", "frozen_auxiliary"):
            getattr(self, name).load_state_dict(state["infer"][name], strict=True)
