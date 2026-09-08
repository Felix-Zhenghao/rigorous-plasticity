from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import Batch, StepResult
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models import Network
from testbed.models.initialization import sample_initial

from .config import ShrinkPerturbConfig


def select_parameters(network: Network, config: ShrinkPerturbConfig) -> dict[str, nn.Parameter]:
    modules = dict(network.named_modules())
    scope = config.parameter_scope
    if isinstance(scope, str) and scope not in {"network", "hidden", "head"}:
        raise ValueError("parameter_scope must be network, hidden, head or a module-name list")
    if not isinstance(scope, str) and (not isinstance(scope, (list, tuple)) or not scope or any(name not in modules for name in scope)):
        raise ValueError("parameter_scope contains an unknown module")
    selected = {}
    for name, parameter in network.named_parameters():
        module_name, _, role = name.rpartition(".")
        module = modules[module_name]
        in_scope = (scope == "network" or scope == "head" and module is network.head
                    or scope == "hidden" and module is not network.head
                    or not isinstance(scope, str) and any(module_name == s or module_name.startswith(s + ".") for s in scope))
        normalization = isinstance(module, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm))
        embedding = isinstance(module, nn.Embedding) or module is network and role in {"cls_token", "pos_embedding"}
        if not in_scope or not parameter.requires_grad:
            continue
        if normalization and not config.include_norm_affine or embedding and not config.include_embeddings:
            continue
        if role == "bias" and not normalization and not config.include_bias:
            continue
        selected[name] = parameter
    if not selected:
        raise ValueError("parameter scope selects no trainable parameters")
    return selected


class ShrinkPerturbLearner(BackpropLearner):
    def __init__(self, *, config: ShrinkPerturbConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.selected_weights = select_parameters(self.network, config)
        self.selected_parameters = list(self.selected_weights)
        self.anchors = {n: p.detach().clone() for n, p in self.selected_weights.items()} if config.noise_source == "saved_init" else {}
        scope = config.parameter_scope
        self.running_norms = [module for name, module in self.network.named_modules()
                              if isinstance(module, nn.modules.batchnorm._BatchNorm) and
                              (scope in ("network", "hidden") or not isinstance(scope, str)
                               and any(name == s or name.startswith(s + ".") for s in scope))]

    @torch.no_grad()
    def intervene(self) -> None:
        for name, parameter in self.selected_weights.items():
            parameter.mul_(self.config.retain)
            if self.config.noise_scale:
                if self.config.noise_source == "saved_init":
                    noise = self.anchors[name]
                elif self.config.noise_source == "gaussian":
                    noise = torch.randn_like(parameter)
                else:
                    noise = sample_initial(self.network, name)
                parameter.add_(noise, alpha=self.config.noise_scale)
            if self.config.optimizer_state == "clear_moments":
                for value in self.optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value.zero_()
        if self.config.optimizer_state == "reset_all":
            self.optimizer.state.clear()
        if self.config.reset_running_stats:
            for module in self.running_norms:
                module.reset_running_stats()

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
            if due:
                self.intervene()
            return StepResult(predictions.detach(), loss.detach(), metrics={"intervention": float(due),
                              "maintenance_seconds": time.perf_counter() - started})

    @property
    def cost_metrics(self) -> dict[str, int]:
        metrics = super().cost_metrics
        metrics["frozen_state_bytes"] = sum(t.numel() * t.element_size() for t in self.anchors.values())
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["shrink_perturb"] = {"anchors": self.anchors, "selected_names": list(self.selected_weights)}
        return deepcopy(state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["shrink_perturb"]["selected_names"] != list(self.selected_weights):
            raise ValueError("S&P parameter scope differs from checkpoint")
        super().load_state_dict(state)
        for name, anchor in self.anchors.items():
            anchor.copy_(state["shrink_perturb"]["anchors"][name])
