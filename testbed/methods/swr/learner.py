import math
import time
from copy import deepcopy

import torch
from torch import nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models.initialization import initial_mean, sample_initial


def select_parameters(network, config):
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


class SWRLearner(BackpropLearner):
    def __init__(self, *, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.selected_weights = select_parameters(self.network, config)
        self.selected_parameters = list(self.selected_weights)
        self.utilities = {n: torch.zeros_like(p) for n, p in self.selected_weights.items()} if config.ema_decay else {}

    @torch.no_grad()
    def maintain(self, due):
        replaced = 0
        for name, parameter in self.selected_weights.items():
            if self.config.utility == "gradient" and parameter.grad is None:
                continue
            if not due and not self.config.ema_decay:
                continue
            utility = parameter.abs() if self.config.utility == "magnitude" else (parameter * parameter.grad).abs()
            if self.config.ema_decay:
                utility = self.utilities[name].lerp_(utility, 1 - self.config.ema_decay)
            if not due:
                continue
            values = utility.flatten()
            if self.config.selection == "fraction":
                count = self.config.fraction * values.numel()
                count = math.floor(count) + int(count % 1 > 0 and torch.rand(()).item() < count % 1)
                indices = torch.argsort(values, stable=True)[:count]
            else:
                indices = torch.where(values <= self.config.threshold * values.mean())[0]
            if not indices.numel():
                continue
            flat = parameter.view(-1)
            if self.config.replacement == "init_mean":
                flat[indices] = initial_mean(self.network, name)
            else:
                flat[indices] = sample_initial(self.network, name, shape=(indices.numel(),))
            replaced += indices.numel()
            if self.config.ema_decay:
                self.utilities[name].zero_()
            if self.config.optimizer_state == "clear_moments":
                for value in self.optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value.view(-1)[indices] = 0
        if due and self.config.optimizer_state == "reset_all":
            self.optimizer.state.clear()
        return replaced

    def train_step(self, batch):
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
            replaced = self.maintain(due)
            return StepResult(predictions.detach(), loss.detach(), metrics={"replaced_weights": float(replaced),
                              "maintenance_seconds": time.perf_counter() - started})

    @property
    def cost_metrics(self):
        metrics = super().cost_metrics
        metrics["frozen_state_bytes"] = sum(t.numel() * t.element_size() for t in self.utilities.values())
        return metrics

    def state_dict(self):
        state = super().state_dict()
        state["swr"] = {"utilities": self.utilities, "selected_names": list(self.selected_weights)}
        return deepcopy(state)

    def load_state_dict(self, state):
        if state["swr"]["selected_names"] != list(self.selected_weights):
            raise ValueError("SWR parameter scope differs from checkpoint")
        super().load_state_dict(state)
        for name, utility in self.utilities.items():
            utility.copy_(state["swr"]["utilities"][name])
