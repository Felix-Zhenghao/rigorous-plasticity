import time
from copy import deepcopy
from math import sqrt

import torch
from torch import nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner


class NaPLearner(BackpropLearner):
    def __init__(self, *, config, selected_weights, **kwargs):
        super().__init__(**kwargs)
        self.config, self.selected_weights = config, selected_weights
        self.selected_parameters = list(self.selected_weights)
        self.radii = {name: p.detach().float().norm() * config.radius_multiplier for name, p in selected_weights.items()}
        self.norms = [m for m in self.network.modules() if isinstance(m, nn.LayerNorm) and m.weight is not None]
        if config.affine_policy == "joint_project" and any(isinstance(m, (nn.GELU, nn.Tanh, nn.Sigmoid)) for m in self.network.modules()):
            raise ValueError("NaP joint affine projection requires homogeneous activations")

    @torch.no_grad()
    def intervene(self):
        skipped = 0
        for name, parameter in self.selected_weights.items():
            norm = parameter.float().norm()
            if norm <= self.config.eps:
                skipped += 1
            else:
                parameter.mul_((self.radii[name] / norm).to(parameter.dtype))
        for module in self.norms:
            if self.config.affine_policy == "free":
                continue
            gain = module.gain if hasattr(module, "gain") else module.weight
            if self.config.affine_policy == "init_decay":
                effective = gain * (1 - self.config.affine_decay) + self.config.affine_decay
                if module.bias is not None:
                    module.bias.mul_(1 - self.config.affine_decay)
            else:
                norm = gain.float().square().sum()
                if module.bias is not None:
                    norm += module.bias.float().square().sum()
                norm = norm.sqrt()
                if norm <= self.config.eps:
                    skipped += 1
                    continue
                factor = sqrt(gain.numel()) / norm
                effective = gain * factor
                if module.bias is not None:
                    module.bias.mul_(factor)
            module.weight.copy_(effective - 1 if getattr(module, "gain_mode", "standard") == "residual" else effective)
        return skipped

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
            skipped = self.intervene() if due else 0
            return StepResult(predictions.detach(), loss.detach(), metrics={"intervention": float(due),
                              "projection_skipped": float(skipped), "maintenance_seconds": time.perf_counter() - started})

    @property
    def cost_metrics(self):
        metrics = super().cost_metrics
        metrics["frozen_state_bytes"] = sum(t.numel() * t.element_size() for t in self.radii.values())
        return metrics

    def state_dict(self):
        state = super().state_dict()
        state["nap"] = {"radii": self.radii, "selected_names": list(self.selected_weights)}
        return deepcopy(state)

    def load_state_dict(self, state):
        if state["nap"]["selected_names"] != list(self.selected_weights):
            raise ValueError("NaP parameter scope differs from checkpoint")
        super().load_state_dict(state)
        for name, radius in self.radii.items():
            radius.copy_(state["nap"]["radii"][name])
