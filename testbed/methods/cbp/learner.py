import time
from copy import deepcopy

import torch
from torch import nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models.initialization import sample_initial


class CBPLearner(BackpropLearner):
    def __init__(self, *, config, sites, feature_forward, **kwargs):
        super().__init__(**kwargs)
        self.config, self.sites, self.feature_forward = config, sites, feature_forward
        self.parameter_names = {id(p): name for name, p in self.network.named_parameters()}
        self.selected_parameters = list(sites)
        self.ages = {name: torch.zeros(incoming.weight.shape[0], dtype=torch.long, device=self.device)
                     for name, (incoming, _, _) in sites.items()}
        self.utilities = {name: torch.zeros_like(age, dtype=torch.float32) for name, age in self.ages.items()}
        self.mean_features = {name: torch.zeros_like(value) for name, value in self.utilities.items()}
        self.fractional_counts = {name: 0.0 for name in sites}
        if config.bias_compensation and any(outgoing.bias is None for _, _, outgoing in sites.values()):
            raise ValueError("bias compensation requires an existing downstream bias")

    @torch.no_grad()
    def update_statistics(self, features):
        selections = {}
        for name, feature in features.items():
            incoming, _, outgoing = self.sites[name]
            axes = (0, *range(2, feature.ndim))
            magnitude = feature.float().abs().mean(dim=axes)
            outgoing_axes = (0, *range(2, outgoing.weight.ndim))
            utility = magnitude * outgoing.weight.detach().float().abs().sum(dim=outgoing_axes)
            self.ages[name].add_(1)
            self.utilities[name].lerp_(utility, 1 - self.config.ema_decay)
            self.mean_features[name].lerp_(feature.float().mean(dim=axes), 1 - self.config.ema_decay)
            mature = torch.where(self.ages[name] > self.config.maturity_updates)[0]
            self.fractional_counts[name] += self.config.replacement_rate * mature.numel()
            count = min(int(self.fractional_counts[name]), mature.numel())
            self.fractional_counts[name] -= count
            corrected = self.utilities[name] / (1 - self.config.ema_decay ** self.ages[name])
            selections[name] = mature[torch.argsort(corrected[mature], stable=True)[:count]]
        if self.config.bias_compensation:
            for name, indices in selections.items():
                _, _, outgoing = self.sites[name]
                means = self.mean_features[name][indices] / (1 - self.config.ema_decay ** self.ages[name][indices])
                outgoing.bias.add_((outgoing.weight[:, indices] * means).sum(dim=1))
        self.replace_units(selections)
        for name, indices in selections.items():
            self.ages[name][indices] = 0
            self.utilities[name][indices] = 0
            self.mean_features[name][indices] = 0
        return sum(indices.numel() for indices in selections.values())

    @torch.no_grad()
    def replace_units(self, selections):
        masks = {}
        def mark(parameter, indices, axis=0):
            if parameter is None:
                return
            mask = masks.setdefault(parameter, torch.zeros_like(parameter, dtype=torch.bool))
            if axis == 0:
                mask[indices] = True
            else:
                mask[:, indices] = True
        for name, indices in selections.items():
            if not indices.numel():
                continue
            incoming, norm, outgoing = self.sites[name]
            mark(incoming.weight, indices)
            mark(incoming.bias, indices)
            mark(outgoing.weight, indices, 1)
            if self.config.bias_compensation:
                mark(outgoing.bias, torch.arange(outgoing.bias.numel(), device=self.device))
            if isinstance(norm, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
                mark(norm.weight, indices)
                mark(norm.bias, indices)
        # All incoming resets precede all outgoing zeros, including adjacent sites.
        for name, indices in selections.items():
            if not indices.numel():
                continue
            incoming, norm, outgoing = self.sites[name]
            incoming.weight[indices] = sample_initial(self.network, self.parameter_names[id(incoming.weight)],
                                                       shape=(indices.numel(), *incoming.weight.shape[1:]))
            if incoming.bias is not None:
                incoming.bias[indices] = 0
            if isinstance(norm, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
                if norm.weight is not None:
                    norm.weight[indices] = 0 if getattr(norm, "gain_mode", "standard") == "residual" else 1
                if norm.bias is not None:
                    norm.bias[indices] = 0
                if isinstance(norm, nn.modules.batchnorm._BatchNorm) and norm.track_running_stats:
                    norm.running_mean[indices] = 0
                    norm.running_var[indices] = 1
        for name, indices in selections.items():
            self.sites[name][2].weight[:, indices] = 0
        if self.config.optimizer_state == "clear_moments":
            for parameter, mask in masks.items():
                for value in self.optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value.masked_fill_(mask, 0)
        elif self.config.optimizer_state == "reset_all" and any(index.numel() for index in selections.values()):
            self.optimizer.state.clear()

    def train_step(self, batch):
        with self.rng:
            x, targets, _ = batch
            lr = set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions, features = self.feature_forward(self.network, x.to(self.device), self.config.statistic_site)
            loss = supervised_loss(predictions, targets.to(self.device), self.problem)
            loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
            started = time.perf_counter()
            replaced = self.update_statistics(features)
            return StepResult(predictions.detach(), loss.detach(), metrics={"lr": lr, "replaced_units": float(replaced),
                              "maintenance_seconds": time.perf_counter() - started})

    @property
    def cost_metrics(self):
        metrics = super().cost_metrics
        metrics["frozen_state_bytes"] = sum(t.numel() * t.element_size() for values in (self.ages, self.utilities, self.mean_features) for t in values.values())
        return metrics

    def state_dict(self):
        state = super().state_dict()
        state["cbp"] = deepcopy({"ages": self.ages, "utilities": self.utilities, "mean_features": self.mean_features,
                                 "fractional_counts": self.fractional_counts, "sites": list(self.sites)})
        return state

    def load_state_dict(self, state):
        saved = state["cbp"]
        if saved["sites"] != list(self.sites):
            raise ValueError("CBP sites differ from checkpoint")
        super().load_state_dict(state)
        for key in ("ages", "utilities", "mean_features"):
            for name, value in getattr(self, key).items():
                value.copy_(saved[key][name])
        self.fractional_counts = dict(saved["fractional_counts"])
