import time
from collections import deque
from copy import deepcopy

import torch
from torch import nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models.initialization import sample_initial


class ReDoLearner(BackpropLearner):
    def __init__(self, *, config, sites, feature_forward, **kwargs):
        super().__init__(**kwargs)
        self.config, self.sites, self.feature_forward = config, sites, feature_forward
        self.parameter_names = {id(p): name for name, p in self.network.named_parameters()}
        self.selected_parameters = list(sites)
        self.windows = {name: deque(maxlen=config.statistics_window_updates) for name in sites}
        self.anchors = {}
        if config.reset_source == "saved_init":
            for incoming, _, _ in sites.values():
                for parameter in (incoming.weight, incoming.bias):
                    if parameter is not None:
                        self.anchors[self.parameter_names[id(parameter)]] = parameter.detach().clone()

    @torch.no_grad()
    def update_statistics(self, features, due):
        selections = {}
        for name, feature in features.items():
            axes = (0, *range(2, feature.ndim))
            total = feature.float().abs().sum(dim=axes)
            count = feature.numel() // feature.shape[1]
            self.windows[name].append((total, count))
            if due:
                window = self.windows[name]
                magnitude = sum(item[0] for item in window) / sum(item[1] for item in window)
                score = magnitude / (magnitude.mean() + self.config.eps)
                selections[name] = torch.where(score <= self.config.threshold)[0]
        if due:
            self.replace_units(selections)
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
            if isinstance(norm, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
                mark(norm.weight, indices)
                mark(norm.bias, indices)
        # All incoming resets precede all outgoing zeros, including adjacent sites.
        for name, indices in selections.items():
            if not indices.numel():
                continue
            incoming, norm, outgoing = self.sites[name]
            weight_name = self.parameter_names[id(incoming.weight)]
            if self.config.reset_source == "saved_init":
                incoming.weight[indices] = self.anchors[weight_name][indices]
                if incoming.bias is not None:
                    incoming.bias[indices] = self.anchors[self.parameter_names[id(incoming.bias)]][indices]
            else:
                incoming.weight[indices] = sample_initial(self.network, weight_name,
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
            due = (self.completed_updates in self.config.at_updates or self.config.every_updates is not None
                   and self.completed_updates % self.config.every_updates == 0)
            replaced = self.update_statistics(features, due)
            return StepResult(predictions.detach(), loss.detach(), metrics={"lr": lr, "replaced_units": float(replaced),
                              "maintenance_seconds": time.perf_counter() - started})

    @property
    def cost_metrics(self):
        metrics = super().cost_metrics
        metrics["frozen_state_bytes"] = sum(t.numel() * t.element_size() for t in self.anchors.values()) + sum(t.numel() * t.element_size() for window in self.windows.values() for t, _ in window)
        return metrics

    def state_dict(self):
        state = super().state_dict()
        state["redo"] = deepcopy({"windows": {name: list(window) for name, window in self.windows.items()},
                                  "anchors": self.anchors, "sites": list(self.sites)})
        return state

    def load_state_dict(self, state):
        saved = state["redo"]
        if saved["sites"] != list(self.sites):
            raise ValueError("ReDo sites differ from checkpoint")
        super().load_state_dict(state)
        self.windows = {name: deque(((total.to(self.device), count) for total, count in window),
                                    maxlen=self.config.statistics_window_updates)
                        for name, window in saved["windows"].items()}
        for name, value in self.anchors.items():
            value.copy_(saved["anchors"][name])
