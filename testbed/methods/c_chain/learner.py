import time
from collections import OrderedDict, deque
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner


class CChainLearner(BackpropLearner):
    """Lagged prediction matching with bounded, disjoint past-input replay."""

    def __init__(self, *, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.distance == "prediction_ce" and self.problem.loss_kind != "cross_entropy":
            raise ValueError("prediction_ce reference distance requires classification")
        self.distance = ("prediction_ce" if self.problem.loss_kind == "cross_entropy" else "logit_mse") if config.distance == "auto" else config.distance
        self.reference = deepcopy(self.network).requires_grad_(False).eval()
        initial = self.network.state_dict()
        self.history = [{name: value.detach().clone() for name, value in initial.items()}
                        for _ in range(config.reference_lag_updates + 1)]
        self.history_cursor, self.history_count = 0, 1
        parameter = next(self.network.parameters())
        self.inputs = torch.empty((config.buffer_size, *self.problem.input_shape), dtype=parameter.dtype, device="cpu")
        self.recency = OrderedDict()
        self.supervised_history = deque(maxlen=config.loss_window)
        self.reference_history = deque(maxlen=config.loss_window)
        self.observations = 0
        self.coefficient = config.coefficient
        self.last_reference_ids = ()

    def reference_loss(self, supervised_ids):
        self.last_reference_ids = ()
        eligible = [key for key in self.recency if key not in supervised_ids]
        if not eligible or self.history_count <= self.config.reference_lag_updates:
            return next(self.network.parameters()).new_zeros(())
        positions = torch.randperm(len(eligible))[:self.config.reference_batch_size].tolist()
        self.last_reference_ids = tuple(eligible[i] for i in positions)
        slots = [self.recency[key] for key in self.last_reference_ids]
        inputs = self.inputs[slots].to(self.device)
        lagged = self.history[(self.history_cursor - self.config.reference_lag_updates) % len(self.history)]
        self.reference.load_state_dict(lagged)
        modes = {module: module.training for module in self.network.modules()}
        try:
            self.network.eval()
            current = self.network(inputs)
            with torch.no_grad():
                frozen = self.reference(inputs)
        finally:
            for module, mode in modes.items():
                module.training = mode
        if self.distance == "prediction_ce":
            return -(frozen.softmax(-1) * current.log_softmax(-1)).sum(-1).mean()
        return F.mse_loss(current, frozen)

    @torch.no_grad()
    def remember(self, inputs, ids):
        for x, key in zip(inputs.detach().cpu(), ids):
            if key in self.recency:
                slot = self.recency.pop(key)
            elif len(self.recency) == self.config.buffer_size:
                _, slot = self.recency.popitem(last=False)
            else:
                slot = len(self.recency)
            self.inputs[slot].copy_(x)
            self.recency[key] = slot
        self.history_cursor = (self.history_cursor + 1) % len(self.history)
        self.history_count = min(self.history_count + 1, len(self.history))
        for name, value in self.network.state_dict().items():
            self.history[self.history_cursor][name].copy_(value)

    def train_step(self, batch):
        with self.rng:
            inputs, targets, example_ids = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            ids = [int(key) for key in example_ids]
            lr = set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.network(inputs)
            loss = supervised_loss(predictions, targets, self.problem)
            reference_loss = self.reference_loss(set(ids))
            coefficient = self.coefficient
            extra = coefficient * reference_loss
            (loss + extra).backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
            started = time.perf_counter()
            self.remember(inputs, ids)
            self.supervised_history.append(abs(loss.item()))
            if self.last_reference_ids:
                self.reference_history.append(reference_loss.item())
            self.observations += 1
            if (self.config.coefficient_mode == "loss_ratio" and self.observations >= self.config.adapt_after_updates
                    and self.reference_history):
                self.coefficient = self.config.target_loss_ratio * (sum(self.supervised_history) / len(self.supervised_history)) / (
                    sum(self.reference_history) / len(self.reference_history) + self.config.eps)
            return StepResult(predictions.detach(), loss.detach(), extra.detach(), {
                "lr": lr, "reference_loss": reference_loss.item(), "coefficient": coefficient,
                "reference_examples": float(len(self.last_reference_ids)), "warmup": float(not self.last_reference_ids),
                "extra_forwards": 2.0 if self.last_reference_ids else 0.0,
                "reference_buffer_bytes": float(self.inputs.numel() * self.inputs.element_size()),
                "maintenance_seconds": time.perf_counter() - started})

    @property
    def cost_metrics(self):
        metrics = super().cost_metrics
        metrics["frozen_state_bytes"] = sum(t.numel() * t.element_size() for snapshot in self.history for t in snapshot.values()) + sum(t.numel() * t.element_size() for t in self.reference.state_dict().values())
        metrics["reference_buffer_bytes"] = self.inputs.numel() * self.inputs.element_size()
        return metrics

    def state_dict(self):
        state = super().state_dict()
        state["c_chain"] = deepcopy({"history": self.history, "history_cursor": self.history_cursor,
            "history_count": self.history_count, "inputs": self.inputs[:len(self.recency)],
            "recency": list(self.recency.items()), "supervised_history": list(self.supervised_history),
            "reference_history": list(self.reference_history), "observations": self.observations,
            "coefficient": self.coefficient, "last_reference_ids": self.last_reference_ids})
        return state

    def load_state_dict(self, state):
        saved = state["c_chain"]
        if len(saved["history"]) != len(self.history) or len(saved["recency"]) > self.config.buffer_size:
            raise ValueError("C-CHAIN checkpoint history or buffer capacity differs")
        super().load_state_dict(state)
        for snapshot, source in zip(self.history, saved["history"]):
            for name, value in snapshot.items():
                value.copy_(source[name])
        self.history_cursor, self.history_count = saved["history_cursor"], saved["history_count"]
        self.inputs[:len(saved["recency"])].copy_(saved["inputs"])
        self.recency = OrderedDict(saved["recency"])
        self.supervised_history = deque(saved["supervised_history"], maxlen=self.config.loss_window)
        self.reference_history = deque(saved["reference_history"], maxlen=self.config.loss_window)
        self.observations, self.coefficient = saved["observations"], saved["coefficient"]
        self.last_reference_ids = tuple(saved["last_reference_ids"])
