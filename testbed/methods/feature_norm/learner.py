from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor, nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import Batch, StepResult
from testbed.methods.backprop.learner import BackpropLearner

from .config import FeatureNormConfig


class FeatureNormLearner(BackpropLearner):
    def __init__(
        self,
        *,
        config: FeatureNormConfig,
        forward_features: Callable[..., tuple[Tensor, dict[str, Tensor]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config, self.forward_features = config, forward_features
        self.selected_sites = list(config.feature_sites)

    def penalty(self, features: Mapping[str, Tensor]) -> Tensor:
        if self.config.reduction == "mean":
            values = [h.square().mean() for h in features.values()]
        else:
            values = [h.square().flatten(1).sum(dim=1).mean() for h in features.values()]
        return self.config.coefficient * sum(values)

    def train_step(self, batch: Batch) -> StepResult:
        x, targets, _ = batch
        with self.rng:
            lr = set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions, features = self.forward_features(self.network, x.to(self.device), self.config.feature_sites)
            loss = supervised_loss(predictions, targets.to(self.device), self.problem)
            extra = self.penalty(features)
            (loss + extra).backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
        return StepResult(predictions.detach(), loss.detach(), extra.detach(), {"lr": lr})

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict() | {"selected_sites": self.selected_sites}
