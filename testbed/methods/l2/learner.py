from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import Batch, StepResult
from testbed.methods.backprop.learner import BackpropLearner
from testbed.models import Network

from .config import L2Config


def select_parameters(network: Network, config: L2Config) -> dict[str, nn.Parameter]:
    selected = {}
    modules = dict(network.named_modules())
    scope = config.parameter_scope
    if not isinstance(scope, str) and any(name not in modules for name in scope):
        raise ValueError("parameter_scope contains an unknown module")
    for module_name, module in modules.items():
        is_head = module is network.head
        if scope == "head" and not is_head or scope == "hidden" and is_head:
            continue
        if not isinstance(scope, str) and not any(module_name == root or module_name.startswith(root + ".") for root in scope):
            continue
        norm = isinstance(module, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm))
        affine = isinstance(module, (nn.Linear, nn.Conv2d))
        for local, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or norm and not config.include_norm_affine:
                continue
            if not norm and not affine and not config.include_embeddings:
                continue
            if local == "bias" and not config.include_bias:
                continue
            name = f"{module_name}.{local}" if module_name else local
            selected[name] = parameter
    if not selected:
        raise ValueError("parameter_scope selects no trainable parameters")
    return selected


class L2Learner(BackpropLearner):
    def __init__(self, *, config: L2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.selected = select_parameters(self.network, config)
        self.selected_parameters = list(self.selected)

    def penalty(self) -> Tensor:
        return self.config.coefficient * 0.5 * sum(p.square().sum() for p in self.selected.values())

    def train_step(self, batch: Batch) -> StepResult:
        x, targets, _ = batch
        with self.rng:
            lr = set_learning_rate(self.optimizer, self.optimizer_config, self.lr_schedule, self.completed_updates)
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.network(x.to(self.device))
            loss = supervised_loss(predictions, targets.to(self.device), self.problem)
            extra = self.penalty()
            (loss + extra).backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            self.optimizer.step()
            self.completed_updates += 1
        return StepResult(predictions.detach(), loss.detach(), extra.detach(), {"lr": lr})
