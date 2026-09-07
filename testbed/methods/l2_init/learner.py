from torch import nn

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner


def select_parameters(network, config):
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


class L2InitLearner(BackpropLearner):
    def __init__(self, *, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.selected = select_parameters(self.network, config)
        self.selected_parameters = list(self.selected)
        self.anchors = {name: parameter.detach().clone() for name, parameter in self.selected.items()}

    def penalty(self):
        return self.config.coefficient * 0.5 * sum((p - self.anchors[name]).square().sum() for name, p in self.selected.items())

    @property
    def cost_metrics(self):
        return super().cost_metrics | {"frozen_state_bytes": sum(value.numel() * value.element_size() for value in self.anchors.values())}

    def train_step(self, batch):
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

    def state_dict(self):
        state = super().state_dict()
        state["l2_init"] = {name: value.clone() for name, value in self.anchors.items()}
        return state

    def load_state_dict(self, state):
        if state["l2_init"].keys() != self.anchors.keys():
            raise ValueError("L2 Init anchor names differ")
        super().load_state_dict(state)
        for name, anchor in self.anchors.items():
            anchor.copy_(state["l2_init"][name])
