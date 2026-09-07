from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from testbed.core.losses import supervised_loss
from testbed.core.optim import set_learning_rate
from testbed.core.types import StepResult
from testbed.methods.backprop.learner import BackpropLearner


class SpectralLearner(BackpropLearner):
    def __init__(self, *, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.matrices, self.vectors, self.gains = {}, {}, {}
        modules = dict(self.network.named_modules())
        scope = config.parameter_scope
        if not isinstance(scope, str) and any(name not in modules for name in scope):
            raise ValueError("parameter_scope contains an unknown module")
        for module_name, module in modules.items():
            is_head = module is self.network.head
            if scope == "head" and not is_head or scope == "hidden" and is_head:
                continue
            if not isinstance(scope, str) and not any(module_name == root or module_name.startswith(root + ".") for root in scope):
                continue
            norm = isinstance(module, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm))
            affine = isinstance(module, (nn.Linear, nn.Conv2d))
            for local, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad or norm and not config.include_norm_affine:
                    continue
                if not affine and not norm and not config.include_embeddings:
                    continue
                if local == "bias" and not config.include_bias:
                    continue
                name = f"{module_name}.{local}" if module_name else local
                if norm and local == "weight":
                    self.gains[name] = module
                elif affine and local == "weight":
                    self.matrices[name] = (parameter, "affine")
                elif not norm and not affine and parameter.squeeze().ndim >= 2:
                    self.matrices[name] = (parameter, "embedding")
                else:
                    self.vectors[name] = parameter
        self.selected_parameters = list(self.matrices) + list(self.vectors) + list(self.gains)
        if not self.selected_parameters:
            raise ValueError("parameter_scope selects no trainable parameters")
        self.power_vectors = {}
        with self.rng:
            for name in self.matrices:
                matrix = self.matrix(name)
                u = F.normalize(torch.randn(matrix.shape[0], device=self.device), dim=0, eps=config.eps)
                v = F.normalize(torch.randn(matrix.shape[1], device=self.device), dim=0, eps=config.eps)
                self.power_vectors[name] = (u, v)

    def matrix(self, name):
        parameter, role = self.matrices[name]
        if role == "embedding":
            parameter = parameter.squeeze()
        return parameter.reshape(parameter.shape[0], -1)

    @property
    def cost_metrics(self):
        size = sum(value.numel() * value.element_size() for vectors in self.power_vectors.values() for value in vectors)
        return super().cost_metrics | {"frozen_state_bytes": size}

    def penalty(self, update_vectors=True):
        total = torch.zeros((), device=self.device)
        for name in self.matrices:
            matrix = self.matrix(name)
            u, v = self.power_vectors[name]
            if update_vectors:
                with torch.no_grad():
                    for _ in range(self.config.power_iterations):
                        u.copy_(F.normalize(matrix @ v, dim=0, eps=self.config.eps))
                        v.copy_(F.normalize(matrix.T @ u, dim=0, eps=self.config.eps))
            # Only the persistent power iteration is detached. W retains its gradient.
            sigma = u.clone() @ matrix @ v.clone()
            total = total + (sigma.abs().pow(self.config.exponent) - 1).square()
        for parameter in self.vectors.values():
            total = total + parameter.square().sum().pow(self.config.exponent)
        for module in self.gains.values():
            gain = module.gain if hasattr(module, "gain") else module.weight
            total = total + (gain - 1).square().sum()
        return self.config.coefficient * total

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
        return super().state_dict() | {"spectral": deepcopy(self.power_vectors)}

    def load_state_dict(self, state):
        if self.power_vectors.keys() != state["spectral"].keys():
            raise ValueError("spectral layer names differ")
        super().load_state_dict(state)
        for name, (u, v) in self.power_vectors.items():
            u.copy_(state["spectral"][name][0])
            v.copy_(state["spectral"][name][1])
