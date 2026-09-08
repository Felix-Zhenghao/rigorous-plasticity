from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

from torch import Tensor, nn

from .config import MLPConfig
from .layers import HiddenLayer

if TYPE_CHECKING:
    from testbed.core.types import ProblemSpec

    from .initialization import InitializationRule


class MLP(nn.Module):
    initialization_rules: dict[str, InitializationRule]
    resolved_config: dict[str, object]
    architecture: str

    def __init__(self, problem: ProblemSpec, config: MLPConfig) -> None:
        super().__init__()
        width = prod(problem.input_shape)
        self.hidden = nn.ModuleList()
        self.head_hidden = nn.ModuleList()
        self.norm_position = config.norm_position
        for container, sizes in ((self.hidden, config.hidden_sizes), (self.head_hidden, config.head_hidden_sizes)):
            for size in sizes:
                container.append(HiddenLayer(width, size, config))
                width = size
        self.head_input_dim = width
        self.head = nn.Linear(width, len(problem.output_ids), bias=config.head_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        for layer in self.hidden:
            x = layer(x)
        for layer in self.head_hidden:
            x = layer(x)
        return self.head(x)


def build(problem: ProblemSpec, config: MLPConfig) -> MLP:
    return MLP(problem, config)
