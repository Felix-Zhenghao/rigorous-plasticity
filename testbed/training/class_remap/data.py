"""Class-label views and fixed targets from each task's frozen teacher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from testbed.core.types import ProblemSpec
from testbed.data.datasets import PreparedSplit


class RemappedData(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    def __init__(
        self, source: PreparedSplit, indices: torch.Tensor | Sequence[int],
        mapping: dict[int, int], *, augment: bool = False,
    ) -> None:
        self.source, self.indices, self.mapping = source, torch.as_tensor(indices, dtype=torch.long), dict(mapping)
        self.augment = augment

    @property
    def ids(self) -> torch.Tensor:
        return self.source.ids[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        x, y, source_id = self.source.read(int(self.indices[index]), augment=self.augment)
        return x, self.mapping[int(y)], source_id


class FixedRegressionData(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, example_ids: torch.Tensor) -> None:
        self.inputs, self.targets, self.ids = inputs, targets, example_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index], self.ids[index]


def draw_base_targets(inputs: torch.Tensor, generator_config: dict[str, Any], *, seed: int) -> torch.Tensor:
    from testbed.core.factory import make_network

    family = generator_config["target_family"]
    if family not in {"teacher", "sine_teacher"}:
        raise ValueError(f"Unknown target family {family!r}")
    config = dict(generator_config["teacher"])
    architecture = config.pop("name", config.pop("architecture", "mlp"))
    model_config = config.pop("model_config", config)
    problem = ProblemSpec(tuple(inputs.shape[1:]), "mse", (0,))
    teacher = make_network(architecture, problem=problem, model_config=model_config, seed=seed)
    teacher.requires_grad_(False).eval()
    with torch.no_grad():
        values = torch.cat([teacher(batch) for batch in inputs.split(512)]).cpu()
    if values.shape != (len(inputs), 1):
        raise ValueError("Regression teacher must have a scalar output")
    return torch.sin(generator_config["omega"] * values) if family == "sine_teacher" else values


def draw_residuals(inputs: torch.Tensor, generator_config: dict[str, Any], *, seed: int) -> torch.Tensor:
    """Draw one fixed residual vector, with offset-independent random seeds."""
    values = draw_base_targets(inputs, generator_config, seed=seed)
    if generator_config.get("center_targets", True):
        values = values - values.mean(0, keepdim=True)
    return values * generator_config.get("target_scale", 1.0)
