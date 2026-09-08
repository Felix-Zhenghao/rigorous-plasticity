"""The entire boundary between data generation, consumption, and learning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

Batch = tuple[Tensor, Tensor, Tensor]


class DataclassInstance(Protocol):
    """Structural type for configuration objects accepted by dataclass helpers."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


@dataclass(frozen=True)
class ProblemSpec:
    """Fixed prediction contract shared by data, models, and losses.

    input_shape: Per-example tensor dimensions, excluding the batch dimension.
    loss_kind: "cross_entropy" uses class IDs and logits; "mse" uses floating
        targets with exactly the same shape as predictions (no broadcasting).
    output_ids: Ordered, distinct output labels. Their count sets the output
        width; classification maps each label to its position in this tuple.
    """

    input_shape: tuple[int, ...]
    loss_kind: str
    output_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "output_ids", tuple(self.output_ids))
        if not self.input_shape or any(n <= 0 for n in self.input_shape):
            raise ValueError("input_shape must contain positive dimensions")
        if self.loss_kind not in {"cross_entropy", "mse"}:
            raise ValueError("loss_kind must be cross_entropy or mse")
        if not self.output_ids or len(set(self.output_ids)) != len(self.output_ids):
            raise ValueError("output_ids must be nonempty and unique")

    @property
    def output_dim(self) -> int:
        return len(self.output_ids)


@dataclass(frozen=True)
class Consumption:
    """How to consume one data block; budgets apply independently to each chunk.

    chunk_size: Maximum arriving examples made available together. The final
        chunk may be smaller; minibatches never cross chunk boundaries.
    epochs: Complete passes per chunk. Set to None to use an update budget.
    updates: Optimizer steps per chunk; requires epochs=None. Short passes
        restart until this budget is reached. Set exactly one budget.
    shuffle_each_epoch: Shuffle positions independently at each pass; False
        preserves the arrival order on every pass.
    """

    chunk_size: int
    epochs: int | None = 1
    updates: int | None = None
    shuffle_each_epoch: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if (self.epochs is None) == (self.updates is None):
            raise ValueError("set exactly one of epochs and updates")
        budget = self.epochs if self.epochs is not None else self.updates
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("consumption budget must be a positive integer")


@dataclass(frozen=True)
class DataBlock:
    data: Dataset[Any]
    consume: Consumption


@dataclass
class StepResult:
    predictions: Tensor
    supervised_loss: float | Tensor
    extra_loss: float | Tensor = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class Paradigm(Protocol):
    problem: ProblemSpec

    def get_data(self) -> DataBlock | None: ...
    def get_eval_data(self, split: str) -> Dataset[Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class Learner(Protocol):
    network: nn.Module
    optimizer: Optimizer
    completed_updates: int
    optimizer_config: dict[str, Any]
    lr_schedule: dict[str, Any]
    resolved_model_config: dict[str, Any]
    resolved_method_config: dict[str, Any]

    def train_step(self, batch: Batch) -> StepResult: ...
    def predict(self, x: Tensor) -> Tensor: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
