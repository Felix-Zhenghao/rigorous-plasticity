"""The entire boundary between data generation, consumption, and learning."""
from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

Batch = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class ProblemSpec:
    input_shape: tuple[int, ...]
    loss_kind: str
    output_ids: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "output_ids", tuple(self.output_ids))
        if not self.input_shape or any(n <= 0 for n in self.input_shape):
            raise ValueError("input_shape must contain positive dimensions")
        if self.loss_kind not in {"cross_entropy", "mse"}:
            raise ValueError("loss_kind must be cross_entropy or mse")
        if not self.output_ids or len(set(self.output_ids)) != len(self.output_ids):
            raise ValueError("output_ids must be nonempty and unique")

    @property
    def output_dim(self):
        return len(self.output_ids)


@dataclass(frozen=True)
class Consumption:
    chunk_size: int
    epochs: int | None = 1
    updates: int | None = None
    shuffle_each_epoch: bool = True

    def __post_init__(self):
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if (self.epochs is None) == (self.updates is None):
            raise ValueError("set exactly one of epochs and updates")
        budget = self.epochs if self.epochs is not None else self.updates
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("consumption budget must be a positive integer")


@dataclass(frozen=True)
class DataBlock:
    data: Dataset
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
    def get_eval_data(self, split: str) -> Dataset: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


class Learner(Protocol):
    network: nn.Module
    optimizer: Optimizer
    completed_updates: int

    def train_step(self, batch: Batch) -> StepResult: ...
    def predict(self, x: Tensor) -> Tensor: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
