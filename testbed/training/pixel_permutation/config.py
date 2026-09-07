from dataclasses import dataclass, field

from testbed.data.sampling import positive, scheduled, validate_data_fields


@dataclass
class PixelPermutationConfig:
    dataset: str
    num_tasks: int
    task_samples: int | str | list
    chunk_size: int | str | list
    validation_fraction: float = 0.1
    resize: int | list[int] | None = None
    normalization: str = "unit_interval"
    augmentation: str = "none"
    pool_size: int | None = None
    pool_refresh: str = "fixed"
    epochs: int | list | None = 1
    updates: int | list | None = None
    sampling: str = "without_replacement"
    shuffle_each_epoch: bool = True
    class_probs: list | None = None
    permutation_axes: str = "spatial"
    permuted_fraction: float | list[float] = 1.0
    first_permutation: str = "random"
    recurrence_period: int | None = None
    data_options: dict = field(default_factory=dict)

    def __post_init__(self):
        positive(self.num_tasks, "num_tasks")
        validate_data_fields(self, self.num_tasks)
        if self.permutation_axes not in {"spatial", "all_values"}:
            raise ValueError("permutation_axes must be spatial or all_values")
        if self.first_permutation not in {"identity", "random"}:
            raise ValueError("first_permutation must be identity or random")
        if self.recurrence_period is not None:
            positive(self.recurrence_period, "recurrence_period")
        fractions = [scheduled(self.permuted_fraction, i, self.num_tasks, "permuted_fraction") for i in range(self.num_tasks)]
        if any(not 0 <= f <= 1 for f in fractions):
            raise ValueError("permuted_fraction must lie in [0, 1]")
        if self.recurrence_period is not None and any(f != fractions[i % self.recurrence_period] for i, f in enumerate(fractions)):
            raise ValueError("permuted_fraction schedules must repeat with the transformation bank")
