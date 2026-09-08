from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from testbed.data.sampling import positive, scheduled, validate_data_fields


@dataclass
class PixelPermutationConfig:
    """Successive image tasks with fixed, task-specific pixel permutations.

    Fields:
        dataset: Dataset registry name: synthetic, mnist, fashion_mnist, emnist_balanced,
            cifar10, cifar100, svhn, tiny_imagenet, or a registered custom loader.
            Loader-specific settings go in data_options.
        num_tasks: Positive number of successive tasks; schedules must have this many entries.
        task_samples: Arrival draws per task: a positive count or "pool" for the eligible
            pool size. A list sets one value per task. Repeated draws count as arrivals.
        chunk_size: Arrivals available together for fitting: a positive count or "task"
            for all task arrivals. A list sets one value per task. Each chunk gets
            its own epochs/updates budget; the last chunk may be smaller.
        validation_fraction: Fraction in [0, 1) held out from each native training class;
            floor(class_count * fraction) examples per class become validation data.
            The official test split is separate; 0 disables the validation holdout.
        resize: None preserves input size; an integer makes square images;
            [height, width] sets both dimensions. Uses bilinear resizing before augmentation.
        normalization: "unit_interval" only converts uint8 pixels to floats divided by 255
            (existing floats keep their scale). "dataset_stats" additionally standardizes
            each channel using the resized training split, excluding validation data.
        augmentation: Training-only policy: "none"; "random_crop" (4-pixel zero padding
            then a crop to the input size); "horizontal_flip" (probability 0.5);
            "random_crop_flip" or its alias "cifar" (both). Applied anew on each visit.
        pool_size: Maximum eligible training examples after the validation split;
            None uses all. Larger values are capped at the available split size.
        pool_refresh: "fixed" reuses the eligible pool across tasks; "per_task" redraws
            it from the training split for each task. Task sampling happens afterward.
        epochs: Positive passes over each arrival chunk; scalar or one entry per task.
            Set to None wherever updates is set; exactly one budget is required per task.
        updates: Positive optimizer steps per arrival chunk, cycling through that chunk
            if needed; scalar or one entry per task. None uses epochs instead.
        sampling: "without_replacement" gives distinct arrivals within a task and
            requires task_samples <= pool size; "with_replacement" allows repeats
            and any positive arrival count.
        shuffle_each_epoch: Whether to reshuffle a chunk at every pass through it.
            False preserves its sampled arrival order on every pass.
        class_probs: None samples examples uniformly from the eligible pool. Otherwise
            give nonnegative probabilities summing to 1, in sorted output-class order,
            or one such row per task. With replacement, each draw first samples a class;
            without replacement, rounded class quotas must fit available class counts.
        permutation_axes: "spatial" permutes H*W positions identically in every channel;
            "all_values" permutes C*H*W values and can mix channels. Applied after
            augmentation and before channel normalization.
        permuted_fraction: Fraction in [0, 1] of positions eligible for permutation;
            scalar or one value per task. floor(fraction * position_count) positions
            are shuffled among themselves, so the fraction actually moved can be smaller.
            With recurrence_period set, the fraction schedule must repeat with the bank.
        first_permutation: "identity" leaves the first task unchanged regardless of
            permuted_fraction; "random" applies a seeded partial permutation immediately.
        recurrence_period: None draws a permutation for every task; a positive integer K
            repeats a bank of K permutations (task i uses permutation i % K).
        data_options: Keyword arguments for the dataset loader. synthetic accepts
            input_shape (default [1, 8, 8]), num_classes (4), n_train (256), n_test (64),
            and seed (defaults to the data seed). Torchvision loaders accept download
            (True); tiny_imagenet accepts download (False), but needs an extracted archive.
            See testbed.data.datasets.load_dataset for constraints and custom loaders.
    """

    dataset: str
    num_tasks: int
    task_samples: int | str | list[int | str]
    chunk_size: int | str | list[int | str]
    validation_fraction: float = 0.1
    resize: int | list[int] | None = None
    normalization: str = "unit_interval"
    augmentation: str = "none"
    pool_size: int | None = None
    pool_refresh: str = "fixed"
    epochs: int | list[int | None] | None = 1
    updates: int | list[int | None] | None = None
    sampling: str = "without_replacement"
    shuffle_each_epoch: bool = True
    class_probs: list[float] | list[list[float]] | None = None
    permutation_axes: str = "spatial"
    permuted_fraction: float | list[float] = 1.0
    first_permutation: str = "random"
    recurrence_period: int | None = None
    data_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
