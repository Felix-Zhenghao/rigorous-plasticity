from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from testbed.data.sampling import positive, validate_data_fields


@dataclass
class ClassRemapConfig:
    """Class-label remapping or stationary scalar-regression data.

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
            give nonnegative probabilities summing to 1 in sorted native-label order
            before remapping, or one row per task. With replacement, each draw samples a class;
            without replacement, rounded class quotas must fit available class counts.
        stable_classes: Native label IDs that always map to themselves. All other IDs
            are randomly permuted among themselves; a random permutation may retain some IDs.
        first_mapping: "identity" keeps native labels for the first task; "random"
            uses a seeded random permutation immediately. Later mappings are random.
        recurrence_period: None draws a mapping for every task; a positive integer K
            reuses a bank of K mappings cyclically (task i uses mapping i % K).
        target_mode: "native" fits remapped class labels with cross-entropy.
            "fixed_regression" fits scalar targets on one fixed input set with MSE;
            it requires num_tasks=1, first_mapping="identity", pool_refresh="fixed",
            task_samples="pool", chunk_size="task" (or equal numeric sizes),
            augmentation="none", sampling="without_replacement", and class_probs=None.
        target_family: For fixed_regression: "iid_normal" draws independent N(0, 1)
            values; "teacher" uses a randomly initialized scalar-output network;
            "sine_teacher" uses sin(omega * teacher(x)). Values are drawn once per input set.
        target_mean: Additive scalar offset b in y = b + target_scale * (base - center);
            used only for fixed_regression. With center_targets=True, training target mean is b.
        target_scale: Nonnegative multiplier on base residuals for fixed_regression.
            It does not normalize their variance; 0 makes every target equal target_mean.
        center_targets: For fixed_regression, subtract the mean base value over the
            fixed training inputs before scaling. Held-out targets use the same training
            mean. False leaves the base values uncentered.
        teacher: Architecture recipe for teacher/sine_teacher targets; required for
            those families and None for iid_normal. Use {name: mlp|resnet_18|vit,
            model_config: {...}}; architecture aliases name, which defaults to "mlp".
            See the selected model config for architecture options; output width is 1.
        omega: Finite angular multiplier in sin(omega * teacher(x)); only for
            sine_teacher. Higher magnitude produces faster oscillations in teacher output.
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
    stable_classes: list[int] = field(default_factory=list)
    first_mapping: str = "random"
    recurrence_period: int | None = None
    target_mode: str = "native"
    target_family: str = "iid_normal"
    target_mean: float = 0.0
    target_scale: float = 1.0
    center_targets: bool = True
    teacher: dict[str, Any] | None = None
    omega: float = 1e5
    data_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive(self.num_tasks, "num_tasks")
        validate_data_fields(self, self.num_tasks)
        if self.first_mapping not in {"identity", "random"}:
            raise ValueError("first_mapping must be identity or random")
        if self.recurrence_period is not None:
            positive(self.recurrence_period, "recurrence_period")
        if len(set(self.stable_classes)) != len(self.stable_classes):
            raise ValueError("stable_classes must be unique")
        if self.target_mode not in {"native", "fixed_regression"}:
            raise ValueError("Unknown target_mode")
        if self.target_family not in {"iid_normal", "teacher", "sine_teacher"}:
            raise ValueError("Unknown target_family")
        if not all(math.isfinite(v) for v in (self.target_mean, self.target_scale, self.omega)) or self.target_scale < 0:
            raise ValueError("Target parameters must be finite and target_scale nonnegative")
        if self.target_mode == "native":
            inactive = [name for name in ("target_family", "target_mean", "target_scale", "center_targets", "teacher", "omega")
                        if getattr(self, name) != self.__dataclass_fields__[name].default]
            if inactive:
                raise ValueError(f"{', '.join(inactive)} require target_mode=fixed_regression")
        if self.target_mode == "fixed_regression":
            if self.num_tasks != 1 or self.first_mapping != "identity" or self.pool_refresh != "fixed":
                raise ValueError("Fixed regression requires one stationary identity-mapped task and a fixed pool")
            if self.augmentation != "none" or self.sampling != "without_replacement" or self.class_probs is not None:
                raise ValueError("Fixed regression requires no augmentation, no replacement, and empirical sampling")
            if self.target_family != "iid_normal" and self.teacher is None:
                raise ValueError("Teacher target families require a teacher architecture specification")
            if self.target_family == "iid_normal" and self.teacher is not None:
                raise ValueError("teacher applies only to teacher or sine_teacher targets")
            if self.target_family != "sine_teacher" and self.omega != self.__dataclass_fields__["omega"].default:
                raise ValueError("omega applies only to sine_teacher targets")
