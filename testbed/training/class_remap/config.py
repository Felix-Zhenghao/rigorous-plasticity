import math
from dataclasses import dataclass, field

from testbed.data.sampling import positive, validate_data_fields


@dataclass
class ClassRemapConfig:
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
    stable_classes: list[int] = field(default_factory=list)
    first_mapping: str = "random"
    recurrence_period: int | None = None
    target_mode: str = "native"
    target_family: str = "iid_normal"
    target_mean: float = 0.0
    target_scale: float = 1.0
    center_targets: bool = True
    teacher: dict | None = None
    omega: float = 1e5
    data_options: dict = field(default_factory=dict)

    def __post_init__(self):
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
