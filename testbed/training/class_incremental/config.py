from dataclasses import dataclass, field

from testbed.data.sampling import positive, scheduled, validate_data_fields


@dataclass
class ClassIncrementalConfig:
    dataset: str
    stage_sizes: list[int | float | str]
    task_samples: int | str | list
    chunk_size: int | str | list
    progression: str = "classes"
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
    class_order: list[int] | str | None = None
    arrival_order: str = "iid"
    uniform_fraction: float = 1.0
    target_dataset: str | None = None
    transition: str | list[str] = "abrupt"
    transition_chunks: int | list[int] = 1
    transition_gamma: float = 0.5
    alpha_values: list | None = None
    source_label_map: dict[int, int] | None = None
    target_label_map: dict[int, int] | None = None
    data_options: dict = field(default_factory=dict)
    target_data_options: dict = field(default_factory=dict)

    @property
    def num_tasks(self):
        return len(self.stage_sizes)

    def __post_init__(self):
        positive(self.num_tasks, "number of stages")
        validate_data_fields(self, self.num_tasks, initial_zero=True)
        if self.progression not in {"classes", "examples", "transfer"}:
            raise ValueError("progression must be classes, examples, or transfer")
        if self.pool_refresh != "fixed":
            raise ValueError("Incremental stage construction requires a fixed master pool")
        if self.arrival_order not in {"iid", "class_ordered", "mixed"} or not 0 <= self.uniform_fraction <= 1:
            raise ValueError("Invalid arrival_order or uniform_fraction")
        if self.progression != "examples" and (self.arrival_order != "iid" or self.uniform_fraction != 1):
            raise ValueError("arrival_order and uniform_fraction overrides require progression=examples")
        if self.arrival_order != "mixed" and self.uniform_fraction != 1:
            raise ValueError("uniform_fraction applies only to mixed example arrival")
        if not 0 < self.transition_gamma < 1:
            raise ValueError("transition_gamma must be strictly between zero and one")
        if self.class_order is not None and self.class_order != "random" and not isinstance(self.class_order, list):
            raise ValueError("class_order must be a class-ID list or random")
        if self.progression == "transfer":
            if self.num_tasks != 2 or self.target_dataset is None:
                raise ValueError("Transfer requires target_dataset and exactly two stage_sizes")
        elif self.target_dataset is not None or self.source_label_map is not None or self.target_label_map is not None:
            raise ValueError("Dataset label maps and target_dataset apply only to transfer")
        if self.progression != "transfer" and self.target_data_options:
            raise ValueError("target_data_options applies only to transfer")
        active_transitions = [scheduled(self.transition, i, self.num_tasks, "transition") for i in range(1, self.num_tasks)]
        if "exponential" not in active_transitions and self.transition_gamma != self.__dataclass_fields__["transition_gamma"].default:
            raise ValueError("transition_gamma requires an exponential transition after the first stage")
        if self.alpha_values is not None and "explicit" not in active_transitions:
            raise ValueError("alpha_values requires an explicit transition after the first stage")
        if self.alpha_values is not None and len(self.alpha_values) != self.num_tasks:
            raise ValueError("alpha_values must have one entry per stage (null for an unused stage)")
        for i in range(self.num_tasks):
            transition = scheduled(self.transition, i, self.num_tasks, "transition")
            chunks = scheduled(self.transition_chunks, i, self.num_tasks, "transition_chunks")
            if transition not in {"abrupt", "linear", "exponential", "explicit"}:
                raise ValueError("Unknown transition")
            positive(chunks, "transition_chunks")
            if self.progression == "transfer" and transition != "abrupt":
                raise ValueError("Transfer permits only abrupt transitions")
            if self.alpha_values is not None:
                if (i == 0 or transition != "explicit") and self.alpha_values[i] is not None:
                    raise ValueError("alpha_values entries must be null for stages without an explicit transition")
            if i > 0 and transition != "abrupt":
                if self.sampling != "with_replacement" or self.class_probs is not None:
                    raise ValueError("Smooth mixtures require replacement sampling and class_probs=null")
                if transition == "linear" and chunks < 2:
                    raise ValueError("Linear transitions require at least two chunks")
                if transition == "explicit":
                    if self.alpha_values is None:
                        raise ValueError("alpha_values must have one entry per stage (null for an unused stage)")
                    values = self.alpha_values[i]
                    if values is None or len(values) != chunks or any(not 0 <= v <= 1 for v in values):
                        raise ValueError("Explicit alpha_values must contain one [0, 1] coefficient per transition chunk")
