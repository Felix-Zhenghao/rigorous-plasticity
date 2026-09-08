from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from testbed.data.sampling import positive, scheduled, validate_data_fields


@dataclass
class ClassIncrementalConfig:
    """Class, example, or source-to-target availability stages.

    Fields:
        dataset: Real dataset registry name: mnist, fashion_mnist, emnist_balanced,
            cifar10, cifar100, svhn, tiny_imagenet, or a registered real dataset loader.
            Loader-specific settings go in data_options.
        stage_sizes: Cumulative availability at each stage. With progression="classes",
            use positive class counts or "all". With "examples", use positive example
            counts, fractions in (0, 1] of the master pool (rounded down), or "all".
            Resolved sizes must strictly increase. With "transfer", provide exactly two
            sizes, each resolved independently against its source/target master pool.
        task_samples: Arrival draws per stage: a positive count or "pool" for that
            stage's eligible pool size. A list sets one value per stage. This controls
            training exposure separately from stage_sizes, which controls availability.
        chunk_size: Arrivals available together for fitting: a positive count or "task"
            for all task arrivals. A list sets one value per task. Each chunk gets
            its own epochs/updates budget; the last chunk may be smaller.
        progression: "classes" expands available classes; "examples" expands a
            nested example pool while retaining the full output label space; "transfer"
            trains on the source dataset, then switches to target_dataset.
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
        pool_size: Cap on each dataset's fixed master training pool after validation
            splitting; None uses all. Stage availability is selected within that pool.
        pool_refresh: Must be "fixed": nested stages and transfer branches require
            a reproducible master pool for each dataset.
        epochs: Positive passes over each arrival chunk; scalar or one entry per task.
            Set to None wherever updates is set; exactly one budget is required per task.
        updates: Optimizer steps per arrival chunk; scalar or one entry per stage.
            None uses epochs. Only stage 0 may use 0 (with epochs=None), making that
            stage available for the next transition without fitting it.
        sampling: "without_replacement" gives distinct arrivals within a task and
            requires task_samples <= pool size; "with_replacement" allows repeats
            and any positive arrival count.
        shuffle_each_epoch: Whether to reshuffle a chunk at every pass through it.
            False preserves its sampled arrival order on every pass.
        class_probs: None draws uniformly from the eligible pool. Otherwise supply
            probabilities summing to 1 in sorted final output-ID order, or one row per
            stage; unavailable classes need zero probability. Class quotas are rounded
            without replacement. Smooth transitions require None.
        class_order: List of distinct native source class IDs, or "random"/None for
            a seeded random order. "classes" may list a subset; "examples"/"transfer"
            require all source classes. Sets class-stage order and class-grouped arrival order.
        arrival_order: For progression="examples": "iid" uses random pool prefixes;
            "class_ordered" groups examples by class_order before taking prefixes.
        target_dataset: Real dataset registry name for stage 1 of progression="transfer";
            required there and None otherwise. Source and target preprocessing must
            produce the same input shape.
        transition: Scalar or one choice per stage. "abrupt" samples the current pool
            immediately. Other choices mix the previous and expanded pools: each arrival
            uses the expanded pool with probability alpha. "linear" ramps alpha from 0
            to 1; "exponential" uses 1 - transition_gamma**(50*k/transition_chunks);
            "explicit" uses alpha_values. Here k is the one-based arrival chunk.
            Smooth transitions require replacement sampling; stage 0 has no transition
            and transfer permits only "abrupt".
        transition_chunks: Positive duration in arrival chunks, scalar or per stage.
            Smooth stages must contain at least this many chunks; linear needs >=2.
            After the duration, alpha=1. Abrupt stages ignore the duration.
        transition_gamma: Base in (0, 1) for the exponential alpha formula; smaller
            values approach expanded-pool sampling faster. A nondefault value requires
            an exponential transition after stage 0.
        alpha_values: For explicit transitions, one entry per stage: None for stage 0
            and non-explicit stages; otherwise a list of transition_chunks probabilities
            in [0, 1]. 0 draws from the previous pool, 1 from the entire expanded pool.
            Coefficients need not increase; None is required if explicit transitions are unused.
        source_label_map: For transfer, map every native source class ID to a shared
            output ID; None keeps native IDs. Source and target IDs define one output
            space, so equal mapped IDs mean the same prediction class.
        target_label_map: Transfer target equivalent of source_label_map; must cover
            every native target class ID. None keeps target native IDs.
        data_options: Keyword arguments for the real dataset loader. Torchvision
            loaders accept download (True); tiny_imagenet accepts download (False),
            but needs an extracted archive.
            See testbed.data.datasets.load_dataset for constraints and custom loaders.
        target_data_options: Loader keyword arguments for target_dataset in transfer;
            same accepted options as data_options. Must be empty outside transfer.
    """

    dataset: str
    stage_sizes: list[int | float | str]
    task_samples: int | str | list[int | str]
    chunk_size: int | str | list[int | str]
    progression: str = "classes"
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
    class_order: list[int] | str | None = None
    arrival_order: str = "iid"
    target_dataset: str | None = None
    transition: str | list[str] = "abrupt"
    transition_chunks: int | list[int] = 1
    transition_gamma: float = 0.5
    alpha_values: list[list[float] | None] | None = None
    source_label_map: dict[int, int] | None = None
    target_label_map: dict[int, int] | None = None
    data_options: dict[str, Any] = field(default_factory=dict)
    target_data_options: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tasks(self) -> int:
        return len(self.stage_sizes)

    def __post_init__(self) -> None:
        positive(self.num_tasks, "number of stages")
        validate_data_fields(self, self.num_tasks, initial_zero=True)
        if "synthetic" in (self.dataset, self.target_dataset):
            raise ValueError("Class incremental training requires real datasets for source and target")
        if self.progression not in {"classes", "examples", "transfer"}:
            raise ValueError("progression must be classes, examples, or transfer")
        if self.pool_refresh != "fixed":
            raise ValueError("Incremental stage construction requires a fixed master pool")
        if self.arrival_order not in {"iid", "class_ordered"}:
            raise ValueError("arrival_order must be iid or class_ordered")
        if self.progression != "examples" and self.arrival_order != "iid":
            raise ValueError("arrival_order overrides require progression=examples")
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
