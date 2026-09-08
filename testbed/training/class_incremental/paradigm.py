from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from testbed.core.types import Consumption, DataBlock, ProblemSpec
from testbed.data.datasets import DatasetInput, load_dataset, prepare_data
from testbed.data.sampling import probabilities, realized_proportions, sample_indices, scheduled

from .config import ClassIncrementalConfig
from .data import IncrementalData, mixed_partitions, mixture_arrivals, resolve_sizes


class ClassIncremental:
    def __init__(
        self, config: ClassIncrementalConfig | dict[str, Any], *, data_root: str | Path = "data",
        seed: int = 0, datasets: DatasetInput = None,
    ) -> None:
        self.config = config if isinstance(config, ClassIncrementalConfig) else ClassIncrementalConfig(**config)
        config, self.seed = self.config, seed
        names = [config.dataset, config.target_dataset] if config.progression == "transfer" else [config.dataset]
        self.bundles, self.sources, source_metadata = [], [], []
        for i, name in enumerate(names):
            bundle = datasets.get(name) if isinstance(datasets, dict) else datasets if i == 0 else None
            options = config.data_options if i == 0 else config.target_data_options
            bundle = bundle or load_dataset(name, data_root, seed=seed + i, options=options)
            splits, metadata = prepare_data(bundle, config, seed=seed + i)
            self.bundles.append(bundle)
            self.sources.append(splits)
            source_metadata.append(metadata)
        self.label_maps = [self._label_map(bundle.output_ids, config.source_label_map if i == 0 else config.target_label_map)
                           for i, bundle in enumerate(self.bundles)]
        self.mapped_labels = []
        for source, mapping in zip(self.sources, self.label_maps):
            native = source["train"].targets
            mapped = native.clone()
            for label, output in mapping.items():
                mapped[native == label] = output
            self.mapped_labels.append(mapped)
        shape = tuple(self.sources[0]["train"][0][0].shape)
        if any(tuple(splits["train"][0][0].shape) != shape for splits in self.sources):
            raise ValueError("Transfer preprocessing must give source and target the same input shape")
        self.master_pools = [self._pool(i) for i in range(len(names))]
        self.class_order = self._class_order()
        self.stage_pools, self.stage_support = self._stages()
        if config.progression == "classes":
            output_ids = tuple(sorted(self.stage_support[-1]))
        else:
            output_ids = tuple(sorted({label for mapping in self.label_maps for label in mapping.values()}))
        self.problem = ProblemSpec(shape, "cross_entropy", output_ids)
        self.next_task, self.active_task = 0, 0
        self.metadata = {"sources": source_metadata, "class_order": self.class_order,
                         "stage_pool_ids": [self.sources[self._source_index(i)]["train"].ids[pool].clone() for i, pool in enumerate(self.stage_pools)],
                         "resolved_stage_sizes": [len(support) if config.progression == "classes" else len(pool)
                                                   for pool, support in zip(self.stage_pools, self.stage_support)],
                         "label_maps": deepcopy(self.label_maps), "tasks": []}
        for i in range(config.num_tasks):
            probabilities(config.class_probs, i, config.num_tasks, self.problem.output_ids)
            self._validate_transition(i)

    def _rng(self, purpose: int, index: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed((self.seed + purpose * 1000003 + index * 9176) % (2**63 - 1))

    @staticmethod
    def _label_map(labels: tuple[int, ...], mapping: dict[int, int] | None) -> dict[int, int]:
        if mapping is None:
            return {label: label for label in labels}
        result = {int(key): int(value) for key, value in mapping.items()}
        if set(result) != set(labels):
            raise ValueError("Transfer label maps must cover every native label exactly")
        return result

    def _pool(self, source: int) -> torch.Tensor:
        size = len(self.sources[source]["train"])
        return torch.randperm(size, generator=self._rng(1, source))[:min(self.config.pool_size or size, size)]

    def _class_order(self) -> list[int]:
        labels = self.bundles[0].output_ids
        if isinstance(self.config.class_order, list):
            order = self.config.class_order
            if not order or len(set(order)) != len(order) or not set(order) <= set(labels):
                raise ValueError("class_order must contain distinct native source classes")
            if self.config.progression != "classes" and set(order) != set(labels):
                raise ValueError("Example/transfer ordering must include every native source class")
            return list(order)
        return [labels[i] for i in torch.randperm(len(labels), generator=self._rng(2)).tolist()]

    def _source_index(self, stage: int) -> int:
        return stage if self.config.progression == "transfer" else 0

    def _stages(self) -> tuple[list[torch.Tensor], list[set[int]]]:
        config, pool, labels = self.config, self.master_pools[0], self.sources[0]["train"].targets
        if config.progression == "transfer":
            pools = []
            for i, value in enumerate(config.stage_sizes):
                count = resolve_sizes([value], len(self.master_pools[i]))[0]
                pools.append(self.master_pools[i][:count])
            return pools, [set(mapping.values()) for mapping in self.label_maps]
        if config.progression == "classes":
            sizes = resolve_sizes(config.stage_sizes, len(self.class_order), classes=True)
            support = [set(self.class_order[:size]) for size in sizes]
            pools = [pool[torch.isin(labels[pool], torch.tensor(sorted(classes)))] for classes in support]
            if any(not len(p) for p in pools):
                raise ValueError("An incremental class pool is empty")
            return pools, support
        sizes = resolve_sizes(config.stage_sizes, len(pool))
        if config.arrival_order == "mixed":
            pools = mixed_partitions(pool, labels, self.class_order, sizes, config.uniform_fraction, self._rng(3))
        else:
            if config.arrival_order == "class_ordered":
                pool = torch.cat([pool[labels[pool] == c] for c in self.class_order])
            else:
                pool = pool[torch.randperm(len(pool), generator=self._rng(3))]
            pools = [pool[:size] for size in sizes]
        return pools, [set(self.bundles[0].output_ids) for _ in sizes]

    def _budget(self, index: int) -> tuple[int, int]:
        config = self.config
        count = scheduled(config.task_samples, index, config.num_tasks, "task_samples")
        count = len(self.stage_pools[index]) if count == "pool" else count
        chunk = scheduled(config.chunk_size, index, config.num_tasks, "chunk_size")
        return count, count if chunk == "task" else chunk

    def _validate_transition(self, index: int) -> None:
        transition = scheduled(self.config.transition, index, self.config.num_tasks, "transition")
        if index == 0 or transition == "abrupt":
            return
        count, chunk = self._budget(index)
        duration = scheduled(self.config.transition_chunks, index, self.config.num_tasks, "transition_chunks")
        if math.ceil(count / chunk) < duration:
            raise ValueError("The stage has fewer arrival chunks than transition_chunks")

    def get_data(self) -> DataBlock | None:
        config = self.config
        while self.next_task < config.num_tasks:
            index = self.next_task
            if scheduled(config.updates, index, config.num_tasks, "updates") != 0:
                break
            self.active_task, self.next_task = index, index + 1
        else:
            return None
        source_index = self._source_index(index)
        source, pool = self.sources[source_index]["train"], self.stage_pools[index]
        count, chunk = self._budget(index)
        transition = scheduled(config.transition, index, config.num_tasks, "transition")
        if index > 0 and transition != "abrupt":
            duration = scheduled(config.transition_chunks, index, config.num_tasks, "transition_chunks")
            indices = mixture_arrivals(self.stage_pools[index - 1], pool, count, chunk, transition, duration,
                                       self._rng(4, index), gamma=config.transition_gamma,
                                       values=config.alpha_values[index] if transition == "explicit" else None)
        else:
            p = probabilities(config.class_probs, index, config.num_tasks, self.problem.output_ids)
            indices = sample_indices(pool, self.mapped_labels[source_index], count, config.sampling, p, self.problem.output_ids, self._rng(4, index))
        data = IncrementalData(source, indices, self.label_maps[source_index], augment=True)
        consume = Consumption(chunk, scheduled(config.epochs, index, config.num_tasks, "epochs"),
                              scheduled(config.updates, index, config.num_tasks, "updates"), config.shuffle_each_epoch)
        self.active_task, self.next_task = index, index + 1
        labels = self.mapped_labels[source_index][indices]
        self.metadata["tasks"].append({"arrival_count": len(data), "realized_class_proportions": realized_proportions(labels, self.problem.output_ids)})
        return DataBlock(data, consume)

    def get_unseen_data(self, split: str) -> IncrementalData:
        if split not in {"val", "test"}:
            raise ValueError("Held-out split must be val or test")
        source = self.sources[self._source_index(self.active_task)][split]
        indices = torch.arange(len(source))
        if self.config.progression == "classes":
            indices = indices[torch.isin(source.targets, torch.tensor(sorted(self.stage_support[-1])))]
        return IncrementalData(source, indices, self.label_maps[self._source_index(self.active_task)])

    def get_eval_data(self, split: str) -> IncrementalData:
        if split not in {"val", "test"}:
            raise ValueError("Held-out split must be val or test")
        source_index = self._source_index(self.active_task)
        source = self.sources[source_index][split]
        indices = torch.arange(len(source))
        if self.config.progression == "classes":
            indices = indices[torch.isin(source.targets, torch.tensor(sorted(self.stage_support[self.active_task])))]
        return IncrementalData(source, indices, self.label_maps[source_index])

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "next_task": self.next_task, "active_task": self.active_task,
                "stage_pools": self.stage_pools, "stage_support": self.stage_support,
                "metadata": {**self.metadata, "tasks": list(self.metadata["tasks"])}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("seed", self.seed) != self.seed:
            raise ValueError("The data seed differs from the saved run")
        if not 0 <= state["next_task"] <= self.config.num_tasks:
            raise ValueError("Saved stage position is incompatible with this recipe")
        for name in ("next_task", "active_task", "stage_pools", "stage_support"):
            setattr(self, name, state[name])
        self.metadata = {**state["metadata"], "tasks": list(state["metadata"]["tasks"])}
        for source, metadata in zip(self.sources, self.metadata["sources"]):
            for split, ids in metadata["split_ids"].items():
                if not torch.equal(ids, source[split].ids):
                    raise ValueError("Dataset split membership differs from the saved run")


def retarget_state(saved_state: dict[str, Any], new_paradigm: ClassIncremental) -> dict[str, Any]:
    """Change an S16 target subset while preserving the source data state.

    The suite checks that source consumption has finished; accepting both the
    pre-block and post-block snapshots permits exact consumer reconstruction.
    """
    if new_paradigm.config.progression != "transfer":
        raise ValueError("Target branches require a transfer paradigm")
    if saved_state["next_task"] not in {0, 1} or saved_state["active_task"] != 0:
        raise ValueError("The source checkpoint has already entered target training")
    fresh = new_paradigm.state_dict()
    if not torch.equal(saved_state["stage_pools"][0], fresh["stage_pools"][0]):
        raise ValueError("A target branch must preserve the source pool")
    if saved_state["metadata"]["label_maps"] != fresh["metadata"]["label_maps"]:
        raise ValueError("A target branch must preserve its full output label mapping")
    for split, ids in saved_state["metadata"]["sources"][0]["split_ids"].items():
        if not torch.equal(ids, fresh["metadata"]["sources"][0]["split_ids"][split]):
            raise ValueError("A target branch must preserve source split membership")
    result = deepcopy(saved_state)
    result["stage_pools"][1] = fresh["stage_pools"][1]
    result["stage_support"][1] = fresh["stage_support"][1]
    metadata = result["metadata"]
    metadata["sources"][1] = fresh["metadata"]["sources"][1]
    metadata["stage_pool_ids"][1] = fresh["metadata"]["stage_pool_ids"][1]
    metadata["resolved_stage_sizes"][1] = fresh["metadata"]["resolved_stage_sizes"][1]
    return result
