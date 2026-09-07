import math

import torch

from testbed.core.types import Consumption, DataBlock, ProblemSpec
from testbed.data.datasets import load_dataset, prepare_data
from testbed.data.sampling import probabilities, realized_proportions, sample_indices, scheduled

from .config import PixelPermutationConfig
from .data import PermutedData


class PixelPermutation:
    def __init__(self, config, *, data_root="data", seed=0, datasets=None):
        self.config = config if isinstance(config, PixelPermutationConfig) else PixelPermutationConfig(**config)
        config, self.seed = self.config, seed
        bundle = datasets.get(config.dataset) if isinstance(datasets, dict) else datasets
        self.bundle = bundle or load_dataset(config.dataset, data_root, seed=seed, options=config.data_options)
        self.splits, self.metadata = prepare_data(self.bundle, config, seed=seed)
        shape = tuple(self.splits["train"][0][0].shape)
        if len(shape) != 3:
            raise ValueError("Pixel permutations require CHW image inputs")
        self.problem = ProblemSpec(shape, "cross_entropy", self.bundle.output_ids)
        self.pool = self._select_pool(0)
        self.permutations = [self._permutation(i) for i in range(min(config.num_tasks, config.recurrence_period or config.num_tasks))]
        self.next_task, self.active_task = 0, 0
        self.metadata.update(pool_ids=self.splits["train"].ids[self.pool], permutations=self.permutations, tasks=[])
        for i in range(config.num_tasks):
            probabilities(config.class_probs, i, config.num_tasks, self.problem.output_ids)

    def _rng(self, purpose, index=0):
        return torch.Generator().manual_seed((self.seed + purpose * 1000003 + index * 9176) % (2**63 - 1))

    def _select_pool(self, index):
        size = len(self.splits["train"])
        return torch.randperm(size, generator=self._rng(1, index))[:min(self.config.pool_size or size, size)]

    def _permutation(self, index):
        shape = self.problem.input_shape
        count = math.prod(shape[1:] if self.config.permutation_axes == "spatial" else shape)
        permutation = torch.arange(count)
        if index == 0 and self.config.first_permutation == "identity":
            return permutation
        fraction = scheduled(self.config.permuted_fraction, index, self.config.num_tasks, "permuted_fraction")
        rng = self._rng(2, index)
        eligible = torch.randperm(count, generator=rng)[:math.floor(fraction * count)]
        permutation[eligible] = eligible[torch.randperm(len(eligible), generator=rng)]
        return permutation

    def get_data(self):
        config, index = self.config, self.next_task
        if index >= config.num_tasks:
            return None
        pool = self._select_pool(index) if config.pool_refresh == "per_task" else self.pool
        count = scheduled(config.task_samples, index, config.num_tasks, "task_samples")
        count = len(pool) if count == "pool" else count
        p = probabilities(config.class_probs, index, config.num_tasks, self.problem.output_ids)
        indices = sample_indices(pool, self.splits["train"].targets, count, config.sampling, p, self.problem.output_ids, self._rng(3, index))
        permutation = self.permutations[index % len(self.permutations)]
        data = PermutedData(self.splits["train"], indices, permutation, config.permutation_axes, augment=True)
        chunk = scheduled(config.chunk_size, index, config.num_tasks, "chunk_size")
        consume = Consumption(len(data) if chunk == "task" else chunk,
                              scheduled(config.epochs, index, config.num_tasks, "epochs"),
                              scheduled(config.updates, index, config.num_tasks, "updates"), config.shuffle_each_epoch)
        self.active_task, self.next_task = index, index + 1
        self.metadata["tasks"].append({"arrival_count": len(data),
                                       "realized_class_proportions": realized_proportions(self.splits["train"].targets[indices], self.problem.output_ids),
                                       "moved_fraction": float((permutation != torch.arange(len(permutation))).float().mean())})
        return DataBlock(data, consume)

    def get_unseen_data(self, split):
        if split not in {"val", "test"}:
            raise ValueError("Held-out split must be val or test")
        return self.splits[split]

    def get_eval_data(self, split):
        source = self.get_unseen_data(split)
        return PermutedData(source, torch.arange(len(source)), self.permutations[self.active_task % len(self.permutations)], self.config.permutation_axes)

    def state_dict(self):
        return {"seed": self.seed, "next_task": self.next_task, "active_task": self.active_task,
                "pool": self.pool, "permutations": self.permutations,
                "metadata": {**self.metadata, "tasks": list(self.metadata["tasks"])}}

    def load_state_dict(self, state):
        if state.get("seed", self.seed) != self.seed:
            raise ValueError("The data seed differs from the saved run")
        if not 0 <= state["next_task"] <= self.config.num_tasks:
            raise ValueError("Saved task position is incompatible with this recipe")
        for name in ("next_task", "active_task", "pool", "permutations"):
            setattr(self, name, state[name])
        self.metadata = {**state["metadata"], "tasks": list(state["metadata"]["tasks"])}
        for split, ids in self.metadata["split_ids"].items():
            if not torch.equal(ids, self.splits[split].ids):
                raise ValueError("Dataset split membership differs from the saved run")
