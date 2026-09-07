from copy import deepcopy

import torch

from testbed.core.types import Consumption, DataBlock, ProblemSpec
from testbed.data.datasets import load_dataset, prepare_data
from testbed.data.sampling import probabilities, realized_proportions, sample_indices, scheduled

from .config import ClassRemapConfig
from .data import FixedRegressionData, RemappedData, draw_base_targets


class ClassRemap:
    def __init__(self, config, *, data_root="data", seed=0, datasets=None):
        self.config = config if isinstance(config, ClassRemapConfig) else ClassRemapConfig(**config)
        config = self.config
        self.seed = seed
        bundle = (datasets.get(config.dataset) if isinstance(datasets, dict) else datasets)
        self.bundle = bundle or load_dataset(config.dataset, data_root, seed=seed, options=config.data_options)
        self.splits, self.metadata = prepare_data(self.bundle, config, seed=seed)
        self.output_ids = self.bundle.output_ids
        if not set(config.stable_classes) <= set(self.output_ids):
            raise ValueError("stable_classes contains labels outside dataset metadata")
        self.pool = self._select_pool(0)
        self.mappings = [self._mapping(i) for i in range(min(config.num_tasks, config.recurrence_period or config.num_tasks))]
        shape = tuple(self.splits["train"][0][0].shape)
        regression = config.target_mode == "fixed_regression"
        self.problem = ProblemSpec(shape, "mse" if regression else "cross_entropy", (0,) if regression else self.output_ids)
        self.next_task, self.active_task = 0, 0
        self.metadata.update(pool_ids=self.splits["train"].ids[self.pool], mappings=deepcopy(self.mappings), tasks=[])
        self.fixed_regression = self._regression_artifact() if regression else None
        self._fixed_eval = {}
        for i in range(config.num_tasks):
            probabilities(config.class_probs, i, config.num_tasks, self.output_ids)

    def _rng(self, purpose, index=0):
        return torch.Generator().manual_seed((self.seed + purpose * 1000003 + index * 9176) % (2**63 - 1))

    def _select_pool(self, index):
        size = len(self.splits["train"])
        requested = self.config.pool_size or size
        return torch.randperm(size, generator=self._rng(1, index))[:min(requested, size)]

    def _mapping(self, index):
        eligible = [label for label in self.output_ids if label not in self.config.stable_classes]
        mapping = {label: label for label in self.output_ids}
        if index != 0 or self.config.first_mapping != "identity":
            order = torch.randperm(len(eligible), generator=self._rng(2, index)).tolist()
            mapping.update({label: eligible[j] for label, j in zip(eligible, order)})
        return mapping

    def _regression_artifact(self):
        config = self.config
        count = scheduled(config.task_samples, 0, 1, "task_samples")
        count = len(self.pool) if count == "pool" else count
        chunk = scheduled(config.chunk_size, 0, 1, "chunk_size")
        if count != len(self.pool) or (chunk != "task" and chunk != count):
            raise ValueError("Fixed regression requires the complete fixed pool in one chunk (M=N)")
        indices = sample_indices(self.pool, self.splits["train"].targets, count, config.sampling, None, self.output_ids, self._rng(3))
        rows = [self.splits["train"][int(i)] for i in indices]
        inputs, ids = torch.stack([row[0] for row in rows]), torch.stack([row[2] for row in rows])
        generator = {"target_family": config.target_family, "target_scale": config.target_scale,
                     "center_targets": config.center_targets, "teacher": deepcopy(config.teacher),
                     "omega": config.omega, "seed": self.seed + 4000012}
        base = draw_base_targets(inputs, generator, seed=generator["seed"])
        center = base.mean(0, keepdim=True) if config.center_targets else torch.zeros((1, 1))
        residuals = (base - center) * config.target_scale
        return {"inputs": inputs, "example_ids": ids, "residuals": residuals,
                "targets": config.target_mean + residuals, "target_mean": config.target_mean,
                "generator": generator, "centering_mean": center}

    def get_data(self):
        config, index = self.config, self.next_task
        if index >= config.num_tasks:
            return None
        pool = self._select_pool(index) if config.pool_refresh == "per_task" else self.pool
        count = scheduled(config.task_samples, index, config.num_tasks, "task_samples")
        count = len(pool) if count == "pool" else count
        mapping = self.mappings[index % len(self.mappings)]
        if self.fixed_regression is not None:
            artifact = self.fixed_regression
            data = FixedRegressionData(artifact["inputs"], artifact["targets"], artifact["example_ids"])
            proportions = None
        else:
            p = probabilities(config.class_probs, index, config.num_tasks, self.output_ids)
            indices = sample_indices(pool, self.splits["train"].targets, count, config.sampling, p, self.output_ids, self._rng(3, index))
            data = RemappedData(self.splits["train"], indices, mapping, augment=True)
            proportions = realized_proportions(self.splits["train"].targets[indices], self.output_ids)
        chunk = scheduled(config.chunk_size, index, config.num_tasks, "chunk_size")
        consume = Consumption(len(data) if chunk == "task" else chunk,
                              scheduled(config.epochs, index, config.num_tasks, "epochs"),
                              scheduled(config.updates, index, config.num_tasks, "updates"), config.shuffle_each_epoch)
        self.active_task, self.next_task = index, index + 1
        self.metadata["tasks"].append({"arrival_count": len(data), "realized_class_proportions": proportions,
                                       "changed_fraction": sum(c != y for c, y in mapping.items()) / len(mapping)})
        return DataBlock(data, consume)

    def get_unseen_data(self, split):
        if split not in {"val", "test"}:
            raise ValueError("Held-out split must be val or test")
        return self.splits[split]

    def get_eval_data(self, split):
        source = self.get_unseen_data(split)
        if self.fixed_regression is None:
            return RemappedData(source, torch.arange(len(source)), self.mappings[self.active_task % len(self.mappings)])
        if split not in self._fixed_eval:
            inputs = torch.stack([source[i][0] for i in range(len(source))]) if len(source) else torch.empty((0, *self.problem.input_shape))
            generator = self.fixed_regression["generator"]
            seed = generator["seed"] + (1 if split == "val" else 2) if generator["target_family"] == "iid_normal" else generator["seed"]
            base = draw_base_targets(inputs, generator, seed=seed) if len(source) else torch.empty((0, 1))
            residuals = (base - self.fixed_regression["centering_mean"]) * generator["target_scale"]
            self._fixed_eval[split] = FixedRegressionData(inputs, self.config.target_mean + residuals, source.ids)
        return self._fixed_eval[split]

    def state_dict(self):
        return {"seed": self.seed, "next_task": self.next_task, "active_task": self.active_task,
                "pool": self.pool, "mappings": self.mappings,
                "metadata": {**self.metadata, "tasks": list(self.metadata["tasks"])},
                "fixed_regression": self.fixed_regression}

    def load_state_dict(self, state):
        if state.get("seed", self.seed) != self.seed:
            raise ValueError("The data seed differs from the saved run")
        if not 0 <= state["next_task"] <= self.config.num_tasks:
            raise ValueError("Saved task position is incompatible with this recipe")
        for name in ("next_task", "active_task", "pool", "mappings", "fixed_regression"):
            setattr(self, name, state[name])
        self.metadata = {**state["metadata"], "tasks": list(state["metadata"]["tasks"])}
        for split, ids in self.metadata["split_ids"].items():
            if not torch.equal(ids, self.splits[split].ids):
                raise ValueError("Dataset split membership differs from the saved run")
        self._fixed_eval = {}
