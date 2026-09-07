"""A lazy chunk sampler with a durable, acknowledged minibatch cursor."""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .random import derive_seed, isolated_rng


@dataclass(frozen=True)
class Cursor:
    start: int = 0
    epoch: int = 0
    offset: int = 0
    updates: int = 0


def schedule(length, consume, batch_size, seed, cursor=Cursor()):
    """Yield index/augmentation-seed pairs, next cursor, and new arrival count."""
    start, epoch, offset, updates = cursor.start, cursor.epoch, cursor.offset, cursor.updates
    while start < length:
        size = min(consume.chunk_size, length - start)
        if size == 1:
            budget = consume.epochs if consume.epochs is not None else consume.updates
            for visit in range(updates, budget):
                final = visit + 1 == budget
                next_cursor = Cursor(start + 1) if final else Cursor(start, visit + 1, 0, visit + 1)
                yield [(start, derive_seed(seed, start, visit))], next_cursor, int(visit == 0)
            start, epoch, offset, updates = start + 1, 0, 0, 0
            continue
        while True:
            if consume.shuffle_each_epoch:
                generator = torch.Generator().manual_seed(derive_seed(seed, start, epoch))
                order = torch.randperm(size, generator=generator)
            else:
                order = range(size)
            while offset < size:
                end = min(offset + batch_size, size)
                indices = [(start + int(i), derive_seed(seed, start + int(i), epoch)) for i in order[offset:end]]
                arrivals = size if epoch == 0 and offset == 0 else 0
                updates += 1
                finished = (consume.updates is not None and updates >= consume.updates) or (
                    consume.epochs is not None and end == size and epoch + 1 >= consume.epochs)
                if finished:
                    next_cursor = Cursor(start + size)
                elif end == size:
                    next_cursor = Cursor(start, epoch + 1, 0, updates)
                else:
                    next_cursor = Cursor(start, epoch, end, updates)
                yield indices, next_cursor, arrivals
                offset = end
                if finished:
                    break
            if finished:
                break
            epoch, offset = epoch + 1, 0
        start, epoch, offset, updates = start + size, 0, 0, 0


class SeededView(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index_seed):
        index, seed = index_seed
        # torch augmentation, numpy policies and Python policies all see the same
        # logical visit regardless of worker assignment or loader prefetch depth.
        with isolated_rng(seed, cuda=False):
            return self.data[index]


class FittingView(Dataset):
    def __init__(self, data, start, end, seed):
        self.data, self.start, self.end, self.seed = SeededView(data), start, end, seed

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        position = self.start + index
        return self.data[position, derive_seed(self.seed, position, 0)]


class BatchSampler:
    def __init__(self, data, consume, batch_size, seed, cursor):
        self.args = (len(data), consume, batch_size, seed, cursor)

    def __iter__(self):
        for indices, _, _ in schedule(*self.args):
            yield indices


class Consumer:
    def __init__(self, paradigm, batch_size, *, num_workers=0, seed=0):
        if batch_size < 1 or num_workers < 0:
            raise ValueError("invalid batch_size or num_workers")
        self.paradigm = paradigm
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.counters = {"arrivals": 0, "training_exposures": 0}
        self.cursor = Cursor()
        self.block = self.before_block = None
        self.block_serial = 0
        self.pending = None
        self.done = False
        self.last_fitting = None
        self._schedule = self._loader = None

    def __iter__(self):
        return self

    def _open(self):
        seed = derive_seed(self.seed, self.block_serial)
        self._schedule = iter(schedule(len(self.block.data), self.block.consume, self.batch_size, seed, self.cursor))
        self._view = SeededView(self.block.data)
        if self.num_workers:
            sampler = BatchSampler(self.block.data, self.block.consume, self.batch_size, seed, self.cursor)
            generator = torch.Generator().manual_seed(seed)
            self._loader = iter(DataLoader(self._view, batch_sampler=sampler, num_workers=self.num_workers,
                                           generator=generator))

    def __next__(self):
        if self.pending is not None:
            raise RuntimeError("commit the completed minibatch before requesting another")
        while not self.done:
            if self.block is None:
                self.before_block = self.paradigm.state_dict()
                self.block = self.paradigm.get_data()
                if self.block is None:
                    self.done = True
                    break
                if len(self.block.data) < 1:
                    raise ValueError("data blocks must be nonempty")
                self.cursor = Cursor()
                self._open()
            try:
                indices, cursor, arrivals = next(self._schedule)
            except StopIteration:
                self.block, self._loader = None, None
                self.block_serial += 1
                continue
            batch = next(self._loader) if self.num_workers else default_collate([self._view[i] for i in indices])
            bounds = (self.cursor.start, min(self.cursor.start + self.block.consume.chunk_size, len(self.block.data)))
            self.pending = (cursor, arrivals, len(indices), bounds)
            return tuple(batch)
        raise StopIteration

    def commit(self):
        if self.pending is None:
            raise RuntimeError("no pending minibatch")
        self.cursor, arrivals, exposures, bounds = self.pending
        self.last_fitting = (self.block.data, *bounds, derive_seed(self.seed, self.block_serial))
        self.counters["arrivals"] += arrivals
        self.counters["training_exposures"] += exposures
        self.pending = None

    def fitting_data(self):
        """An evaluation view of the last consumed chunk, never future inputs."""
        return FittingView(*self.last_fitting) if self.last_fitting is not None else None

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("checkpoint only after commit")
        return deepcopy({"cursor": vars(self.cursor), "counters": self.counters,
                         "before_block": self.before_block if self.block is not None else None,
                         "paradigm_state": self.paradigm.state_dict(), "block_serial": self.block_serial,
                         "done": self.done, "batch_size": self.batch_size, "seed": self.seed})

    def load_state_dict(self, state):
        if state["batch_size"] != self.batch_size or state["seed"] != self.seed:
            raise ValueError("consumer seed and batch size must match on resume")
        self.pending = None
        self.last_fitting = None
        self.cursor = Cursor(**state["cursor"])
        self.counters = dict(state["counters"])
        self.block_serial, self.done = state["block_serial"], state["done"]
        self.before_block = deepcopy(state["before_block"])
        self.block, self._loader = None, None
        if self.before_block is not None:
            self.paradigm.load_state_dict(deepcopy(self.before_block))
            self.block = self.paradigm.get_data()
            if self.block is None:
                raise ValueError("checkpoint could not reconstruct its active data")
            self._open()
        else:
            self.paradigm.load_state_dict(deepcopy(state["paradigm_state"]))
