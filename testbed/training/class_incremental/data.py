"""Nested availability, paper-style mixed partitions, and arrival mixtures."""

import math

import torch
from torch.utils.data import Dataset


class IncrementalData(Dataset):
    def __init__(self, source, indices, label_map, *, augment=False):
        self.source, self.indices = source, torch.as_tensor(indices, dtype=torch.long)
        self.label_map, self.augment = dict(label_map), augment

    @property
    def ids(self):
        return self.source.ids[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y, source_id = self.source.read(int(self.indices[index]), augment=self.augment)
        return x, self.label_map[int(y)], source_id


def resolve_sizes(values, maximum, *, classes=False):
    resolved = []
    for value in values:
        if value == "all":
            count = maximum
        elif isinstance(value, float) and not classes:
            if not 0 < value <= 1:
                raise ValueError("Fractional stage sizes must lie in (0, 1]")
            count = math.floor(value * maximum)
        elif isinstance(value, int) and not isinstance(value, bool):
            count = value
        else:
            raise ValueError("Stage sizes must be positive totals, fractions for examples/transfer, or all")
        if not 0 < count <= maximum:
            raise ValueError(f"Resolved stage size {count} is outside [1, {maximum}]")
        resolved.append(count)
    if any(b <= a for a, b in zip(resolved, resolved[1:])):
        raise ValueError("Nested stage sizes must be strictly increasing")
    return resolved


def mixed_partitions(pool, labels, class_order, sizes, fraction, generator):
    """P02 Appendix A.3: disjoint IID and class components, then cumulative unions.

    Exact requested totals constrain the partition. An incompatible class grouping
    is rejected rather than moving examples between the two components.
    """
    full_sizes = sizes if sizes[-1] == len(pool) else [*sizes, len(pool)]
    increments = [full_sizes[0], *[b - a for a, b in zip(full_sizes, full_sizes[1:])]]
    shuffled = pool[torch.randperm(len(pool), generator=generator)]
    n_uniform = math.floor(fraction * len(pool))
    uniform, grouped = shuffled[:n_uniform], shuffled[n_uniform:]
    class_groups = torch.tensor_split(torch.tensor(class_order), len(increments))
    partitions, cursor = [], 0
    for size, classes in zip(increments, class_groups):
        fixed = grouped[torch.isin(labels[grouped], classes)]
        missing = size - len(fixed)
        if missing < 0 or cursor + missing > len(uniform):
            raise ValueError("Mixed class partitions cannot realize stage_sizes; change totals or uniform_fraction")
        partitions.append(torch.cat([fixed, uniform[cursor:cursor + missing]]))
        cursor += missing
    if cursor != len(uniform) or sum(map(len, partitions)) != len(pool):
        raise ValueError("Mixed arrival partitions do not cover the master pool")
    return [torch.cat(partitions[:i + 1]) for i in range(len(sizes))]


def mixture_alpha(transition, chunk, duration, *, gamma=0.5, values=None):
    """One-based arrival-chunk coefficient; finite ramps end with full-pool draws."""
    if chunk < 1:
        raise ValueError("Mixture chunk indices start at one")
    if transition == "abrupt" or chunk > duration:
        return 1.0
    if transition == "linear":
        if duration < 2:
            raise ValueError("A linear ramp requires at least two chunks")
        return min((chunk - 1) / (duration - 1), 1.0)
    if transition == "exponential":
        return 1 - gamma ** (50 * chunk / duration)
    if transition == "explicit":
        return values[chunk - 1]
    raise ValueError(f"Unknown transition {transition!r}")


def mixture_arrivals(old, expanded, count, chunk_size, transition, duration, generator, *, gamma=0.5, values=None):
    indices = torch.empty(count, dtype=torch.long)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        alpha = mixture_alpha(transition, start // chunk_size + 1, duration, gamma=gamma, values=values)
        component = torch.rand(end - start, generator=generator) < alpha
        n_new, n_old = int(component.sum()), int((~component).sum())
        arrivals = indices[start:end]
        arrivals[component] = expanded[torch.randint(len(expanded), (n_new,), generator=generator)]
        arrivals[~component] = old[torch.randint(len(old), (n_old,), generator=generator)]
    return indices
