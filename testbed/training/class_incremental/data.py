"""Nested availability and smooth arrival mixtures."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from testbed.data.datasets import PreparedSplit


class IncrementalData(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    def __init__(
        self, source: PreparedSplit, indices: torch.Tensor | Sequence[int],
        label_map: dict[int, int], *, augment: bool = False,
    ) -> None:
        self.source, self.indices = source, torch.as_tensor(indices, dtype=torch.long)
        self.label_map, self.augment = dict(label_map), augment

    @property
    def ids(self) -> torch.Tensor:
        return self.source.ids[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        x, y, source_id = self.source.read(int(self.indices[index]), augment=self.augment)
        return x, self.label_map[int(y)], source_id


def resolve_sizes(values: Sequence[int | float | str], maximum: int, *, classes: bool = False) -> list[int]:
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


def mixture_alpha(
    transition: str, chunk: int, duration: int, *, gamma: float = 0.5,
    values: Sequence[float] | None = None,
) -> float:
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


def mixture_arrivals(
    old: torch.Tensor, expanded: torch.Tensor, count: int, chunk_size: int, transition: str,
    duration: int, generator: torch.Generator, *, sampling: str = "with_replacement", gamma: float = 0.5,
    values: Sequence[float] | None = None,
) -> torch.Tensor:
    """Sample nested-pool mixtures, depleting both pools together without replacement."""
    replacement = sampling == "with_replacement"
    if not replacement:
        if count > len(expanded):
            raise ValueError("task_samples exceeds the expanded pool without replacement")
        added = expanded[~torch.isin(expanded, old)]
        remaining_old = old[torch.randperm(len(old), generator=generator)].tolist()
        remaining_added = added[torch.randperm(len(added), generator=generator)].tolist()

    indices = torch.empty(count, dtype=torch.long)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        alpha = mixture_alpha(transition, start // chunk_size + 1, duration, gamma=gamma, values=values)
        draws = torch.rand(end - start, generator=generator)
        arrivals = indices[start:end]
        if not replacement:
            selected = []
            for draw in draws.tolist():
                n_old = len(remaining_old)
                total = n_old + len(remaining_added)
                # The expanded component also contains every remaining old example.
                p_old = 1 - alpha + alpha * n_old / total if n_old else 0.0
                pool = remaining_old if draw < p_old else remaining_added
                selected.append(pool.pop())
            arrivals[:] = torch.tensor(selected, dtype=torch.long)
            continue
        component = draws < alpha
        n_new, n_old = int(component.sum()), int((~component).sum())
        arrivals[component] = expanded[torch.randint(len(expanded), (n_new,), generator=generator)]
        arrivals[~component] = old[torch.randint(len(old), (n_old,), generator=generator)]
    return indices
