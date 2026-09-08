"""Deterministic sampling primitives; no knowledge of training paradigms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypeVar

import torch

from .datasets import PreprocessingConfig

T = TypeVar("T")


class SamplingConfig(PreprocessingConfig, Protocol):
    """Shared arrival-pool and per-chunk consumption fields."""

    pool_size: int | None
    pool_refresh: str
    sampling: str
    task_samples: int | str | list[int | str]
    chunk_size: int | str | list[int | str]
    epochs: int | list[int | None] | None
    updates: int | list[int | None] | None


def scheduled(value: T | list[T] | tuple[T, ...], index: int, length: int, name: str) -> T:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"{name} must contain exactly {length} entries")
        return value[index]
    return value


def positive(value: object, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")


def validate_data_fields(config: SamplingConfig, length: int, *, initial_zero: bool = False) -> None:
    """Validate the shared vocabulary while keeping configuration ownership local."""
    if not 0 <= config.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if config.pool_size is not None:
        positive(config.pool_size, "pool_size")
    if config.pool_refresh not in {"fixed", "per_task"}:
        raise ValueError("pool_refresh must be fixed or per_task")
    if config.sampling not in {"without_replacement", "with_replacement"}:
        raise ValueError("Invalid sampling policy")
    if config.normalization not in {"unit_interval", "dataset_stats"}:
        raise ValueError("Invalid normalization policy")
    if config.augmentation not in {"none", "random_crop", "horizontal_flip", "random_crop_flip", "cifar"}:
        raise ValueError("Unknown augmentation policy")
    if config.resize is not None:
        resize = [config.resize] if isinstance(config.resize, int) else config.resize
        valid_length = len(resize) == (1 if isinstance(config.resize, int) else 2)
        if not valid_length or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in resize):
            raise ValueError("resize must be a positive integer or [height, width]")
    for index in range(length):
        for key, sentinel in (("task_samples", "pool"), ("chunk_size", "task")):
            value = scheduled(getattr(config, key), index, length, key)
            if value != sentinel:
                positive(value, key)
        epochs = scheduled(config.epochs, index, length, "epochs")
        updates = scheduled(config.updates, index, length, "updates")
        if (epochs is None) == (updates is None):
            raise ValueError("Exactly one of epochs and updates must be set for every task")
        if epochs is not None:
            positive(epochs, "epochs")
        else:
            positive(updates, "updates", zero=initial_zero and index == 0)


def probabilities(
    value: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor | None,
    index: int, length: int, output_ids: Sequence[int],
) -> torch.Tensor | None:
    if value is None:
        return None
    p = torch.as_tensor(value, dtype=torch.double)
    if p.ndim == 2:
        if p.shape[0] != length:
            raise ValueError(f"class_probs must have {length} rows")
        p = p[index]
    if p.ndim != 1 or len(p) != len(output_ids) or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("class_probs must contain one finite nonnegative probability per output class")
    if not math.isclose(float(p.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("class_probs must sum to one")
    return p / p.sum()


def sample_indices(
    pool: torch.Tensor | Sequence[int], labels: torch.Tensor, count: int, sampling: str,
    class_probs: torch.Tensor | None, output_ids: Sequence[int], generator: torch.Generator,
) -> torch.Tensor:
    pool = torch.as_tensor(pool, dtype=torch.long)
    if not len(pool):
        raise ValueError("Cannot sample an empty pool")
    replacement = sampling == "with_replacement"
    if not replacement and count > len(pool):
        raise ValueError("task_samples exceeds the pool without replacement")
    if class_probs is None:
        positions = torch.randint(len(pool), (count,), generator=generator) if replacement else torch.randperm(len(pool), generator=generator)[:count]
        return pool[positions]
    groups = [pool[labels[pool] == label] for label in output_ids]
    if replacement:
        choices = torch.multinomial(class_probs, count, replacement=True, generator=generator)
        result = torch.empty(count, dtype=torch.long)
        for c, group in enumerate(groups):
            positions = torch.where(choices == c)[0]
            if class_probs[c] > 0 and not len(group):
                raise ValueError(f"Positive probability assigned to unavailable class {output_ids[c]}")
            if len(positions):
                result[positions] = group[torch.randint(len(group), (len(positions),), generator=generator)]
        return result
    expected = class_probs * count
    allocations = expected.floor().long()
    remainder = count - int(allocations.sum())
    order = torch.argsort(expected - allocations, descending=True, stable=True)
    allocations[order[:remainder]] += 1
    selected = []
    for label, group, n in zip(output_ids, groups, allocations.tolist()):
        if n > len(group):
            raise ValueError(f"Class {label} has {len(group)} eligible examples but requires {n}")
        selected.append(group[torch.randperm(len(group), generator=generator)[:n]])
    result = torch.cat(selected)
    return result[torch.randperm(count, generator=generator)]


def realized_proportions(labels: torch.Tensor, output_ids: Sequence[int]) -> dict[int, float]:
    positions = torch.searchsorted(torch.tensor(output_ids, dtype=labels.dtype), labels)
    counts = torch.bincount(positions, minlength=len(output_ids))
    return {int(label): int(count) / len(labels) for label, count in zip(output_ids, counts)}
