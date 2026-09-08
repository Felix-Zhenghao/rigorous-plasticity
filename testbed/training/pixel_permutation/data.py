from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from testbed.data.datasets import PreparedSplit


class PermutedData(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self, source: PreparedSplit, indices: torch.Tensor | Sequence[int],
        permutation: torch.Tensor, axes: str, *, augment: bool = False,
    ) -> None:
        self.source, self.indices = source, torch.as_tensor(indices, dtype=torch.long)
        self.permutation, self.axes, self.augment = permutation, axes, augment

    @property
    def ids(self) -> torch.Tensor:
        return self.source.ids[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.source.read(int(self.indices[index]), augment=self.augment, permutation=self.permutation, axes=self.axes)
