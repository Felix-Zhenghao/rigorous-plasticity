from __future__ import annotations

from pathlib import Path
from typing import Any

from testbed.data.datasets import DatasetInput

from .class_incremental import ClassIncremental, ClassIncrementalConfig
from .class_remap import ClassRemap, ClassRemapConfig
from .pixel_permutation import PixelPermutation, PixelPermutationConfig

PARADIGMS = {"class_remap": ClassRemap, "pixel_permutation": PixelPermutation, "class_incremental": ClassIncremental}
CONFIGS = {"class_remap": ClassRemapConfig, "pixel_permutation": PixelPermutationConfig, "class_incremental": ClassIncrementalConfig}


def make_paradigm(
    name: str, config: ClassRemapConfig | PixelPermutationConfig | ClassIncrementalConfig | dict[str, Any],
    *, data_root: str | Path = "data", seed: int = 0, datasets: DatasetInput = None,
) -> ClassRemap | PixelPermutation | ClassIncremental:
    if name not in PARADIGMS:
        raise ValueError(f"Unknown paradigm {name!r}; choose from {sorted(PARADIGMS)}")
    return PARADIGMS[name](config, data_root=data_root, seed=seed, datasets=datasets)
