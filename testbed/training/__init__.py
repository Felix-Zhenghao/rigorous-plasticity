from .class_incremental import ClassIncremental, ClassIncrementalConfig
from .class_remap import ClassRemap, ClassRemapConfig
from .pixel_permutation import PixelPermutation, PixelPermutationConfig

PARADIGMS = {"class_remap": ClassRemap, "pixel_permutation": PixelPermutation, "class_incremental": ClassIncremental}
CONFIGS = {"class_remap": ClassRemapConfig, "pixel_permutation": PixelPermutationConfig, "class_incremental": ClassIncrementalConfig}


def make_paradigm(name, config, *, data_root="data", seed=0, datasets=None):
    if name not in PARADIGMS:
        raise ValueError(f"Unknown paradigm {name!r}; choose from {sorted(PARADIGMS)}")
    return PARADIGMS[name](config, data_root=data_root, seed=seed, datasets=datasets)
