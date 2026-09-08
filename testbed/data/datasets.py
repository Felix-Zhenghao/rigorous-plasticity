"""Raw storage, stable source identities, and split-local image preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

ImageValue: TypeAlias = torch.Tensor | npt.NDArray[np.generic] | Image.Image
RawStorage: TypeAlias = (
    torch.Tensor | npt.NDArray[np.generic] | Sequence[ImageValue]
    | Dataset[ImageValue] | Dataset[tuple[ImageValue, int]]
)
DataSample: TypeAlias = tuple[torch.Tensor, torch.Tensor | int, torch.Tensor | int]


class PreprocessingConfig(Protocol):
    """Fields consumed by split construction and image preprocessing."""

    validation_fraction: float
    resize: int | list[int] | None
    normalization: str
    augmentation: str


_SOURCE_NAMESPACES: dict[int, str] = {}


def source_ids(name: str, split: str, length: int, version: str = "1") -> torch.Tensor:
    """Encode a dataset/version/split namespace and its original index in int64."""
    if length >= 2**32:
        raise ValueError("A raw split must contain fewer than 2**32 examples")
    namespace = f"{name}:{version}:{split}"
    prefix = int.from_bytes(blake2b(namespace.encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF
    if prefix in _SOURCE_NAMESPACES and _SOURCE_NAMESPACES[prefix] != namespace:
        raise ValueError("Source ID namespace collision; give the custom dataset a distinct name/version")
    _SOURCE_NAMESPACES[prefix] = namespace
    return torch.arange(length, dtype=torch.int64) + (prefix << 32)


def image_tensor(value: ImageValue) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value
    else:
        result = torch.from_numpy(np.asarray(value).copy())
        if result.ndim == 3:
            result = result.permute(2, 0, 1)
    if result.ndim == 2:
        result = result.unsqueeze(0)
    return result.float().div(255) if result.dtype == torch.uint8 else result.float()


class RawSplit(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self, data: RawStorage, targets: torch.Tensor | npt.ArrayLike, *, name: str, split: str,
        version: str = "1", paired: bool = False,
    ) -> None:
        self.data = data
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        if self.targets.ndim != 1:
            raise ValueError("Raw classification labels must be one dimensional")
        self.ids = source_ids(name, split, len(self.targets), version)
        self.paired = paired
        if len(data) != len(self.targets):
            raise ValueError("Input and label lengths differ")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.data[index][0] if self.paired else self.data[index]
        return image_tensor(value), self.targets[index], self.ids[index]


@dataclass
class DatasetBundle:
    """Raw classification dataset and its label/identity metadata.

    Fields:
        name: Stable dataset namespace used in manifests; custom datasets need a
            distinct name/version combination to avoid source-ID collisions.
        train: Raw training inputs and native integer labels; validation is split from this.
        test: Explicit held-out inputs and labels, separate from the training split.
        output_ids: Sorted native class IDs, including classes absent from a particular split.
        version: Dataset version string recorded in manifests; pass the same version
            when constructing RawSplit so source IDs distinguish dataset revisions.
        test_designation: Human-readable provenance of the held-out split, such as
            "official_test", "official_labeled_validation", or "explicit_labeled_holdout".
    """

    name: str
    train: RawSplit
    test: RawSplit
    output_ids: tuple[int, ...]
    version: str = "1"
    test_designation: str = "official_test"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.train[0][0].shape)


DatasetInput: TypeAlias = DatasetBundle | dict[str, DatasetBundle] | None


def tensor_bundle(
    train_x: torch.Tensor | npt.NDArray[np.generic], train_y: torch.Tensor | npt.ArrayLike,
    test_x: torch.Tensor | npt.NDArray[np.generic] | None = None,
    test_y: torch.Tensor | npt.ArrayLike | None = None, *, name: str = "tensor",
    output_ids: Sequence[int] | None = None, version: str = "1",
) -> DatasetBundle:
    """Build an injectable dataset. Held-out tensors must be explicitly supplied."""
    train_y = torch.as_tensor(train_y, dtype=torch.long)
    if not len(train_y):
        raise ValueError("A tensor dataset requires nonempty training data")
    if test_x is None:
        test_x = torch.empty((0, *torch.as_tensor(train_x).shape[1:]))
        test_y = torch.empty(0, dtype=torch.long)
    if test_y is None:
        raise ValueError("test_y is required when test_x is supplied")
    labels = tuple(sorted(set(train_y.tolist()) | set(torch.as_tensor(test_y).tolist())))
    labels = labels if output_ids is None else tuple(output_ids)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("output_ids must be nonempty and unique")
    if not (set(train_y.tolist()) | set(torch.as_tensor(test_y).tolist())) <= set(labels):
        raise ValueError("Native labels are missing from output_ids")
    return DatasetBundle(name,
                         RawSplit(train_x, train_y, name=name, split="train", version=version),
                         RawSplit(test_x, test_y, name=name, split="test", version=version),
                         tuple(sorted(labels)), version, "explicit_labeled_holdout")


def _synthetic(root: str | Path, *, seed: int = 0, **kwargs: Any) -> DatasetBundle:
    generator = torch.Generator().manual_seed(seed)
    shape = tuple(kwargs.pop("input_shape", (1, 8, 8)))
    classes = int(kwargs.pop("num_classes", 4))
    n_train, n_test = int(kwargs.pop("n_train", 256)), int(kwargs.pop("n_test", 64))
    if kwargs or classes < 1 or n_train < classes or n_test < 1 or any(d < 1 for d in shape):
        raise ValueError("Invalid synthetic dataset options")
    prototypes = torch.rand((classes, *shape), generator=generator)
    def split(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.arange(length) % classes
        x = (prototypes[y] + 0.05 * torch.randn((length, *shape), generator=generator)).clamp(0, 1)
        return x, y
    x, y = split(n_train)
    tx, ty = split(n_test)
    identity = f"synthetic-{seed}-{shape}-{classes}-{n_train}-{n_test}"
    return tensor_bundle(x, y, tx, ty, name=identity)


def _torchvision(name: str, root: str | Path, *, download: bool = True, **kwargs: Any) -> DatasetBundle:
    if kwargs:
        raise ValueError(f"Unexpected {name} dataset options: {sorted(kwargs)}")
    try:
        from torchvision import datasets
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Real image adapters require a working torchvision installation") from exc
    constructors = {"mnist": (datasets.MNIST, 10), "fashion_mnist": (datasets.FashionMNIST, 10),
                    "emnist_balanced": (datasets.EMNIST, 47), "cifar10": (datasets.CIFAR10, 10),
                    "cifar100": (datasets.CIFAR100, 100), "svhn": (datasets.SVHN, 10)}
    constructor, count = constructors[name]
    extra = {"split": "balanced"} if name == "emnist_balanced" else {}
    raw = {}
    for split in ("train", "test"):
        split_args = {"split": split} if name == "svhn" else {"train": split == "train", **extra}
        ds = constructor(root=str(root), download=download, **split_args)
        targets = ds.labels if name == "svhn" else ds.targets
        raw[split] = RawSplit(ds, targets, name=name, split=split, paired=True)
    return DatasetBundle(name, raw["train"], raw["test"], tuple(range(count)))


class _ImageFiles(Dataset[npt.NDArray[np.uint8]]):
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> npt.NDArray[np.uint8]:
        with Image.open(self.paths[index]) as source:
            return np.asarray(source.convert("RGB")).copy()


def _tiny_imagenet(root: str | Path, *, download: bool = False, **kwargs: Any) -> DatasetBundle:
    if kwargs:
        raise ValueError(f"Unexpected Tiny ImageNet options: {sorted(kwargs)}")
    root = Path(root)
    if not (root / "wnids.txt").exists():
        root = root / "tiny-imagenet-200"
    if not (root / "wnids.txt").exists():
        if download:
            raise ValueError("Extract the official tiny-imagenet-200 archive under data_root before running")
        raise FileNotFoundError(f"Missing Tiny ImageNet wnids.txt under {root}")
    classes = sorted(root.joinpath("wnids.txt").read_text().split())
    mapping = {name: i for i, name in enumerate(classes)}
    train_paths, train_labels = [], []
    for name in classes:
        paths = sorted(root.joinpath("train", name, "images").glob("*"))
        train_paths.extend(paths)
        train_labels.extend([mapping[name]] * len(paths))
    annotations = [line.split() for line in root.joinpath("val", "val_annotations.txt").read_text().splitlines()]
    annotations.sort(key=lambda row: row[0])
    test_paths = [root / "val" / "images" / row[0] for row in annotations]
    test_labels = [mapping[row[1]] for row in annotations]
    if not train_paths or not test_paths:
        raise ValueError("Tiny ImageNet training or labeled validation split is empty")
    return DatasetBundle("tiny_imagenet",
                         RawSplit(_ImageFiles(train_paths), train_labels, name="tiny_imagenet", split="train"),
                         RawSplit(_ImageFiles(test_paths), test_labels, name="tiny_imagenet", split="official_val"),
                         tuple(range(len(classes))), test_designation="official_labeled_validation")


DATASETS: dict[str, Callable[..., DatasetBundle]] = {"synthetic": _synthetic, "tiny_imagenet": _tiny_imagenet}
for _name in ("mnist", "fashion_mnist", "emnist_balanced", "cifar10", "cifar100", "svhn"):
    DATASETS[_name] = lambda root, _name=_name, **kwargs: _torchvision(_name, root, **kwargs)
DATASETS["cifar_10"] = DATASETS["cifar10"]
DATASETS["cifar_100"] = DATASETS["cifar100"]
DATASETS["fashion-mnist"] = DATASETS["fashion_mnist"]


def register_dataset(name: str, loader: Callable[..., DatasetBundle]) -> None:
    if name in DATASETS:
        raise ValueError(f"Dataset {name!r} is already registered")
    DATASETS[name] = loader


def load_dataset(
    name: str, root: str | Path = "data", *, seed: int = 0, options: dict[str, Any] | None = None,
) -> DatasetBundle:
    """Load a registered dataset with the following ``data_options`` keyword arguments.

    ``synthetic`` accepts ``input_shape`` (default ``[1, 8, 8]``, positive dimensions
    in channel-first order), ``num_classes`` (4, at least 1), ``n_train`` (256,
    at least num_classes), ``n_test`` (64, at least 1), and ``seed`` (defaults to
    the caller's data seed). It creates random class prototypes with Gaussian
    noise of standard deviation 0.05, clipped to [0, 1]. Native labels are
    0 through num_classes-1 and cycle evenly through each split.

    ``mnist``, ``fashion_mnist`` (alias ``fashion-mnist``), ``emnist_balanced``,
    ``cifar10`` (``cifar_10``), ``cifar100`` (``cifar_100``), and ``svhn`` accept
    only ``download`` (default True). False requires files already under root.

    ``tiny_imagenet`` accepts only ``download`` (default False); automatic download
    is unsupported. Extract the official archive at root or root/tiny-imagenet-200.
    Its official labeled validation split serves as the held-out test split.

    Other registry entries define their own options. Their loader is called as
    ``loader(root, **options)`` and must return a DatasetBundle. Only the synthetic
    loader receives this function's seed automatically. Unknown built-in options fail.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    kwargs = dict(options or {})
    if name == "synthetic":
        kwargs.setdefault("seed", seed)
    return DATASETS[name](root, **kwargs)


class PreparedSplit(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """A fixed raw split; augmentation, permutation, and normalization stay ordered."""
    def __init__(
        self, raw: RawSplit, indices: torch.Tensor | Sequence[int], *,
        resize: int | Sequence[int] | None = None, normalization: str = "unit_interval",
        augmentation: str = "none", mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
    ) -> None:
        self.raw, self.indices = raw, torch.as_tensor(indices, dtype=torch.long)
        self.targets, self.ids = raw.targets[self.indices], raw.ids[self.indices]
        self.resize = (resize, resize) if isinstance(resize, int) else tuple(resize) if resize else None
        self.normalization, self.augmentation = normalization, augmentation
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.indices)

    def read(
        self, index: int, *, augment: bool = False, permutation: torch.Tensor | None = None,
        axes: str = "spatial",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, source_id = self.raw[int(self.indices[index])]
        if self.resize is not None:
            if x.ndim != 3:
                raise ValueError("Image resizing requires CHW inputs")
            x = F.interpolate(x.unsqueeze(0), size=self.resize, mode="bilinear", align_corners=False, antialias=True)[0]
        if augment and self.augmentation != "none":
            if x.ndim != 3:
                raise ValueError("Image augmentation requires CHW inputs")
            if self.augmentation in {"random_crop", "random_crop_flip", "cifar"}:
                height, width = x.shape[-2:]
                x = F.pad(x, (4, 4, 4, 4))
                row, col = torch.randint(9, (2,)).tolist()
                x = x[:, row:row + height, col:col + width]
            if self.augmentation in {"horizontal_flip", "random_crop_flip", "cifar"} and torch.rand(()) < 0.5:
                x = x.flip(-1)
        if permutation is not None:
            if axes == "spatial":
                x = x.reshape(x.shape[0], -1)[:, permutation].reshape(x.shape)
            else:
                x = x.flatten()[permutation].reshape(x.shape)
        if self.normalization == "dataset_stats":
            x = (x - self.mean) / self.std
        return x, y, source_id

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.read(index)


def prepare_data(
    bundle: DatasetBundle, config: PreprocessingConfig, *, seed: int,
) -> tuple[dict[str, PreparedSplit], dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in bundle.output_ids:
        indices = torch.where(bundle.train.targets == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_val = int(len(indices) * config.validation_fraction)
        val.extend(indices[:n_val].tolist())
        train.extend(indices[n_val:].tolist())
    train, val = torch.tensor(sorted(train), dtype=torch.long), torch.tensor(sorted(val), dtype=torch.long)
    if not len(train):
        raise ValueError("The training split is empty")
    fields = dict(resize=config.resize, normalization=config.normalization, augmentation=config.augmentation)
    if config.normalization == "dataset_stats":
        statistics_view = PreparedSplit(bundle.train, train, resize=config.resize)
        total = square = None
        count = 0
        for x, _, _ in statistics_view:
            flat = x.double().reshape(x.shape[0], -1)
            sums, squares = flat.sum(1), flat.square().sum(1)
            total = sums if total is None else total + sums
            square = squares if square is None else square + squares
            count += flat.shape[1]
        shape = (-1,) + (1,) * (len(statistics_view[0][0].shape) - 1)
        fields["mean"] = (total / count).float().reshape(shape)
        fields["std"] = (square / count - (total / count).square()).clamp_min(1e-12).sqrt().float().reshape(shape)
    splits = {"train": PreparedSplit(bundle.train, train, **fields),
              "val": PreparedSplit(bundle.train, val, **fields),
              "test": PreparedSplit(bundle.test, torch.arange(len(bundle.test)), **fields)}
    metadata = {"dataset": bundle.name, "version": bundle.version, "output_ids": bundle.output_ids,
                "test_designation": bundle.test_designation,
                "split_ids": {key: view.ids.clone() for key, view in splits.items()},
                "preprocessing": {"resize": config.resize, "normalization": config.normalization,
                                  "augmentation": config.augmentation, "mean": fields.get("mean"), "std": fields.get("std")}}
    return splits, metadata
