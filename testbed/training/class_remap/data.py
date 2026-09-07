"""Class-label views and the stationary S05 target construction."""

import torch
from torch.utils.data import Dataset

from testbed.core.types import ProblemSpec


class RemappedData(Dataset):
    def __init__(self, source, indices, mapping, *, augment=False):
        self.source, self.indices, self.mapping = source, torch.as_tensor(indices, dtype=torch.long), dict(mapping)
        self.augment = augment

    @property
    def ids(self):
        return self.source.ids[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y, source_id = self.source.read(int(self.indices[index]), augment=self.augment)
        return x, self.mapping[int(y)], source_id


class FixedRegressionData(Dataset):
    def __init__(self, inputs, targets, example_ids):
        self.inputs, self.targets, self.ids = inputs, targets, example_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.ids[index]


def draw_base_targets(inputs, generator_config, *, seed):
    family = generator_config["target_family"]
    if family == "iid_normal":
        values = torch.randn((len(inputs), 1), generator=torch.Generator().manual_seed(seed))
    else:
        from testbed.core.factory import make_network
        config = dict(generator_config["teacher"])
        architecture = config.pop("name", config.pop("architecture", "mlp"))
        model_config = config.pop("model_config", config)
        problem = ProblemSpec(tuple(inputs.shape[1:]), "mse", (0,))
        teacher = make_network(architecture, problem=problem, model_config=model_config, seed=seed)
        teacher.requires_grad_(False).eval()
        with torch.no_grad():
            values = torch.cat([teacher(batch) for batch in inputs.split(512)]).cpu()
        if values.shape != (len(inputs), 1):
            raise ValueError("S05 teacher must have a scalar output")
        if family == "sine_teacher":
            values = torch.sin(generator_config["omega"] * values)
        elif family != "teacher":
            raise ValueError(f"Unknown target family {family!r}")
    return values


def draw_residuals(inputs, generator_config, *, seed):
    """Draw one fixed residual vector, with offset-independent random seeds."""
    values = draw_base_targets(inputs, generator_config, seed=seed)
    if generator_config.get("center_targets", True):
        values = values - values.mean(0, keepdim=True)
    return values * generator_config.get("target_scale", 1.0)
