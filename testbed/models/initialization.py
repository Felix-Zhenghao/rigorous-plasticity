"""Save sampling rules using the original layer fans, including for sliced resets."""
from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
from torch import Tensor, nn

from .config import ModelConfig

if TYPE_CHECKING:
    from . import Network


class _RequiredInitializationRule(TypedDict):
    distribution: Literal["constant", "uniform", "normal"]
    mean: float


class InitializationRule(_RequiredInitializationRule, total=False):
    """Saved sampling metadata; distribution selects which optional keys exist."""

    low: float
    high: float
    std: float
    fan_in: int
    fan_out: int
    shape: list[int]


def initialize(network: Network, config: ModelConfig) -> None:
    rules: dict[str, InitializationRule] = {}
    rule: InitializationRule
    for module_name, module in network.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            name = f"{module_name}.{local_name}" if module_name else local_name
            if isinstance(module, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
                mean = 0.0 if local_name == "bias" or getattr(module, "gain_mode", "standard") == "residual" else 1.0
                rule = {"distribution": "constant", "mean": mean}
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if local_name == "bias":
                    rule = ({"distribution": "uniform", "low": -1 / sqrt(fan_in), "high": 1 / sqrt(fan_in), "mean": 0.0}
                            if config.initialization == "torch_default" else {"distribution": "constant", "mean": 0.0})
                else:
                    nonlinearity = getattr(config, "activation", "relu")
                    gain = nn.init.calculate_gain(nonlinearity if nonlinearity != "gelu" else "relu", getattr(config, "negative_slope", 0.01))
                    if module is network.head:
                        gain = 1.0
                    preset = config.initialization
                    if preset == "kaiming_normal":
                        rule = {"distribution": "normal", "std": gain / sqrt(fan_in), "mean": 0.0}
                    else:
                        bound = (1 / sqrt(fan_in) if preset == "torch_default" else
                                 gain * sqrt(6 / (fan_in + fan_out)) if preset == "xavier_uniform" else gain * sqrt(3 / fan_in))
                        rule = {"distribution": "uniform", "low": -bound, "high": bound, "mean": 0.0}
                rule |= {"fan_in": fan_in, "fan_out": fan_out}
            else:
                rule = {"distribution": "normal", "mean": 0.0, "std": 0.02}
            rules[name] = rule | {"shape": list(parameter.shape)}
    network.initialization_rules = rules
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            parameter.copy_(sample_initial(network, name))


def sample_initial(
    network: Network,
    name: str,
    shape: Sequence[int] | None = None,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    parameter = network.get_parameter(name)
    rule = network.initialization_rules[name]
    target_device = parameter.device if device is None else torch.device(device)
    sampling_device = torch.device(generator.device) if generator is not None else target_device
    values = torch.empty(tuple(parameter.shape if shape is None else shape), dtype=parameter.dtype, device=sampling_device)
    distribution = rule["distribution"]
    if distribution == "constant":
        values.fill_(rule["mean"])
    elif distribution == "uniform":
        values.uniform_(rule["low"], rule["high"], generator=generator)
    else:
        values.normal_(rule["mean"], rule["std"], generator=generator)
    return values.to(target_device)


def initial_mean(network: Network, name: str) -> float:
    return network.initialization_rules[name]["mean"]
