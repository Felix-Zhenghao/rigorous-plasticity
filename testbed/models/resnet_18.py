from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from .config import ResNet18Config
from .layers import HiddenLayer, activation, layer_norm

if TYPE_CHECKING:
    from testbed.core.types import ProblemSpec

    from .initialization import InitializationRule


def normalization(channels: int, spatial: tuple[int, int], config: ResNet18Config) -> nn.Module:
    if config.norm == "batch":
        return nn.BatchNorm2d(channels)
    if config.ln_axes == "channels":
        return layer_norm(channels, config, channels=True)
    return layer_norm((channels, *spatial), config)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int,
        spatial: tuple[int, int],
        config: ResNet18Config,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=config.bias)
        self.norm1 = normalization(channels, spatial, config)
        self.activation1 = activation(config.activation, config.negative_slope)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=config.bias)
        self.norm2 = normalization(channels, spatial, config)
        self.shortcut = nn.Identity() if stride == 1 and channels == in_channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride=stride, bias=config.bias),
            normalization(channels, spatial, config))
        self.sum_norm = normalization(channels, spatial, config) if config.normalize_residual_sum else nn.Identity()
        self.activation2 = activation(config.activation, config.negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.activation1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation2(self.sum_norm(x + residual))


class ResNet18(nn.Module):
    initialization_rules: dict[str, InitializationRule]
    resolved_config: dict[str, object]
    architecture: str

    def __init__(self, problem: ProblemSpec, config: ResNet18Config) -> None:
        super().__init__()
        if len(problem.input_shape) != 3:
            raise ValueError("ResNet requires CHW inputs")
        inputs, height, width = problem.input_shape
        channels = config.base_channels
        large = config.stem == "imagenet"
        self.stem_conv = nn.Conv2d(inputs, channels, 7 if large else 3, stride=2 if large else 1,
                                  padding=3 if large else 1, bias=config.bias)
        if large:
            height, width = (height + 1) // 2, (width + 1) // 2
        self.stem_norm = normalization(channels, (height, width), config)
        self.stem_activation = activation(config.activation, config.negative_slope)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1) if large else nn.Identity()
        if large:
            height, width = (height + 1) // 2, (width + 1) // 2
        self.stages = nn.ModuleList()
        previous = channels
        for stage in range(4):
            channels = config.base_channels * 2**stage
            stride = 1 if stage == 0 else 2
            height, width = (height + stride - 1) // stride, (width + stride - 1) // stride
            self.stages.append(nn.Sequential(BasicBlock(previous, channels, stride, (height, width), config),
                                             BasicBlock(channels, channels, 1, (height, width), config)))
            previous = channels
        self.head_hidden = nn.ModuleList()
        for size in config.head_hidden_sizes:
            self.head_hidden.append(HiddenLayer(channels, size, config))
            channels = size
        self.head_input_dim = channels
        self.head = nn.Linear(channels, len(problem.output_ids), bias=config.head_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_pool(self.stem_activation(self.stem_norm(self.stem_conv(x))))
        for stage in self.stages:
            x = stage(x)
        x = x.mean(dim=(2, 3))
        for layer in self.head_hidden:
            x = layer(x)
        return self.head(x)


def build(problem: ProblemSpec, config: ResNet18Config) -> ResNet18:
    return ResNet18(problem, config)
