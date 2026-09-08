from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from torch import Tensor, nn
from torch.nn import functional as F

from .config import MLPConfig, ModelConfig, ResNet18Config


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        shape: int | Sequence[int],
        *,
        eps: float = 1e-5,
        affine: bool = True,
        gain: str = "standard",
    ) -> None:
        self.gain_mode = gain
        super().__init__(shape, eps=eps, elementwise_affine=affine)
        if affine and gain == "residual":
            nn.init.zeros_(self.weight)

    @property
    def gain(self) -> Tensor | None:
        if self.weight is None:
            return None
        return self.weight + 1 if self.gain_mode == "residual" else self.weight

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.gain, self.bias, self.eps)


class ChannelLayerNorm(LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x.movedim(1, -1).contiguous())
        return x.movedim(-1, 1).contiguous()


def activation(name: str, negative_slope: float = 0.01) -> nn.Module:
    return {"relu": lambda: nn.ReLU(), "leaky_relu": lambda: nn.LeakyReLU(negative_slope),
            "gelu": nn.GELU, "tanh": nn.Tanh}[name]()


def layer_norm(
    shape: int | Sequence[int],
    config: ModelConfig | SimpleNamespace,
    *,
    channels: bool = False,
) -> LayerNorm:
    cls = ChannelLayerNorm if channels else LayerNorm
    return cls(shape, eps=config.ln_eps, affine=config.ln_affine, gain=config.ln_gain)


class HiddenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: MLPConfig | ResNet18Config | SimpleNamespace,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=config.bias)
        self.norm = layer_norm(out_features, config) if config.norm == "layer" else nn.Identity()
        self.activation = activation(config.activation, config.negative_slope)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.0))
        self.norm_position = config.norm_position

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.activation(self.norm(x)) if self.norm_position == "pre_activation" else self.norm(self.activation(x))
        return self.dropout(x)
