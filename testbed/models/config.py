import math
from dataclasses import asdict, dataclass


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def probability(value, name):
    if not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1)")


def validate_common(config):
    for name in ("head_bias", "ln_affine", "bias", "normalize_residual_sum"):
        if hasattr(config, name) and type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    for name in ("head_hidden_sizes",):
        for width in getattr(config, name):
            positive_int(width, name)
    if config.initialization not in {"kaiming_uniform", "kaiming_normal", "xavier_uniform", "torch_default"}:
        raise ValueError("unknown initialization preset")
    if config.ln_eps <= 0 or not math.isfinite(config.ln_eps):
        raise ValueError("ln_eps must be finite and positive")
    if config.ln_gain not in {"standard", "residual"}:
        raise ValueError("ln_gain must be standard or residual")
    if hasattr(config, "activation"):
        if config.activation not in {"relu", "leaky_relu", "gelu", "tanh"}:
            raise ValueError("unknown activation")
        if config.negative_slope < 0 or not math.isfinite(config.negative_slope):
            raise ValueError("negative_slope must be finite and nonnegative")
        if config.activation != "leaky_relu" and config.negative_slope != 0.01:
            raise ValueError("negative_slope only applies to leaky_relu")


@dataclass(frozen=True)
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    negative_slope: float = 0.01
    bias: bool = True
    dropout: float = 0.0
    norm: str = "none"
    norm_position: str = "pre_activation"
    ln_eps: float = 1e-5
    ln_affine: bool = True
    ln_gain: str = "standard"
    head_hidden_sizes: tuple[int, ...] = ()
    head_bias: bool = True
    initialization: str = "kaiming_uniform"

    def __post_init__(self):
        validate_common(self)
        for width in self.hidden_sizes:
            positive_int(width, "hidden_sizes")
        probability(self.dropout, "dropout")
        if self.norm not in {"none", "layer"}:
            raise ValueError("MLP norm must be none or layer")
        if self.norm_position not in {"pre_activation", "post_activation"}:
            raise ValueError("invalid norm_position")
        if self.norm == "none" and (self.norm_position != "pre_activation" or self.ln_gain != "standard" or not self.ln_affine or self.ln_eps != 1e-5):
            raise ValueError("non-default LayerNorm options require norm=layer")


@dataclass(frozen=True)
class ResNet18Config:
    stem: str = "small"
    base_channels: int = 64
    activation: str = "relu"
    negative_slope: float = 0.01
    normalize_residual_sum: bool = False
    norm: str = "batch"
    norm_position: str = "pre_activation"
    ln_eps: float = 1e-5
    ln_affine: bool = True
    ln_gain: str = "standard"
    ln_axes: str = "channels"
    bias: bool = False
    head_hidden_sizes: tuple[int, ...] = ()
    head_bias: bool = True
    initialization: str = "kaiming_uniform"

    def __post_init__(self):
        validate_common(self)
        positive_int(self.base_channels, "base_channels")
        if self.stem not in {"small", "imagenet"}:
            raise ValueError("stem must be small or imagenet")
        if self.norm not in {"batch", "layer"}:
            raise ValueError("ResNet norm must be batch or layer")
        if self.norm_position != "pre_activation":
            raise ValueError("ResNet normalization precedes activations")
        if self.ln_axes not in {"channels", "all_features"}:
            raise ValueError("ln_axes must be channels or all_features")
        if self.norm == "batch" and (self.ln_axes != "channels" or self.ln_gain != "standard" or not self.ln_affine or self.ln_eps != 1e-5):
            raise ValueError("non-default LayerNorm options require norm=layer")


@dataclass(frozen=True)
class ViTConfig:
    patch_size: int = 4
    depth: int = 6
    embed_dim: int = 192
    num_heads: int = 3
    head_dim: int | None = None
    mlp_dim: int = 768
    pool: str = "cls"
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    feedforward_dropout: float = 0.0
    patch_embedding: str = "conv"
    normalization_layout: str = "pre_norm"
    ln_eps: float = 1e-5
    ln_affine: bool = True
    ln_gain: str = "standard"
    head_hidden_sizes: tuple[int, ...] = ()
    head_bias: bool = True
    initialization: str = "kaiming_uniform"

    def __post_init__(self):
        validate_common(self)
        for name in ("patch_size", "depth", "embed_dim", "num_heads", "mlp_dim"):
            positive_int(getattr(self, name), name)
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.embed_dim // self.num_heads)
        if self.head_dim != self.embed_dim // self.num_heads:
            raise ValueError("head_dim * num_heads must equal embed_dim")
        for name in ("attention_dropout", "embedding_dropout", "feedforward_dropout"):
            probability(getattr(self, name), name)
        if self.pool not in {"cls", "mean"}:
            raise ValueError("pool must be cls or mean")
        if self.patch_embedding not in {"conv", "linear"}:
            raise ValueError("patch_embedding must be conv or linear")
        if self.normalization_layout not in {"pre_norm", "patch_norm"}:
            raise ValueError("normalization_layout must be pre_norm or patch_norm")


MODEL_CONFIGS = {"mlp": MLPConfig, "resnet_18": ResNet18Config, "vit": ViTConfig}
PRESETS = {
    "p08_mlp": ("mlp", {"hidden_sizes": [100, 100]}),
    "p09_mlp": ("mlp", {"hidden_sizes": [2000, 2000, 2000]}),
    "p12_vit": ("vit", {"depth": 12, "embed_dim": 192, "num_heads": 3, "mlp_dim": 768, "patch_size": 4}),
    "p13_vit": ("vit", {"depth": 8, "embed_dim": 384, "num_heads": 12, "mlp_dim": 1536, "patch_size": 4}),
}


def resolve_config(architecture, values=None):
    values = asdict(values) if hasattr(values, "__dataclass_fields__") else dict(values or {})
    declared = values.pop("name", architecture)
    if declared != architecture:
        raise ValueError("architecture name and model config disagree")
    preset = values.pop("preset", None)
    if preset:
        if preset not in PRESETS or PRESETS[preset][0] != architecture:
            raise ValueError(f"invalid {architecture} preset: {preset}")
        values = PRESETS[preset][1] | values
    if architecture not in MODEL_CONFIGS:
        raise ValueError(f"unknown architecture: {architecture}")
    try:
        return MODEL_CONFIGS[architecture](**values)
    except TypeError as error:
        raise ValueError(f"invalid {architecture} config: {error}") from error
