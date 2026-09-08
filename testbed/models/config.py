"""Architecture options, validated before a network is constructed."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TypeAlias, cast


def positive_int(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def probability(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1)")


def validate_common(config: ModelConfig) -> None:
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
    """Configure the flattened-input MLP and its final linear readout.

    hidden_sizes: Width of each trunk hidden layer, in order; each must be
        positive. An empty tuple connects the input directly to the head.
    activation: Hidden-layer activation: ``relu``, ``leaky_relu``, ``gelu``,
        or ``tanh``; also applies to any hidden layers in the head.
    negative_slope: Multiplier for negative inputs to ``leaky_relu``; finite
        and nonnegative. Other activations require the default value, 0.01.
    bias: Include a learned bias in all hidden linear layers, including head
        hidden layers. The final readout is controlled by ``head_bias``.
    dropout: Probability of dropping each hidden activation during training,
        in [0, 1); applied after activation and normalization. Zero disables it.
    norm: ``none`` disables hidden normalization; ``layer`` normalizes each
        example across the hidden layer's output features.
    norm_position: ``pre_activation`` applies Linear -> LayerNorm -> activation;
        ``post_activation`` applies Linear -> activation -> LayerNorm.
        ``post_activation`` requires ``norm="layer"``.
    ln_eps: Positive finite constant added to LayerNorm's variance denominator
        for numerical stability. Nondefault values require ``norm="layer"``.
    ln_affine: Learn a per-feature scale and bias in LayerNorm. False uses only
        normalization and requires ``norm="layer"``.
    ln_gain: ``standard`` stores the LayerNorm scale directly, initialized to
        one; ``residual`` stores a zero-initialized offset and uses 1 + offset.
        This changes the parameter seen by penalties and resets. ``residual``
        requires ``norm="layer"``; without affine parameters it has no effect.
    head_hidden_sizes: Widths of extra hidden layers between the trunk and
        final output; positive widths, or () for a linear readout. They use
        the same activation, normalization, bias and dropout as the trunk.
    head_bias: Include a learned bias in the final output linear layer.
    initialization: ``kaiming_uniform`` or ``kaiming_normal`` use fan-in-scaled
        weights; ``xavier_uniform`` uses both fan-in and fan-out. These use
        activation-dependent gain (ReLU gain for GELU) and zero linear biases.
        ``torch_default`` uses uniform weights and biases in +/-1/sqrt(fan-in).
        The final readout uses gain 1 for the first three presets.
    """

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

    def __post_init__(self) -> None:
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
    """Configure an 18-layer residual network for channel-first images.

    stem: ``small`` uses a 3x3 convolution with stride 1 and no pooling;
        ``imagenet`` uses a 7x7 stride-2 convolution followed by stride-2 max
        pooling, reducing spatial resolution fourfold before the first stage.
    base_channels: Positive width of the stem and first stage. The four stages
        have widths base_channels * (1, 2, 4, 8), with two blocks per stage.
    activation: ``relu``, ``leaky_relu``, ``gelu``, or ``tanh`` in the stem,
        both activation sites of each block, and any hidden head layers.
    negative_slope: Multiplier for negative inputs to ``leaky_relu``; finite
        and nonnegative. Other activations require the default value, 0.01.
    normalize_residual_sum: Add another normalization after each residual plus
        shortcut sum and before its activation, using the selected ``norm``.
    norm: ``batch`` uses BatchNorm2d in the stem, blocks and projected
        shortcuts, with running statistics for evaluation. ``layer`` uses
        LayerNorm over ``ln_axes`` and also normalizes any hidden head layers.
    norm_position: Only ``pre_activation`` is supported: normalize convolution
        outputs before activation; the second convolution is normalized before
        addition to the shortcut.
    ln_eps: Positive finite constant added to LayerNorm's variance denominator.
        Nondefault values require ``norm="layer"``; BatchNorm keeps PyTorch's
        default epsilon independently of this field.
    ln_affine: Learn a scale and bias along LayerNorm's normalized axes. False
        requires ``norm="layer"``; BatchNorm always has affine parameters.
    ln_gain: ``standard`` stores the LayerNorm scale, initialized to one;
        ``residual`` stores a zero-initialized offset and uses 1 + offset.
        Penalties and resets act on the stored parameter. ``residual`` requires
        ``norm="layer"``; without affine parameters it has no effect.
    ln_axes: ``channels`` normalizes channels separately at each image pixel,
        sharing affine parameters across pixels. ``all_features`` normalizes
        all CxHxW values per example, with separate affine parameters per value.
        ``all_features`` requires ``norm="layer"``. Hidden head LayerNorm
        always normalizes the head's feature dimension.
    bias: Include biases in all convolutions (including shortcut projections)
        and hidden head linear layers. Use ``head_bias`` for the final readout.
    head_hidden_sizes: Positive widths of hidden layers after global spatial
        average pooling; () uses a linear readout. They use ``activation`` and
        ``bias``, no dropout, and LayerNorm only when ``norm="layer"``.
    head_bias: Include a learned bias in the final output linear layer.
    initialization: ``kaiming_uniform`` or ``kaiming_normal`` use fan-in-scaled
        convolution/linear weights; ``xavier_uniform`` uses fan-in and fan-out.
        These use activation-dependent gain (ReLU gain for GELU) and zero
        convolution/linear biases. ``torch_default`` uses uniform weights and
        biases in +/-1/sqrt(fan-in). The final readout uses gain 1 for the first
        three presets. BatchNorm scale and bias initialize to one and zero.
    """

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

    def __post_init__(self) -> None:
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
    """Configure a vision transformer for channel-first images.

    patch_size: Positive side length of square, nonoverlapping image patches;
        must divide both image height and width. Smaller patches create more
        tokens and increase attention memory and computation.
    depth: Positive number of transformer blocks.
    embed_dim: Positive token/embedding width; divisible by ``num_heads``.
    num_heads: Positive number of parallel attention heads per block.
    head_dim: Width per attention head. None fills in embed_dim // num_heads;
        an explicit value must equal that quotient.
    mlp_dim: Positive hidden width of each block's two-layer GELU feedforward
        network; its output width remains ``embed_dim``.
    pool: ``cls`` reads the class token; ``mean`` averages only image-patch
        tokens after the final LayerNorm. Both modes retain the class token
        throughout attention.
    attention_dropout: Dropout probability in [0, 1) on softmax attention
        weights during training; zero disables it.
    embedding_dropout: Dropout probability in [0, 1) after adding positional
        embeddings to patch and class tokens during training.
    feedforward_dropout: Dropout probability in [0, 1) after GELU and the
        second linear layer in each block, and after GELU in hidden head layers.
    patch_embedding: ``conv`` projects patches with a convolution whose kernel
        and stride equal ``patch_size``; ``linear`` flattens each patch then
        applies a linear projection. Both produce ``embed_dim``-wide tokens.
    normalization_layout: Both ``pre_norm`` and ``patch_norm`` normalize before
        each attention/feedforward branch and after the final block.
        ``patch_norm`` additionally normalizes flattened patches before their
        projection and tokens immediately after it; requires ``patch_embedding="linear"``.
    ln_eps: Positive finite constant added to every LayerNorm's variance
        denominator for numerical stability.
    ln_affine: Learn a per-feature scale and bias in every LayerNorm. False
        keeps normalization but removes those learned parameters.
    ln_gain: ``standard`` stores the LayerNorm scale, initialized to one;
        ``residual`` stores a zero-initialized offset and uses 1 + offset.
        Penalties and resets act on the stored parameter. This has no effect
        when ``ln_affine=False``.
    head_hidden_sizes: Positive widths of extra layers after token pooling;
        () uses a linear readout. Hidden head layers use GELU, learned biases,
        ``feedforward_dropout`` and no normalization.
    head_bias: Include a learned bias in the final output linear layer.
    initialization: ``kaiming_uniform`` or ``kaiming_normal`` use fan-in-scaled
        convolution/linear weights with ReLU gain; ``xavier_uniform`` uses that
        gain and both fans. These zero convolution/linear biases and use gain
        1 for the final readout. ``torch_default`` uses uniform weights and
        biases in +/-1/sqrt(fan-in). In every preset, class and positional
        embeddings are sampled from a zero-mean normal with standard deviation 0.02.
    """

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

    def __post_init__(self) -> None:
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


ModelConfig: TypeAlias = MLPConfig | ResNet18Config | ViTConfig

MODEL_CONFIGS: dict[str, type[MLPConfig] | type[ResNet18Config] | type[ViTConfig]] = {
    "mlp": MLPConfig, "resnet_18": ResNet18Config, "vit": ViTConfig,
}
PRESETS: dict[str, tuple[str, dict[str, object]]] = {
    "p08_mlp": ("mlp", {"hidden_sizes": [100, 100]}),
    "p09_mlp": ("mlp", {"hidden_sizes": [2000, 2000, 2000]}),
    "p12_vit": ("vit", {"depth": 12, "embed_dim": 192, "num_heads": 3, "mlp_dim": 768, "patch_size": 4}),
    "p13_vit": ("vit", {"depth": 8, "embed_dim": 384, "num_heads": 12, "mlp_dim": 1536, "patch_size": 4}),
}


def resolve_config(
    architecture: str,
    values: Mapping[str, object] | ModelConfig | None = None,
) -> ModelConfig:
    """Resolve ``mlp``, ``resnet_18`` or ``vit`` options into their config.

    ``values`` may be a matching config, an option mapping, or None for defaults.
    A mapping's optional ``name`` must match ``architecture``. Its optional
    ``preset`` supplies defaults, overridden by explicitly supplied fields:
    ``p08_mlp`` uses hidden widths (100, 100); ``p09_mlp`` uses (2000, 2000, 2000);
    ``p12_vit`` uses depth 12, width 192, 3 heads and feedforward width 768;
    ``p13_vit`` uses depth 8, width 384, 12 heads and feedforward width 1536.
    Both ViT presets use patch size 4. Other fields keep class defaults.
    There are no ResNet presets. Unknown fields and mismatched names/presets fail.
    """
    values = (asdict(cast(ModelConfig, values)) if hasattr(values, "__dataclass_fields__")
              else dict(cast(Mapping[str, object] | None, values) or {}))
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
