from dataclasses import dataclass


@dataclass(frozen=True)
class LayerNormConfig:
    """Backpropagation with LayerNorm, with no method-specific options.

    MLP and ResNet builders force model.norm="layer". ViT already uses
    LayerNorm. Set normalization axes, affine parameters, gain mode, epsilon
    and placement through the architecture's model configuration.
    """

    pass


CONFIG_CLASS = LayerNormConfig
