from dataclasses import dataclass


@dataclass(frozen=True)
class BackpropConfig:
    """Plain supervised backpropagation, with no method-specific options.

    Configure the architecture, optimizer, learning-rate schedule and gradient
    clipping through their respective model/training settings.
    """

    pass


CONFIG_CLASS = BackpropConfig
