from dataclasses import dataclass


@dataclass(frozen=True)
class LayerNormConfig:
    pass


CONFIG_CLASS = LayerNormConfig
