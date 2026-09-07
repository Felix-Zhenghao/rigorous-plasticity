from dataclasses import dataclass


@dataclass(frozen=True)
class BackpropConfig:
    pass


CONFIG_CLASS = BackpropConfig
