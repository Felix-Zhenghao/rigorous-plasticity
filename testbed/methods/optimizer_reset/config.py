from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptResetConfig:
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None
    base_method: str = "backprop"
    base_config: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")
        if self.base_method == "optimizer_reset":
            raise ValueError("optimizer reset wrappers cannot be nested")
        if self.base_config.get("optimizer_state") == "reset_all":
            raise ValueError("a base method cannot also reset all optimizer state")


CONFIG_CLASS = OptResetConfig
