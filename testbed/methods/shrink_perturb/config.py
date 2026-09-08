from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShrinkPerturbConfig:
    """Apply parameter <- retain * parameter + noise_scale * noise on schedule.

    retain: Fraction in [0, 1] of each selected parameter kept per intervention.
        One performs no shrinkage; zero discards the current value.
    noise_scale: Finite nonnegative multiplier of the chosen noise tensor;
        zero performs shrinkage alone.
    noise_source: "fresh_init" draws from each parameter's original
        initialization distribution; "gaussian" draws independent N(0,1)
        entries; "saved_init" reuses the exact saved initial parameter tensor
        at every intervention.
    optimizer_state: "keep" preserves optimizer history; "clear_moments"
        zeros parameter-shaped optimizer state only at modified entries,
        preserving step counters; "reset_all" discards all optimizer state.
        "clear_moments" clears all entries of selected tensors; "reset_all"
        runs at every scheduled intervention.
    reset_running_stats: Reset scoped BatchNorm running means to zero,
        variances to one and batch counters to zero at each intervention.
        This is independent of include_norm_affine; "head" selects no running
        statistics in the supported architectures.
    parameter_scope: Modules to select before applying the include_* filters.
        "network" selects the whole network; "head" selects only the final
        output layer (network.head); "hidden" selects everything except that
        layer, including head_hidden layers. A nonempty tuple of module
        names from network.named_modules() selects those modules and their
        descendants, e.g. ("hidden.0",) for the first MLP hidden layer.
        The root name "" selects only parameters owned directly by the root;
        use "network" to include all descendants.
        The resulting selection must contain trainable parameters.
    include_bias: Include biases outside normalization layers (e.g. Linear
        and Conv2d). Normalization biases follow include_norm_affine instead.
    include_norm_affine: Include LayerNorm/BatchNorm affine weights and biases.
        This flag does not select their running statistics.
    include_embeddings: Include nn.Embedding tables and the ViT class token
        and position embeddings.
    at_updates: Sorted, distinct positive completed-update numbers at which
        to intervene, after the optimizer step. Use () with every_updates;
        otherwise set every_updates=None and provide at least one number.
    every_updates: Intervene after every N completed optimizer updates (N > 0),
        counting from the start of the run and across resumes. None disables
        the periodic schedule; exactly one schedule must be configured.
    """

    retain: float = 0.999
    noise_scale: float = 0.001
    noise_source: str = "fresh_init"
    optimizer_state: str = "keep"
    reset_running_stats: bool = False
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = True
    include_embeddings: bool = True
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('retain', 'noise_scale')):
            raise ValueError("method numeric parameters must be finite")
        if not 0 <= self.retain <= 1 or self.noise_scale < 0:
            raise ValueError("retain must lie in [0, 1] and noise_scale must be nonnegative")
        if self.noise_source not in {"fresh_init", "gaussian", "saved_init"}:
            raise ValueError("unknown noise_source")
        if self.optimizer_state not in {"keep", "clear_moments", "reset_all"}:
            raise ValueError("unknown optimizer_state")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = ShrinkPerturbConfig
