from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SWRConfig:
    """Replace individual low-utility parameter entries at scheduled updates.

    utility: "magnitude" uses abs(weight). "gradient" uses abs(weight * grad)
        with post-step weights and the last training gradient (after any
        clipping); parameters without gradients are skipped.
    selection: "fraction" replaces the lowest-utility fraction within each
        parameter tensor. "threshold" replaces entries with utility <=
        threshold * that tensor's mean utility.
    fraction: Fraction in [0, 1] used only by "fraction" selection. Each
        tensor's requested count is stochastically rounded to an integer;
        zero replaces none and one replaces all eligible entries.
    threshold: Finite nonnegative relative cutoff used only by "threshold"
        selection. Zero still replaces entries with exactly zero utility.
    replacement: "init_sample" draws new entries from the parameter's
        original initialization distribution; "init_mean" uses its mean.
    ema_decay: Decay in [0, 1) for per-entry utility averages; zero uses the
        current utility only. Positive values update every training step
        from zero, without bias correction. Replacing any entry clears that
        tensor's entire utility average.
    optimizer_state: "keep" preserves optimizer history; "clear_moments"
        zeros parameter-shaped optimizer state only at modified entries,
        preserving step counters; "reset_all" discards all optimizer state.
        "reset_all" runs at every scheduled intervention, even with no replacements.
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

    utility: str = "magnitude"
    selection: str = "fraction"
    fraction: float = 0.001
    threshold: float = 0.01
    replacement: str = "init_sample"
    ema_decay: float = 0.0
    optimizer_state: str = "keep"
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = True
    include_embeddings: bool = True
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('fraction', 'threshold', 'ema_decay')):
            raise ValueError("method numeric parameters must be finite")
        if self.utility not in {"magnitude", "gradient"} or self.selection not in {"fraction", "threshold"}:
            raise ValueError("invalid SWR utility or selection")
        if self.replacement not in {"init_sample", "init_mean"}:
            raise ValueError("invalid SWR replacement")
        if not 0 <= self.fraction <= 1 or self.threshold < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid SWR fraction, threshold, or ema_decay")
        if self.optimizer_state not in {"keep", "clear_moments", "reset_all"}:
            raise ValueError("unknown optimizer_state")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = SWRConfig
