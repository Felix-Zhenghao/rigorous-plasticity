from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CChainConfig:
    """Match current predictions to a lagged model on buffered past inputs.

    buffer_size: Positive maximum number of unique example IDs retained on
        CPU. Reobserved IDs become most recent; the least recently observed
        ID is evicted when full. Memory grows with buffer_size * input size.
    reference_batch_size: Positive maximum buffered examples sampled without
        replacement per update, excluding IDs in the current supervised
        batch. Uses fewer examples when fewer are eligible.
    coefficient: Finite nonnegative multiplier of reference loss. Constant
        in "fixed" mode; the initial value before adaptation in "loss_ratio"
        mode. Zero disables the initial/fixed reference-loss contribution.
    reference_lag_updates: Positive number of optimizer updates between the
        reference snapshot and the model at the start of a training step.
        No reference loss is applied until that history exists; storing
        lag+1 snapshots increases memory with the lag.
    distance: "prediction_ce" matches the frozen model's softmax distribution
        using cross entropy and requires classification. "logit_mse" uses
        mean squared error between raw outputs. "auto" chooses prediction_ce
        for cross-entropy problems and logit_mse for regression.
    coefficient_mode: "fixed" keeps coefficient unchanged. "loss_ratio"
        sets the next step's coefficient to target_loss_ratio *
        mean(abs(supervised_loss)) / (mean(reference_loss) + eps).
    target_loss_ratio: Finite nonnegative target for the weighted reference
        loss relative to supervised loss; used only in "loss_ratio" mode.
    loss_window: Positive maximum number of recent loss observations used
        for each adaptation mean. Reference means include only steps with
        eligible reference examples.
    adapt_after_updates: Nonnegative number of this learner's training steps
        to observe before adapting; zero allows adaptation as soon as a
        reference loss is available. The observation count is checkpointed.
    eps: Finite positive denominator offset in coefficient adaptation; keeps
        a near-zero reference-loss mean from causing division by zero.
    """

    buffer_size: int
    reference_batch_size: int
    coefficient: float = 0.1
    reference_lag_updates: int = 1
    distance: str = "auto"
    coefficient_mode: str = "fixed"
    target_loss_ratio: float = 0.01
    loss_window: int = 1000
    adapt_after_updates: int = 1000
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('coefficient', 'target_loss_ratio', 'eps')):
            raise ValueError("method numeric parameters must be finite")
        for key in ("buffer_size", "reference_batch_size", "reference_lag_updates", "loss_window"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.adapt_after_updates) is not int or self.adapt_after_updates < 0:
            raise ValueError("adapt_after_updates must be a nonnegative integer")
        if self.coefficient < 0 or self.target_loss_ratio < 0 or self.eps <= 0:
            raise ValueError("invalid C-CHAIN coefficient, ratio or eps")
        if self.distance not in {"auto", "prediction_ce", "logit_mse"}:
            raise ValueError("invalid reference distance")
        if self.coefficient_mode not in {"fixed", "loss_ratio"}:
            raise ValueError("invalid coefficient_mode")


CONFIG_CLASS = CChainConfig
