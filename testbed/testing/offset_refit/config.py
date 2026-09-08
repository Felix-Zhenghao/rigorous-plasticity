from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from testbed.testing.common import validate_probe


@dataclass
class OffsetProbeConfig:
    """Refit scalar targets on the exact fixed-regression pretraining inputs.

    Fields:
        updates: Positive optimizer-step budget for each fitting branch on one fixed
            input/target set. Probe steps start at 0, independently of pretraining steps.
        batch_size: Positive minibatch size for probe fitting and evaluation. The last
            minibatch of each pass may be smaller; inputs are reshuffled on every pass.
        eval_every_updates: Measure fitting-set loss (and classification accuracy) every
            this many probe steps. Step 0 and the final step are always measured;
            loss_auc is the trapezoidal average over these measurements.
        target_offsets: Finite scalar offsets b for y = b + new residuals on the exact
            pretraining inputs. "same" uses the pretraining target_mean; duplicate
            resolved offsets are removed. All offsets in a repeat share residuals
            and minibatch order, isolating the effect of the offset.
        repeats: Positive number of independently redrawn residual vectors.
            A repeat shares its residuals across all target_offsets and both fitting branches.
        load_optimizer_state: True restores the checkpoint's optimizer moments/counters
            for the aged branch; all non-LR settings and parameter names must match. False
            starts a new optimizer. The probe LR schedule always starts at probe step 0;
            a fresh reference always starts with a new optimizer.
        optimizer: Probe optimizer recipe: name is "sgd", "adam", or "adamw"; lr is
            the probe learning rate. Other options and constraints are documented in
            testbed.core.optim.OptimizerConfig. Used without pretraining method penalties.
        lr_schedule: Probe learning-rate policy: "constant", or a mapping with name
            "linear"/"cosine" (requires horizon; terminal_lr defaults to 0), or "step"
            (requires sorted positive milestones; gamma defaults to 0.1). All accept
            warmup_updates (default 0). Counts are probe optimizer steps, not source steps;
            see testbed.core.optim.resolve_schedule for the formulas.
        grad_clip_norm: Positive maximum L2 norm of the full gradient before each probe
            optimizer step. None disables clipping.
        fresh_reference: True fits a paired network recreated from the source architecture
            and original initialization seed, then reports aged/fresh loss gaps (and
            accuracy gaps for classification). False fits only the aged checkpoint network.
    """

    updates: int
    batch_size: int = 32
    eval_every_updates: int = 10
    target_offsets: tuple[float | str, ...] | list[float | str] = ("same", 0)
    repeats: int = 1
    load_optimizer_state: bool = False
    optimizer: dict[str, Any] = field(default_factory=lambda: {"name": "adam", "lr": 0.001})
    lr_schedule: str | dict[str, Any] = "constant"
    grad_clip_norm: float | None = None
    fresh_reference: bool = True

    def __post_init__(self) -> None:
        validate_probe(self)
        if not self.target_offsets or any(x != "same" and (type(x) not in (float, int) or not math.isfinite(x))
                                          for x in self.target_offsets):
            raise ValueError("target_offsets must contain numbers or 'same'")
