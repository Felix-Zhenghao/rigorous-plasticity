from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NaPConfig:
    """Normalize and project MLP/ResNet parameters at scheduled updates.

    The builder forces LayerNorm and disables hidden/convolution biases;
    final output bias still follows model.head_bias. ResNet additionally
    normalizes residual sums. All LayerNorm affine parameters follow the
    affine policy, regardless of the selected weight scope.

    radius: Only "initial" is supported: save each selected weight tensor's
        initial Frobenius norm as the basis of its projection radius.
    radius_multiplier: Finite positive multiplier of each saved initial norm.
        Each projection rescales the whole weight tensor to that radius.
    affine_policy: "init_decay" moves effective LayerNorm gains toward one
        and biases toward zero using affine_decay. "joint_project" rescales
        each norm's combined gain/bias vector to norm sqrt(number_of_gains);
        it rejects GELU, tanh and sigmoid because activations must be
        homogeneous. "free" leaves learned affine parameters untouched.
    affine_decay: Fraction in [0, 1] used only by "init_decay":
        gain <- (1-decay)*gain + decay; bias <- (1-decay)*bias.
        Zero leaves affine parameters unchanged; one resets gain=1, bias=0.
    eps: Finite positive norm cutoff; projections at or below it are skipped.
    parameter_scope: Linear/Conv2d weights to project: "network" selects all,
        "head" selects only the final output layer, and "hidden" selects all
        except that layer, including head_hidden. A nonempty tuple of nonempty names
        from network.named_modules() selects matching modules and descendants.
        At least one weight must be selected; biases are never projected.
    at_updates: Sorted, distinct positive completed-update numbers at which
        to project, after the optimizer step. Use () for a periodic schedule.
    every_updates: Positive interval in completed optimizer updates, including
        across resumes; None disables the periodic schedule. If both schedule
        fields are omitted, this becomes 1 (project every update). Otherwise,
        provide exactly one of nonempty at_updates or every_updates.
    """

    radius: str = "initial"
    radius_multiplier: float = 1.0
    affine_policy: str = "init_decay"
    affine_decay: float = 0.0
    eps: float = 1e-12
    parameter_scope: str | tuple[str, ...] = "network"
    at_updates: tuple[int, ...] = ()
    every_updates: int | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, key)) for key in ('radius_multiplier', 'affine_decay', 'eps')):
            raise ValueError("method numeric parameters must be finite")
        if not self.at_updates and self.every_updates is None:
            object.__setattr__(self, "every_updates", 1)
        if self.radius != "initial" or self.radius_multiplier <= 0 or self.eps <= 0:
            raise ValueError("NaP requires initial radii, a positive multiplier and eps")
        if self.affine_policy not in {"init_decay", "joint_project", "free"} or not 0 <= self.affine_decay <= 1:
            raise ValueError("invalid NaP affine policy or decay")
        object.__setattr__(self, "at_updates", tuple(self.at_updates))
        if bool(self.at_updates) == (self.every_updates is not None):
            raise ValueError("set exactly one of at_updates and every_updates")
        if self.at_updates != tuple(sorted(set(self.at_updates))) or any(type(t) is not int or t < 1 for t in self.at_updates):
            raise ValueError("at_updates must be sorted distinct positive integers")
        if self.every_updates is not None and (type(self.every_updates) is not int or self.every_updates < 1):
            raise ValueError("every_updates must be a positive integer")


CONFIG_CLASS = NaPConfig
