from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class L2InitConfig:
    """Penalize movement away from the network's initial parameters.

    coefficient: Finite nonnegative lambda in
        extra_loss = (lambda / 2) * sum_p ||p - p_initial||_2^2.
        Initial values are saved when the learner is built and checkpointed;
        zero removes this penalty's contribution to gradients.
    parameter_scope: Modules to select before applying the include_* filters.
        "network" selects the whole network; "head" selects only the final
        output layer (network.head); "hidden" selects everything except that
        layer, including head_hidden layers. A nonempty tuple of distinct
        names from network.named_modules() selects those modules and their
        descendants, e.g. ("hidden.0",) for the first MLP hidden layer.
        The root name "" selects only parameters owned directly by the root;
        use "network" to include all descendants.
        The resulting selection must contain trainable parameters.
    include_bias: Include parameters named "bias", including Linear/Conv2d
        biases and normalization biases when include_norm_affine=True.
    include_norm_affine: Include trainable LayerNorm/BatchNorm affine weights
        and biases; running means/variances are never regularized. Stored
        residual gain offsets are treated as parameters, like ordinary gains.
    include_embeddings: Include trainable parameters outside Linear, Conv2d,
        LayerNorm and BatchNorm, including embedding tables and ViT class and
        position embeddings.
    """

    coefficient: float
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = False
    include_embeddings: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        if isinstance(self.parameter_scope, str):
            if self.parameter_scope not in {"network", "hidden", "head"}:
                raise ValueError("invalid parameter_scope")
        elif not self.parameter_scope or len(set(self.parameter_scope)) != len(self.parameter_scope):
            raise ValueError("explicit parameter_scope must contain distinct module names")


CONFIG_CLASS = L2InitConfig
