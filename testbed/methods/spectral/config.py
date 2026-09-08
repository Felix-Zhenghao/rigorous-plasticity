from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralConfig:
    """Regularize matrix spectral norms, vector magnitudes and normalization gains.

    coefficient: Finite nonnegative lambda multiplying the sum of matrix,
        vector and gain penalties below. Zero removes penalty gradients.
    exponent: Finite positive p. Each selected matrix contributes
        (estimated_sigma_max**p - 1)**2; each selected vector contributes
        ||v||_2**(2*p). Effective normalization gains contribute ||gain-1||_2^2
        independently of p. Linear/Conv2d weights flatten to [out, -1];
        embedding tensors are squeezed and flattened when at least 2-D.
    power_iterations: Positive iterations per training penalty evaluation
        used to estimate each matrix's largest singular value. Persistent
        vectors warm-start the next estimate; larger values cost more work.
    eps: Finite positive minimum denominator when normalizing power vectors.
    parameter_scope: Modules to select before applying the include_* filters.
        "network" selects the whole network; "head" selects only the final
        output layer (network.head); "hidden" selects everything except that
        layer, including head_hidden layers. A nonempty tuple of distinct
        names from network.named_modules() selects those modules and their
        descendants, e.g. ("hidden.0",) for the first MLP hidden layer.
        The root name "" selects only parameters owned directly by the root;
        use "network" to include all descendants.
        The resulting selection must contain trainable parameters.
    include_bias: Include all parameters named "bias", including normalization
        biases when include_norm_affine=True. Biases receive vector penalties.
    include_norm_affine: Penalize effective LayerNorm/BatchNorm gains toward
        one and their biases toward zero (subject to include_bias). Running
        statistics are excluded; residual gain offsets are converted to gains.
    include_embeddings: Include parameters outside Linear, Conv2d, LayerNorm
        and BatchNorm, such as ViT class/position embeddings. Squeezed tensors
        with at least two dimensions receive matrix penalties, others vectors.
    """

    coefficient: float
    exponent: float = 2.0
    power_iterations: int = 1
    eps: float = 1e-12
    parameter_scope: str | tuple[str, ...] = "network"
    include_bias: bool = True
    include_norm_affine: bool = True
    include_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.coefficient < 0 or not math.isfinite(self.coefficient):
            raise ValueError("coefficient must be finite and nonnegative")
        if self.exponent <= 0 or not math.isfinite(self.exponent):
            raise ValueError("exponent must be finite and positive")
        if type(self.power_iterations) is not int or self.power_iterations <= 0:
            raise ValueError("power_iterations must be positive")
        if self.eps <= 0 or not math.isfinite(self.eps):
            raise ValueError("eps must be finite and positive")
        if isinstance(self.parameter_scope, str):
            if self.parameter_scope not in {"network", "hidden", "head"}:
                raise ValueError("invalid parameter_scope")
        elif not self.parameter_scope or len(set(self.parameter_scope)) != len(self.parameter_scope):
            raise ValueError("parameter_scope must contain distinct module names")


CONFIG_CLASS = SpectralConfig
