from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from testbed.core.config import strict_dataclass
from testbed.core.metrics import write_json
from testbed.core.random import derive_seed, isolated_rng
from testbed.testing.common import ProbeResult, fit_pair, source_state
from testbed.training.class_remap.data import draw_residuals

from .config import OffsetProbeConfig


def run(
    config: OffsetProbeConfig | dict[str, Any], checkpoint: str | Path | dict[str, Any],
    output_dir: str | Path, *, device: str | torch.device = "cpu", seed: int = 0, **unused: Any,
) -> ProbeResult:
    config = strict_dataclass(OffsetProbeConfig, config) if isinstance(config, dict) else config
    checkpoint = source_state(checkpoint)
    artifact = checkpoint["paradigm_state"].get("fixed_regression")
    if artifact is None or any(k not in artifact for k in ("inputs", "example_ids", "generator", "target_mean")):
        raise ValueError("S05 refitting requires the checkpoint's exact-X pretraining artifact")
    if checkpoint["problem"]["loss_kind"] != "mse" or len(checkpoint["problem"]["output_ids"]) != 1:
        raise ValueError("S05 requires the original scalar regression head")
    offsets = list(dict.fromkeys(artifact["target_mean"] if b == "same" else float(b) for b in config.target_offsets))
    results = []
    with isolated_rng():
        for repeat in range(config.repeats):
            residual_seed = derive_seed(seed, "residuals", repeat)
            residuals = draw_residuals(artifact["inputs"], artifact["generator"], seed=residual_seed)
            for offset in offsets:
                results.append(fit_pair(checkpoint, config, artifact["inputs"], residuals + offset,
                                        artifact["example_ids"], output_dir=output_dir,
                                        seed=derive_seed(seed, "order", repeat), device=device,
                                        metadata={"probe": "offset_refit", "repeat": repeat,
                                                  "pretraining_offset": artifact["target_mean"],
                                                  "target_offset": offset, "residual_seed": residual_seed,
                                                  "input_source": "exact_pretraining"}))
    result = ProbeResult(str(output_dir), checkpoint["source_update"], results)
    write_json(Path(output_dir) / "result.json", vars(result))
    return result
