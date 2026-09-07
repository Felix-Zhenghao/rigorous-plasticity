from pathlib import Path

import torch

from testbed.core.config import strict_dataclass
from testbed.core.metrics import write_json
from testbed.core.random import derive_seed, isolated_rng
from testbed.core.types import ProblemSpec
from testbed.testing.common import ProbeResult, fit_pair, seen_inputs, source_state, unseen_inputs

from .config import TeacherProbeConfig


def run(config, checkpoint, output_dir, *, device="cpu", seed=0, data_root=None, datasets=None):
    from testbed.core.factory import make_network
    config = strict_dataclass(TeacherProbeConfig, config) if isinstance(config, dict) else config
    checkpoint = source_state(checkpoint)
    problem = ProblemSpec(**checkpoint["problem"])
    results = []
    with isolated_rng():
        for repeat in range(config.repeats):
            input_seed = derive_seed(seed, "inputs", repeat)
            if config.input_source == "seen":
                inputs, ids = seen_inputs(checkpoint, config.n_samples, seed=input_seed)
            else:
                inputs, ids = unseen_inputs(checkpoint, config, seed=input_seed, data_root=data_root, datasets=datasets)
            teacher_seed = derive_seed(seed, "teacher", repeat)
            teacher = make_network(checkpoint["architecture"], problem=problem,
                                   model_config=checkpoint["model_config"], device=device, seed=teacher_seed)
            teacher.requires_grad_(False).eval()
            with torch.no_grad():
                predictions = torch.cat([teacher(x.to(device)).cpu() for x in inputs.split(config.batch_size)])
            if problem.loss_kind == "cross_entropy":
                targets = torch.tensor(problem.output_ids)[predictions.argmax(1)]
            else:
                targets = predictions
            del teacher
            results.append(fit_pair(checkpoint, config, inputs, targets, ids, output_dir=output_dir,
                                    seed=derive_seed(seed, "order", repeat), device=device,
                                    metadata={"probe": "random_teacher", "repeat": repeat,
                                              "input_source": config.input_source, "teacher_seed": teacher_seed}))
    result = ProbeResult(str(output_dir), checkpoint["source_update"], results)
    write_json(Path(output_dir) / "result.json", vars(result))
    return result
