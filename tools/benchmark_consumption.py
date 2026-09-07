"""Measure loading and optional learning separately, preserving scientific budgets."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from testbed.core.consumption import Consumer
from testbed.core.factory import make_model
from testbed.training import make_paradigm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-learner", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for name, chunk, epochs, batch in (("online", 1, 1, 1), ("repeated", "task", 3, 32), ("intermediate", 10, 2, 4)):
        paradigm = make_paradigm("pixel_permutation", dict(dataset="synthetic", num_tasks=args.tasks,
                                  task_samples=args.samples, chunk_size=chunk, epochs=epochs,
                                  sampling="with_replacement", validation_fraction=0), data_root="data", seed=0)
        consumer = Consumer(paradigm, batch, seed=0)
        learner = make_model("backprop", "mlp", problem=paradigm.problem, model_config={"hidden_sizes": [32]},
                             optimizer_config={"name": "sgd", "lr": .01}, device=args.device) if args.include_learner else None
        loading = learning = 0.0
        count = 0
        trace = hashlib.sha256()
        while True:
            started = time.perf_counter()
            try:
                sample = next(consumer)
            except StopIteration:
                break
            loading += time.perf_counter() - started
            trace.update(sample[2].numpy().tobytes())
            if learner is not None:
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                learner.train_step(sample)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                learning += time.perf_counter() - started
            consumer.commit()
            count += 1
        result = {"regime": name, "task_views": args.tasks, "samples_per_task": args.samples,
                  "chunk_size": chunk, "epochs": epochs, "batch_size": batch, "updates": count,
                  **consumer.counters, "loading_seconds": loading, "learner_seconds": learning,
                  "source_trace_sha256": trace.hexdigest(), "device": args.device,
                  "torch": torch.__version__}
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
