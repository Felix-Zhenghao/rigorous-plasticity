import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import class_positions, supervised_loss
from .random import isolated_rng


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False) + "\n")


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, default=json_value, allow_nan=False) + "\n")


class Score:
    def __init__(self, problem):
        self.problem = problem
        self.loss_sum = 0.0
        self.count = 0
        self.correct = torch.zeros(len(problem.output_ids), dtype=torch.long)
        self.total = self.correct.clone()

    def add(self, predictions, targets, loss):
        self.count += len(targets)
        self.loss_sum += float(loss) * len(targets)
        if self.problem.loss_kind == "cross_entropy":
            labels = class_positions(targets.detach().cpu(), self.problem)
            correct = predictions.detach().cpu().argmax(1) == labels
            self.total += torch.bincount(labels, minlength=len(self.total))
            self.correct += torch.bincount(labels[correct], minlength=len(self.total))

    def result(self):
        if not self.count:
            raise ValueError("cannot score an empty dataset")
        result = {"loss": self.loss_sum / self.count, "samples": self.count}
        if self.problem.loss_kind == "cross_entropy":
            present = self.total > 0
            accuracy = self.correct[present] / self.total[present]
            result.update(accuracy=float(self.correct.sum()) / self.count,
                          macro_accuracy=float(accuracy.mean()),
                          per_class_accuracy={str(label): float(c / n) for label, c, n in
                                              zip(self.problem.output_ids, self.correct, self.total) if n})
        return result


def evaluate(learner, dataset, problem, *, batch_size=256):
    if len(dataset) == 0:
        return None
    score = Score(problem)
    device = next(learner.network.parameters()).device
    with isolated_rng(0), torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(0))
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            predictions = learner.predict(x)
            score.add(predictions, y, supervised_loss(predictions, y, problem))
    return score.result()


class Recorder:
    def __init__(self, path, problem):
        self.writer = JsonlWriter(path)
        self.online = Score(problem)

    def record(self, result, update, counters, *, batch, lr, elapsed):
        x, y, _ = batch
        self.online.add(result.predictions, y, result.supervised_loss)
        record = {"kind": "train", "completed_updates": update, **counters,
                  "pre_update_loss": float(result.supervised_loss), "extra_loss": float(result.extra_loss),
                  "learning_rate": lr, "elapsed_seconds": elapsed,
                  **{key: value for key, value in result.metrics.items() if key != "lr"}}
        if self.online.problem.loss_kind == "cross_entropy":
            positions = class_positions(y.detach().cpu(), self.online.problem)
            record["pre_update_accuracy"] = float((result.predictions.detach().cpu().argmax(1) == positions).float().mean())
        self.writer.write(record)

    def state_dict(self):
        return {k: getattr(self.online, k) for k in ("loss_sum", "count", "correct", "total")}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self.online, key, value)
