import torch
from torch.nn import functional as F


def class_positions(targets, problem):
    ids = torch.as_tensor(problem.output_ids, device=targets.device)
    matches = targets.long().reshape(-1, 1) == ids
    if not bool(matches.any(dim=1).all()):
        raise ValueError("target label is outside the fixed output space")
    return matches.long().argmax(dim=1)


def supervised_loss(predictions, targets, problem):
    if predictions.ndim != 2 or predictions.shape[1] != len(problem.output_ids):
        raise ValueError("predictions must match the fixed output head")
    if problem.loss_kind == "cross_entropy":
        return F.cross_entropy(predictions, class_positions(targets, problem))
    if predictions.shape != targets.shape:
        raise ValueError(f"MSE forbids broadcasting: {predictions.shape} != {targets.shape}")
    return F.mse_loss(predictions, targets)
