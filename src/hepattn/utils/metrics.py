from typing import Literal

import torch
from torch import Tensor


def mask_metric_cost(
    preds: Tensor,
    targets: Tensor,
    input_pad_mask: Tensor | None = None,
    metric: Literal["iou", "jac", "dice", "smc", "eff", "pur"] = "iou",
) -> Tensor:
    # Pred and target masks have shape (batch, num_objects, num_constituents)
    targets = targets.type_as(preds)

    # Used to mask out invalid constituents during the score calculation
    if input_pad_mask is not None:
        mask = input_pad_mask.unsqueeze(1).float()
        preds = preds * mask

    # Calculate binary metrics, (batch, objects, hits) -> (batch, objects, objects)
    tp = torch.einsum("bnc,bmc->bnm", preds, targets)
    tn = torch.einsum("bnc,bmc->bnm", 1 - preds, (1 - targets))
    fp = torch.einsum("bnc,bmc->bnm", preds, (1 - targets))
    fn = torch.einsum("bnc,bmc->bnm", 1 - preds, targets)

    eps = 1e-6

    num_true = torch.sum(targets, dim=-1).unsqueeze(-1)
    num_pred = torch.sum(preds, dim=-1).unsqueeze(-2)

    if metric == "smc":
        score = (tp + tn) / (tp + tn + fp + fn)
    elif metric == "dice":
        score = 2 * tp / (2 * tp + fp + fn + eps)
    elif metric in {"iou", "jac"}:
        score = tp / (tp + fp + fn + eps)
    elif metric == "eff":
        score = tp / (num_true + eps)
    elif metric == "pur":
        score = tp / (num_pred + eps)

    return 1 - score


def mask_metric_score(
    preds: Tensor,
    targets: Tensor,
    input_pad_mask: Tensor | None = None,
    metric: Literal["iou", "jac", "dice", "smc", "eff", "pur"] = "iou",
) -> Tensor:
    targets = targets.type_as(preds)

    if input_pad_mask is not None:
        mask = input_pad_mask.unsqueeze(1).float()
        preds = preds * mask

    tp = torch.sum(preds * targets, dim=-1)
    tn = torch.sum((1 - preds) * (1 - targets), dim=-1)
    fp = torch.sum(preds * (1 - targets), dim=-1)
    fn = torch.sum((1 - preds) * targets, dim=-1)

    n_true = torch.sum(targets, dim=-1)
    n_pred = torch.sum(preds, dim=-1)

    eps = 1e-6

    if metric == "smc":
        score = (tp + tn) / (tp + tn + fp + fn + eps)
    elif metric == "dice":
        score = 2 * tp / (2 * tp + fp + fn + eps)
    elif metric in {"iou", "jac"}:
        score = tp / (tp + fp + fn + eps)
    elif metric == "eff":
        score = tp / (n_true + eps)
    elif metric == "pur":
        score = tp / (n_pred + eps)
    else:
        raise ValueError(f"Unknown metric '{metric}'")

    return score
