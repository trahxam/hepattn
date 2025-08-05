import torch
from torch import Tensor, BoolTensor


def mask_metric_cost(
    preds: Tensor,
    targets: Tensor,
    input_pad_mask: BoolTensor | None = None,
    metric: str = "iou",
    ):

    # Pred and target masks have shape (batch, num_objects, num_constituents)
    num_objects = preds.shape[1]
    targets = targets.type_as(preds)
    
    # Used to mask out invalid constituents during the score calculation
    mask = input_pad_mask.unsqueeze(1).float()

    preds = preds * mask

    # Calculate binary metrics, (batch, objects, hits) -> (batch, objects, objects)
    tp = torch.einsum("bnc,bmc->bnm", preds, targets)
    tn = torch.einsum("bnc,bmc->bnm", 1 - preds, (1 - targets))
    fp = torch.einsum("bnc,bmc->bnm", preds, (1 - targets))
    fn = torch.einsum("bnc,bmc->bnm", 1 - preds, targets)

    p = tp + fn
    n = tn + fp

    eps = 1e-6

    if metric == "smc":
        score = (tp + tn) / (tp + tn + fp + fn)
    elif metric == "dice":
        score = 2 * tp / (2 * tp + fp + fn + eps)
    elif metric == "iou" or metric == "jac":
        score = tp / (tp + fp + fn + eps)
    elif metric == "overlap":
        score = tp / (torch.minimum(p, n) + eps)
    elif metric == "eff":
        score = tp / (p + eps)
    elif metric == "pur":
        score = tp / (tp + fp + eps)

    cost = 1 - score

    return cost


def mask_metric_score(
    preds: Tensor,
    targets: Tensor,
    input_pad_mask: BoolTensor | None = None,
    metric: str = "iou",
    ):

    # Pred and target masks have shape (batch, num_objects, num_constituents)
    num_objects = preds.shape[1]
    targets = targets.type_as(preds)
    
    # Used to mask out invalid constituents during the score calculation
    mask = input_pad_mask.unsqueeze(1).float()
    preds = preds * mask

    # Calculate binary metrics, (batch, object, hits) -> (batch, object)
    tp = torch.sum(preds * targets, dim=-1)
    tn = torch.sum((1 - preds) * (1 - targets), dim=-1)
    fp = torch.sum(preds * (1 - targets), dim=-1)
    fn = torch.sum((1 - preds) * targets, dim=-1)

    p = tp + fn
    n = tn + fp

    eps = 1

    if metric == "smc":
        score = (tp + tn) / (tp + tn + fp + fn)
    elif metric == "dice":
        score = 2 * tp / (2 * tp + fp + fn + eps)
    elif metric == "iou" or metric == "jac":
        score = tp / (tp + fp + fn + eps)
    elif metric == "overlap":
        score = tp / (torch.minimum(p, n) + eps)
    elif metric == "eff":
        score = tp / (p + eps)
    elif metric == "pur":
        score = tp / (tp + fp + eps)

    return score