import torch
import torch.nn.functional as F


def _reduce_object_loss(loss_per_object, object_valid_mask=None):
    """Average per-object losses, optionally selecting only valid objects."""
    if object_valid_mask is not None:
        loss_per_object = loss_per_object[object_valid_mask]

    if loss_per_object.numel() == 0:
        return loss_per_object.new_tensor(0.0)

    return loss_per_object.mean()


def object_bce_loss(pred_logits, targets, sample_weight=None):
    """Loss function for binary object classification.

    Args:
        pred_logits: [batch_size, num_objects] - predicted logits for binary classification
        targets: [batch_size, num_objects] - ground truth class labels
        sample_weight: Optional sample weights for each element

    Returns:
        loss: Scalar tensor representing the binary cross-entropy loss
    """
    return F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight)


def object_bce_cost(pred_logits, targets):
    """Compute batched binary object classification cost for object matching.
    Approximate the CE loss using -probs[target_class].
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects] - predicted logits for binary classification
        targets: [batch_size, num_objects] - ground truth class labels

    Returns:
        cost_class: [batch_size, num_objects, num_objects] - classification cost matrix
    """
    probs = pred_logits.sigmoid().unsqueeze(-1)
    targets = targets.unsqueeze(1)
    return -probs * targets - (1 - probs) * (1 - targets)


def object_ce_loss(pred_probs, true, mask=None, weight=None):  # noqa: ARG001
    losses = F.cross_entropy(pred_probs.flatten(0, 1), true.flatten(0, 1), weight=weight)
    return losses.mean()


def object_ce_cost(pred_logits, targets):
    """Compute batched multiclass object classification cost for object matching.
    Approximate the CE loss using -probs[target_class].
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_classes] - predicted logits (num_classes>1)
        targets: [batch_size, num_objects] - ground truth class labels

    Returns:
        cost_class: [batch_size, num_objects, num_targets] - classification cost matrix
    """
    assert pred_logits.shape[-1] > 1
    probs = torch.softmax(pred_logits, dim=-1)

    batch_size, num_queries, _ = probs.shape
    num_targets = targets.size(1)

    index = targets.unsqueeze(1).expand(batch_size, num_queries, num_targets)
    return -torch.gather(probs, dim=2, index=index)


def mask_dice_loss(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, sample_weight=None):  # noqa: ARG001
    """Compute the DICE loss for binary masks.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Not used by DICE!

    Returns:
        loss: Scalar tensor representing the DICE loss
    """
    probs = pred_logits.sigmoid()
    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1).to(probs.dtype)
        probs = probs * input_mask
        targets = targets * input_mask

    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    denom_safe = denominator.clamp_min(1e-8)
    dice = numerator / denom_safe
    # If both are empty, treat as perfect match
    dice = torch.where(denominator == 0, torch.ones_like(dice), dice)
    loss = 1 - dice
    return _reduce_object_loss(loss, object_valid_mask)


def mask_iou_loss(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, sample_weight=None, eps=1e-6):  # noqa: ARG001
    """Compute the IoU loss for binary masks.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Not used by IoU
        eps: Small value to avoid division by zero

    Returns:
        loss: Scalar tensor representing the IoU loss
    """
    probs = pred_logits.sigmoid()
    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1).to(probs.dtype)
        probs = probs * input_mask
        targets = targets * input_mask

    intersection = (probs * targets).sum(-1)
    union = probs.sum(-1) + targets.sum(-1) - intersection
    iou = (intersection + eps) / (union + eps)
    # If both are empty, treat as perfect match
    iou = torch.where(union == 0, torch.ones_like(iou), iou)
    loss = 1 - iou
    return _reduce_object_loss(loss, object_valid_mask)


def mask_dice_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None):
    """Compute DICE costs.
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Not used by DICE!

    Returns:
        cost: [batch_size, num_objects, num_objects] - DICE cost matrix
    """
    assert sample_weight is None
    inputs = pred_logits.sigmoid()

    # apply input padding mask
    if input_pad_mask is not None:
        inputs = inputs * input_pad_mask.unsqueeze(1)

    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)
    denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)
    denom_safe = denominator.clamp_min(1e-8)
    dice = numerator / denom_safe
    # If both are empty, treat as perfect match
    dice = torch.where(denominator == 0, torch.ones_like(dice), dice)
    return 1 - dice


def mask_iou_cost(pred_logits, targets, input_pad_mask=None, eps=1e-6):
    # Apply input padding mask
    probs = pred_logits.sigmoid()
    if input_pad_mask is not None:
        probs = probs * input_pad_mask.unsqueeze(1)

    num_pred = probs.sum(-1).unsqueeze(2)
    num_targets = targets.sum(-1).unsqueeze(1)

    # Context manager necessary to overwrite global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        intersection = torch.einsum("bnc,bmc->bnm", probs, targets)
        return 1 - (intersection + eps) / (eps + num_pred + num_targets - intersection)


def mask_fm_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None, eps=1e-8):
    """Compute Fowlkes-Mallows costs for binary masks.
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Not used
        eps: Small value to avoid division by zero

    Returns:
        cost: [batch_size, num_objects, num_objects] - Fowlkes-Mallows cost matrix
    """
    assert sample_weight is None
    probs = pred_logits.sigmoid()
    targets = targets.type_as(probs)

    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1).to(probs.dtype)
        probs = probs * input_mask
        targets = targets * input_mask

    # Context manager necessary to overwrite global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        tp = torch.einsum("bnc,bmc->bnm", probs, targets)
        num_pred = probs.sum(-1).unsqueeze(2)
        num_targets = targets.sum(-1).unsqueeze(1)

        denominator = torch.sqrt((num_pred * num_targets).clamp_min(0.0))
        fm = tp / denominator.clamp_min(eps)
        both_empty = (num_pred == 0) & (num_targets == 0)
        fm = torch.where(denominator == 0, both_empty.to(fm.dtype), fm)
        return 1 - fm


def mask_focal_loss(pred_logits, targets, gamma=2.0, object_valid_mask=None, input_pad_mask=None, sample_weight=None):
    """Compute the focal loss for binary classification.

    Args:
        pred_logits: [batch_size, num_objects] - predicted logits for binary classification
        targets: [batch_size, num_objects] - ground truth class labels
        gamma: Focusing parameter for the focal loss
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Optional sample weights for each element

    Returns:
        loss: Scalar tensor representing the focal loss
    """
    pred = pred_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.type_as(pred_logits), weight=sample_weight, reduction="none")

    # Apply input padding mask
    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1).to(ce_loss.dtype)
        ce_loss = ce_loss * input_mask
        pred = pred * input_mask
        targets = targets * input_mask

    p_t = pred * targets + (1 - pred) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Normalise by valid elements such that each mask contributes equally
    if input_pad_mask is not None:
        valid_counts = input_pad_mask.sum(-1, keepdim=True).to(loss.dtype)
        loss_per_object = loss.sum(-1) / valid_counts.clamp_min(1.0)
    else:
        loss_per_object = loss.mean(-1)

    return _reduce_object_loss(loss_per_object, object_valid_mask)


def mask_focal_cost(pred_logits, targets, gamma=2.0, input_pad_mask=None, sample_weight=None):
    """Compute focal costs for binary masks.
    Invalid objects are handled later in the matching process.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        gamma: Focusing parameter for the focal loss
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Optional sample weights for each element (effectively the focal alpha)

    Returns:
        cost: [batch_size, num_objects, num_objects] - focal cost matrix
    """
    pred = pred_logits.sigmoid()
    focal_pos = ((1 - pred) ** gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred), weight=sample_weight, reduction="none")
    focal_neg = (pred**gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred), weight=sample_weight, reduction="none")

    # Apply input padding mask
    if input_pad_mask is not None:
        focal_pos = focal_pos * input_pad_mask.unsqueeze(1)
        focal_neg = focal_neg * input_pad_mask.unsqueeze(1)

    # Context manager necessary to overwride global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        return torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - targets))


def mask_bce_loss(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, sample_weight=None):
    """Compute the binary cross-entropy loss for binary masks.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Optional sample weights for each element.  Recommended to use focal instead.

    Returns:
        loss: Scalar tensor representing the binary cross-entropy loss
    """
    loss = F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight, reduction="none")

    # Apply input padding mask
    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1).to(loss.dtype)
        loss = loss * input_mask

    # Normalise by valid elements such that each mask contributes equally
    if input_pad_mask is not None:
        valid_counts = input_pad_mask.sum(-1, keepdim=True).to(loss.dtype)
        loss_per_object = loss.sum(-1) / valid_counts.clamp_min(1.0)
    else:
        loss_per_object = loss.mean(-1)

    return _reduce_object_loss(loss_per_object, object_valid_mask)


def mask_bce_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None):
    pred_logits = torch.clamp(pred_logits, -100, 100)

    pos = F.binary_cross_entropy_with_logits(
        pred_logits, torch.ones_like(pred_logits),
        weight=sample_weight, reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        pred_logits, torch.zeros_like(pred_logits),
        weight=sample_weight, reduction="none"
    )

    if input_pad_mask is not None:
        pos = pos * input_pad_mask.unsqueeze(1)
        neg = neg * input_pad_mask.unsqueeze(1)
        norm = input_pad_mask.sum(-1).clamp_min(1.0)          # [b]
    else:
        norm = pred_logits.new_full((pred_logits.size(0),), pred_logits.size(-1))  # [b]

    with torch.autocast(device_type="cuda", enabled=False):
        cost = (
            torch.einsum("bnc,bmc->bnm", pos, targets) +
            torch.einsum("bnc,bmc->bnm", neg, (1 - targets))
        )
        cost = cost / norm[:, None, None]
        return cost


def mask_recall_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None, eps=1e-8):
    """Recall-only cost: 1 - intersection / num_target_hits.
    Only penalizes false negatives (missed target hits), not false positives (extra predicted hits).
    This avoids penalizing a query for predicting hits that belong to overlapping nearby particles.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits for binary masks
        targets: [batch_size, num_objects, num_inputs] - ground truth binary masks
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Not used
        eps: Small value to avoid division by zero

    Returns:
        cost: [batch_size, num_objects, num_objects] - recall cost matrix
    """
    assert sample_weight is None
    inputs = pred_logits.sigmoid()

    if input_pad_mask is not None:
        inputs = inputs * input_pad_mask.unsqueeze(1)

    with torch.autocast(device_type="cuda", enabled=False):
        intersection = torch.einsum("bnc,bmc->bnm", inputs, targets)
        num_targets = targets.sum(-1).unsqueeze(1)  # (B, 1, M)
        recall = intersection / num_targets.clamp_min(eps)
        # If target is empty, treat as perfect match (nothing to recall)
        recall = torch.where(num_targets == 0, torch.ones_like(recall), recall)
        return 1 - recall


def mask_bce_balanced_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None):
    """Balanced BCE cost with equal weight on positive and negative targets."""
    pred_logits = torch.clamp(pred_logits, -100, 100)

    pos = F.binary_cross_entropy_with_logits(
        pred_logits,
        torch.ones_like(pred_logits),
        weight=sample_weight,
        reduction="none",
    )
    neg = F.binary_cross_entropy_with_logits(
        pred_logits,
        torch.zeros_like(pred_logits),
        weight=sample_weight,
        reduction="none",
    )

    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1).to(targets.dtype)
        pos = pos * input_mask
        neg = neg * input_mask
        target_pos = targets * input_mask
        target_neg = (1 - targets) * input_mask
    else:
        target_pos = targets
        target_neg = 1 - targets

    with torch.autocast(device_type="cuda", enabled=False):
        pos_sum = torch.einsum("bnc,bmc->bnm", pos, target_pos)
        neg_sum = torch.einsum("bnc,bmc->bnm", neg, target_neg)

        pos_count = target_pos.sum(-1).unsqueeze(1)
        neg_count = target_neg.sum(-1).unsqueeze(1)

        pos_avg = pos_sum / pos_count.clamp_min(1.0)
        neg_avg = neg_sum / neg_count.clamp_min(1.0)

        has_pos = (pos_count > 0).to(pos_avg.dtype)
        has_neg = (neg_count > 0).to(neg_avg.dtype)
        denom = (has_pos + has_neg).clamp_min(1.0)

        return (has_pos * pos_avg + has_neg * neg_avg) / denom


def kl_div_loss(pred_logits, true, mask=None, weight=None, eps=1e-8):  # noqa: ARG001
    loss = -true * torch.log(pred_logits + eps)
    # if weight is not None:
    #     loss *= weight
    if mask is not None:
        loss = loss[mask]
    return loss.mean()


# Context manager necessary to overwride global autocast to ensure float32 cost is returned
# @torch.autocast(device_type="cuda", enabled=False)
def kl_div_cost(pred_logits, true, eps=1e-8):
    return (-true[:, None, :] * torch.log(pred_logits[:, :, None] + eps)).mean(-1)


def mask_kl_div_loss(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, sample_weight=None, eps=1e-8):  # noqa: ARG001
    """KL divergence loss for hit-object assignment (recommend using energy_fractions as input).

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits
        targets: [batch_size, num_objects, num_inputs] - ground truth
        object_valid_mask: [batch_size, num_objects] - mask indicating valid target objects
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: not used
        eps: Small value to avoid log(0)

    Returns:
        loss: KL loss
    """
    if input_pad_mask is not None:
        input_mask = input_pad_mask.unsqueeze(1)
        pred_logits = pred_logits.masked_fill(~input_mask, float("-inf"))
        targets = targets * input_mask.to(targets.dtype)
        # Renormalise to keep targets sums to 1 for each object
        targets = targets / (targets.sum(-1, keepdim=True) + eps)

    pred_probs = torch.softmax(pred_logits, dim=-1)
    loss = -targets * torch.log(pred_probs + eps)

    # Apply input padding mask such that each mask contributes equally
    if input_pad_mask is not None:
        valid_counts = input_pad_mask.sum(-1, keepdim=True).to(loss.dtype)
        loss_per_object = loss.sum(-1) / (valid_counts + eps)
    else:
        loss_per_object = loss.mean(-1)

    return _reduce_object_loss(loss_per_object, object_valid_mask)


def mask_kl_div_cost(pred_logits, targets, input_pad_mask=None, sample_weight=None, eps=1e-8):  # noqa: ARG001
    """Compute KL costs.

    Args:
        pred_logits: [batch_size, num_objects, num_inputs] - predicted logits
        targets: [batch_size, num_objects, num_inputs] - ground truth
        input_pad_mask: [batch_size, num_inputs] - mask indicating valid inputs
        sample_weight: Not used
        eps: Small value to avoid log(0)

    Returns:
        cost: [batch_size, num_objects, num_objects] - KL cost
    """
    if input_pad_mask is not None:
        pred_logits = pred_logits.masked_fill(~input_pad_mask.unsqueeze(1), float("-inf"))
        targets = targets * input_pad_mask.unsqueeze(1)
        # Renormalise to keep targets sums to 1 for each object
        targets = targets / (targets.sum(-1, keepdim=True) + eps)

    pred_probs = torch.softmax(pred_logits, dim=-1)

    # Context manager necessary to overwrite global autocast to ensure float32 cost is returned
    with torch.autocast(device_type="cuda", enabled=False):
        log_pred = torch.log(pred_probs + eps)
        return torch.einsum("bnm,btm->bnt", -log_pred, targets)


def regr_mse_loss(pred, targets):
    return torch.nn.functional.mse_loss(pred, targets, reduction="none")


def regr_smooth_l1_loss(pred, targets):
    return torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none")


def regr_mse_cost(pred, targets):
    return torch.nn.functional.mse_loss(pred.unsqueeze(-2), targets.unsqueeze(-3), reduction="none")


def regr_smooth_l1_cost(pred, targets):
    return torch.nn.functional.smooth_l1_loss(pred.unsqueeze(-2), targets.unsqueeze(-3), reduction="none")


cost_fns = {
    "object_bce": torch.compile(object_bce_cost, dynamic=True),
    "object_ce": torch.compile(object_ce_cost, dynamic=True),
    "mask_fm": torch.compile(mask_fm_cost, dynamic=True),
    "mask_bce": torch.compile(mask_bce_cost, dynamic=True),
    "mask_bce_balanced": torch.compile(mask_bce_balanced_cost, dynamic=True),
    "mask_dice": torch.compile(mask_dice_cost, dynamic=True),
    "mask_recall": torch.compile(mask_recall_cost, dynamic=True),
    "mask_focal": torch.compile(mask_focal_cost, dynamic=True),
    "mask_iou": torch.compile(mask_iou_cost, dynamic=True),
    "kl_div": torch.compile(kl_div_cost, dynamic=True),
    "mask_kl_div": torch.compile(mask_kl_div_cost, dynamic=True),
}

loss_fns = {
    "object_bce": torch.compile(object_bce_loss, dynamic=True),
    "object_ce": torch.compile(object_ce_loss, dynamic=True),
    "mask_bce": torch.compile(mask_bce_loss, dynamic=True),
    "mask_dice": torch.compile(mask_dice_loss, dynamic=True),
    "mask_iou": torch.compile(mask_iou_loss, dynamic=True),
    "mask_focal": torch.compile(mask_focal_loss, dynamic=True),
    "kl_div": torch.compile(kl_div_loss, dynamic=True),
    "mask_kl_div": torch.compile(mask_kl_div_loss, dynamic=True),
}
