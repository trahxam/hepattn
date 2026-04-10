from torch import Tensor


def compute_hit_assignment_metrics(
    preds: dict,
    targets: dict,
    hits: list[str],
    pred_object: str,
    target_object: str,
    working_points: list[float] | None = None,
) -> dict[str, Tensor]:
    """Compute efficiency and purity metrics for hit-to-object assignment tasks.

    This implements the standard matching metric used across tracking and
    particle flow experiments: for each working point, an object is "efficient"
    if the fraction of correctly assigned hits exceeds the threshold, and "pure"
    if the fraction of its predicted hits that are correct exceeds the threshold.

    Args:
        preds: Predictions dict (typically preds["final"]).
        targets: Targets dict from the dataloader.
        hits: List of hit type names (e.g., ["pix", "sct"] or ["vtxd", "trkr"]).
        pred_object: Name of the predicted object (e.g., "track", "flow", "pred").
        target_object: Name of the target object (e.g., "particle", "sudo").
        working_points: List of matching thresholds. Defaults to [0.5, 0.75, 1.0].

    Returns:
        Dict of metric_name -> scalar tensor, ready for self.log_dict().
    """
    if working_points is None:
        working_points = [0.5, 0.75, 1.0]

    metrics = {}

    for hit in hits:
        assignment_key = f"{pred_object}_{hit}_assignment"
        if assignment_key not in preds:
            continue

        pred_hit_masks = preds[assignment_key][f"{pred_object}_{hit}_valid"]
        true_hit_masks = targets[f"{target_object}_{hit}_valid"]

        pred_valid = preds[f"{pred_object}_valid"][f"{pred_object}_valid"]
        true_valid = targets[f"{target_object}_valid"]

        # Optionally restrict validity to objects with hits
        pred_valid = pred_valid & (pred_hit_masks.sum(-1) > 0)
        true_valid = true_valid & (true_hit_masks.sum(-1) > 0)

        # Mask hits by object validity
        pred_hit_masks = pred_hit_masks & pred_valid.unsqueeze(-1)
        true_hit_masks = true_hit_masks & true_valid.unsqueeze(-1)

        # Hit-level counts
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
        hit_p = pred_hit_masks.sum(-1)
        hit_t = true_hit_masks.sum(-1)

        both_valid = true_valid & pred_valid

        for wp in working_points:
            effs = ((hit_tp / hit_t) >= wp) & both_valid
            purs = ((hit_tp / hit_p) >= wp) & both_valid

            eff = effs.float().sum(-1) / true_valid.float().sum(-1)
            pur = purs.float().sum(-1) / pred_valid.float().sum(-1)

            metrics[f"p{wp}_{hit}_eff"] = eff.nanmean()
            metrics[f"p{wp}_{hit}_pur"] = pur.nanmean()

        # Count statistics (only logged once per hit, not per wp)
        metrics[f"num_{hit}_per_{pred_object}"] = pred_hit_masks.sum(-1).float()[pred_valid].mean()
        metrics[f"num_{hit}_per_{target_object}"] = true_hit_masks.sum(-1).float()[true_valid].mean()
        metrics[f"num_{pred_object}s"] = pred_valid.sum(-1).float().mean()
        metrics[f"num_{target_object}s"] = true_valid.sum(-1).float().mean()

    return metrics
