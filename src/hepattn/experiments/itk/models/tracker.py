from hepattn.models import ModelWrapper
from hepattn.utils.metrics import compute_hit_assignment_metrics


class ITkTracker(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        metrics = compute_hit_assignment_metrics(preds["final"], targets, hits=["pixel", "strip"], pred_object="track", target_object="particle")
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()})
