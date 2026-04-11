from reconstruct_anything.models import ModelWrapper
from reconstruct_anything.utils.metrics import compute_hit_assignment_metrics


class TIDEModel(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        if "pred_valid" not in preds.get("final", {}):
            return
        metrics = compute_hit_assignment_metrics(preds["final"], targets, hits=["pix", "sct"], pred_object="pred", target_object="sudo")
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()})
