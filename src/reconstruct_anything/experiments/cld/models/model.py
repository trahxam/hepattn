from reconstruct_anything.models import ModelWrapper
from reconstruct_anything.utils.metrics import compute_hit_assignment_metrics


class CLDReconstructor(ModelWrapper):
    def log_custom_metrics(self, preds, targets, stage):
        hits = ["vtxd", "trkr", "sihit", "ecal", "hcal", "vtb", "muon"]
        metrics = compute_hit_assignment_metrics(preds["final"], targets, hits, pred_object="flow", target_object="particle")
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()})
