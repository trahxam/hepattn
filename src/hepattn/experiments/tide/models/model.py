from torch import nn

from hepattn.models import ModelWrapper
from hepattn.utils.metrics import compute_hit_assignment_metrics


class TIDEModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def log_custom_metrics(self, preds, targets, stage):
        if "pred_valid" not in preds.get("final", {}):
            return
        metrics = compute_hit_assignment_metrics(
            preds["final"], targets, hits=["pix", "sct"], pred_object="pred", target_object="sudo"
        )
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()})
