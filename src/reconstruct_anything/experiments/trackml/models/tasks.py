import torch
from torch import Tensor

from reconstruct_anything.components.dense import Dense
from reconstruct_anything.models.tasks import Task


class IoUPredictionTask(Task):
    """Task that predicts the IoU of mask predictions for use as a confidence score."""

    def __init__(
        self,
        name: str,
        input_object: str,
        mask_task_name: str,
        mask_logit_key: str,
        target_mask_key: str,
        dim: int,
        loss_weight: float = 1.0,
        input_constituent: str | None = None,
        target_field: str = "valid",
    ):
        """Task for predicting IoU of mask predictions.

        This task computes the IoU between predicted and target masks, and trains
        a network to predict this IoU value. It only runs on the final decoder layer
        (has_intermediate_loss=False) to avoid computational overhead during intermediate
        decoder layers.

        Args:
            name: Name of the task.
            input_object: Name of the input object (e.g., "particle").
            mask_task_name: Name of the task that produces mask logits (e.g., "track_hit_valid").
            mask_logit_key: Key to read mask logits from the mask task's outputs (e.g., "track_hit_logit").
            target_mask_key: Base key for target mask (e.g., "particle_hit"), will be combined with target_field.
            dim: Embedding dimension.
            loss_weight: Weight for the IoU MSE loss.
            input_constituent: Name of the constituent type (e.g., "hit"), used for validity masking. If None, inferred from mask_logit_key.
            target_field: Target field name (default: "valid").
        """
        super().__init__(has_intermediate_loss=False)

        self.name = name
        self.input_object = input_object
        self.mask_task_name = mask_task_name
        self.mask_logit_key = mask_logit_key
        self.target_mask_key = target_mask_key
        self.target_field = target_field
        self.loss_weight = loss_weight
        self.dim = dim

        if input_constituent is None:
            parts = mask_logit_key.replace("_logit", "").split("_")
            if len(parts) >= 2:
                self.input_constituent = parts[-1]
            else:
                raise ValueError(f"Cannot infer input_constituent from mask_logit_key '{mask_logit_key}'. Please provide it explicitly.")
        else:
            self.input_constituent = input_constituent

        self.iou_net = Dense(dim, 1)

        self.inputs = [input_object + "_embed"]
        self.outputs = [input_object + "_iou_logit"]

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute per-object IoU logits from object embeddings."""
        iou_logit = self.iou_net(x[self.input_object + "_embed"]).squeeze(-1)
        return {self.input_object + "_iou_logit": iou_logit}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return predicted IoU scores as sigmoid-activated probabilities."""
        iou = outputs[self.input_object + "_iou_logit"].detach().sigmoid()

        if query_mask is not None:
            iou = iou * query_mask.float()

        return {self.input_object + "_iou": iou}

    def calculate_iou(self, pred_probs: Tensor, target: Tensor) -> Tensor:
        """Calculate IoU between predicted probabilities and target mask."""
        intersection = (pred_probs * target).sum(dim=-1)
        union = pred_probs.sum(dim=-1) + target.sum(dim=-1) - intersection
        return intersection / (union + 1e-6)

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute MSE loss between predicted IoU scores and true mask IoU values."""
        if layer_outputs is None or self.mask_task_name not in layer_outputs:
            raise ValueError(f"Mask task '{self.mask_task_name}' not found in layer_outputs. Make sure the mask task runs before IoUPredictionTask.")

        mask_task_outputs = layer_outputs[self.mask_task_name]
        if self.mask_logit_key not in mask_task_outputs:
            raise ValueError(
                f"Mask logits key '{self.mask_logit_key}' not found in task '{self.mask_task_name}' outputs. "
                f"Available keys: {list(mask_task_outputs.keys())}"
            )

        mask_logits = mask_task_outputs[self.mask_logit_key]
        pred_probs = mask_logits.sigmoid()

        target = targets[self.target_mask_key + "_" + self.target_field].type_as(mask_logits)

        iou_target = self.calculate_iou(pred_probs, target)

        iou_pred = outputs[self.input_object + "_iou_logit"].sigmoid()

        object_pad = targets.get(self.input_object + "_valid")
        query_mask = targets.get("query_mask")
        if object_pad is not None:
            valid_mask = object_pad
            if query_mask is not None:
                valid_mask = valid_mask & query_mask
            iou_target = iou_target[valid_mask]
            iou_pred = iou_pred[valid_mask]
        elif query_mask is not None:
            iou_target = iou_target[query_mask]
            iou_pred = iou_pred[query_mask]

        iou_loss = torch.nn.functional.mse_loss(iou_pred, iou_target.detach())
        return {"iou_mse": self.loss_weight * iou_loss}
