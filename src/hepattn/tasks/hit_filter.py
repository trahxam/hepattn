from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.components.dense import Dense
from hepattn.losses import mask_focal_loss
from hepattn.tasks.base import Task


class HitFilterTask(Task):
    """Task for classifying individual hits as belonging to reconstructable objects or noise."""

    def __init__(
        self,
        name: str,
        input_object: str,
        target_field: str,
        dim: int,
        threshold: float = 0.1,
        mask_keys: bool = False,
        loss_fn: Literal["bce", "focal", "both"] = "bce",
        has_intermediate_loss: bool = True,
    ):
        """Task used for classifying whether constituents belong to reconstructable objects or not.

        Args:
            name: Name of the task.
            input_object: Name of the constituent type.
            target_field: Name of the target field to predict.
            dim: Embedding dimension.
            threshold: Threshold for classification.
            mask_keys: Whether to mask keys.
            loss_fn: Loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)

        self.name = name
        self.input_object = input_object
        self.target_field = target_field
        self.dim = dim
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.mask_keys = mask_keys

        # Internal
        self.input_objects = [f"{input_object}_embed"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute per-hit classification logits."""
        x_logit = self.net(x[f"{self.input_object}_embed"])
        return {f"{self.input_object}_logit": x_logit.squeeze(-1)}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return per-hit probabilities and binary valid predictions."""
        probs = outputs[f"{self.input_object}_logit"].sigmoid()
        return {
            f"{self.input_object}_{self.target_field}_prob": probs,
            f"{self.input_object}_{self.target_field}": probs >= self.threshold,
        }

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute BCE or focal loss for hit classification."""
        output = outputs[f"{self.input_object}_logit"]
        target = targets[f"{self.input_object}_{self.target_field}"].type_as(output)

        if self.loss_fn == "bce":
            pos_weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "focal":
            loss = mask_focal_loss(output, target)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "both":
            pos_weight = 1 / target.float().mean()
            bce_loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
            focal_loss_value = mask_focal_loss(output, target)
            return {
                f"{self.input_object}_bce": bce_loss,
                f"{self.input_object}_focal": focal_loss_value,
            }
        raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def key_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> dict[str, Tensor]:
        """Return a key padding mask suppressing low-confidence hits, or empty dict if disabled."""
        if not self.mask_keys:
            return {}

        return {self.input_object: outputs[f"{self.input_object}_logit"].detach().sigmoid() >= threshold}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute hit-level accuracy, recall, and precision metrics."""
        expected_key = f"{self.input_object}_{self.target_field}"
        pred = preds[expected_key]
        true = targets[expected_key]

        tp = (pred * true).sum()
        tn = ((~pred) * (~true)).sum()

        return {
            "nh_total_pre": float(pred.shape[1]),
            "nh_total_post": float(pred.sum()),
            "nh_pred_true": pred.float().sum(),
            "nh_pred_false": (~pred).float().sum(),
            "nh_valid_pre": true.float().sum(),
            "nh_valid_post": (pred & true).float().sum(),
            "nh_noise_pre": (~true).float().sum(),
            "nh_noise_post": (pred & ~true).float().sum(),
            "acc": (pred == true).half().mean(),
            "valid_recall": tp / true.sum(),
            "valid_precision": tp / pred.sum(),
            "noise_recall": tn / (~true).sum(),
            "noise_precision": tn / (~pred).sum(),
        }


class HitFilterTaskBatched(HitFilterTask):
    """Batched variant of `HitFilterTask`.

    This subclass overrides the `loss` method to support batched inputs where
    hit tensors are padded to a common length. It uses a provided valid mask
    in `targets` (key: `{input_object}_valid`) to ignore padded positions
    when computing losses.
    """

    def loss(self, outputs: dict, targets: dict) -> dict:
        """Compute loss over valid (unpadded) hits only, using the stored valid mask."""
        output = outputs[f"{self.input_object}_logit"]
        target = targets[f"{self.input_object}_{self.target_field}"].type_as(output)

        valid_mask = targets.get(f"{self.input_object}_valid", None)

        if valid_mask is not None:
            output = output[valid_mask]
            target = target[valid_mask]

        if self.loss_fn == "bce":
            target_mean = target.float().mean()
            weight = 1 / target_mean if target_mean > 0 else torch.tensor(1.0, device=output.device)
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "focal":
            loss = mask_focal_loss(output, target)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "both":
            target_mean = target.float().mean()
            weight = 1 / target_mean if target_mean > 0 else torch.tensor(1.0, device=output.device)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            focal_loss_value = mask_focal_loss(output, target)
            return {
                f"{self.input_object}_bce": bce_loss,
                f"{self.input_object}_focal": focal_loss_value,
            }
        raise ValueError(f"Unknown loss function: {self.loss_fn}")
