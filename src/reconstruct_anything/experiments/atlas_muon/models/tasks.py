import torch
from torch import Tensor, nn

from reconstruct_anything.components.losses import mask_focal_loss
from reconstruct_anything.models.tasks import HitFilterTask


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
