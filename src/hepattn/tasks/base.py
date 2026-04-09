from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

# Mapping of loss function names to torch.nn.functional loss functions
REGRESSION_LOSS_FNS = {
    "l1": torch.nn.functional.l1_loss,
    "l2": torch.nn.functional.mse_loss,
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
}

# Define the literal type for regression losses based on the dictionary keys
RegressionLossType = Literal["l1", "l2", "smooth_l1"]


class Task(nn.Module, ABC):
    """Abstract base class for all tasks.

    A task represents a specific learning objective (e.g., classification, regression)
    that can be trained as part of a multi-task learning setup.
    """

    def __init__(self, has_intermediate_loss: bool, has_first_layer_loss: bool | None = None, permute_loss: bool = True):
        super().__init__()
        self.has_intermediate_loss = has_intermediate_loss
        self.has_first_layer_loss = has_first_layer_loss if has_first_layer_loss is not None else has_intermediate_loss
        self.permute_loss = permute_loss

    def should_run_at_layer(self, layer_index: int) -> bool:
        """Check if the task should run at the given decoder layer index."""
        if not self.has_intermediate_loss:
            return False
        return not (layer_index == 0 and not self.has_first_layer_loss)

    def should_permute_outputs(self, layer_name: str, layer_outputs: dict) -> bool:
        """Check if the task outputs should be permuted for matching at this layer."""
        if not self.permute_loss:
            return False
        return self.name in layer_outputs

    @abstractmethod
    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute the forward pass of the task.

        Args:
            x: Dictionary of embeddings and features.
            outputs: Optional dictionary of outputs from other tasks at the current layer.
                     Allows tasks to read outputs from previously executed tasks.
        """

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return predictions from model outputs."""

    @abstractmethod
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute loss between outputs and targets.

        Args:
            outputs: This task's outputs.
            targets: Target tensors.
            layer_outputs: Optional dict of all task outputs at this layer,
                keyed by task name. Useful for tasks that need outputs from other tasks.
        """

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def key_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def query_mask(self, outputs: dict[str, Tensor], **kwargs) -> Tensor | None:
        return None

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}
