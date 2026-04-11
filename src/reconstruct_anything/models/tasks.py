import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from reconstruct_anything.components.costs import cost_fns
from reconstruct_anything.components.dense import Dense
from reconstruct_anything.components.losses import loss_fns, mask_focal_loss

# Mapping of loss function names to torch.nn.functional loss functions
REGRESSION_LOSS_FNS = {
    "l1": torch.nn.functional.l1_loss,
    "l2": torch.nn.functional.mse_loss,
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
}

RegressionLossType = Literal["l1", "l2", "smooth_l1"]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Task(nn.Module, ABC):
    """Abstract base class for all tasks."""

    def __init__(self, has_intermediate_loss: bool, has_first_layer_loss: bool | None = None, permute_loss: bool = True):
        """Configure intermediate-loss and permutation behaviour for this task.

        Args:
            has_intermediate_loss: Whether to compute loss at each intermediate decoder layer.
            has_first_layer_loss: Whether to compute loss at the first decoder layer.
                Defaults to ``has_intermediate_loss`` when ``None``.
            permute_loss: Whether query outputs should be permuted (matched) before loss computation.
        """
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
        """Compute the forward pass of the task."""

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
        """Compute loss between outputs and targets."""

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Compute pairwise assignment costs between outputs and targets."""
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return cross-attention masks derived from task outputs."""
        return {}

    def key_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return key padding masks derived from task outputs."""
        return {}

    def query_mask(self, outputs: dict[str, Tensor], **kwargs) -> Tensor | None:
        """Return a boolean query mask, or None if all queries are active."""
        return None

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute task-specific evaluation metrics from predictions and targets."""
        return {}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class ObjectClassificationTask(Task):
    """Task for object detection and set-prediction classification with an implicit null class."""

    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        net: Dense | None = None,
        dim: int | None = None,
        num_classes: int = 1,
        class_weights: list[float] | None = None,
        null_weight: float = 1.0,
        mask_queries: bool = False,
        has_intermediate_loss: bool = True,
        has_first_layer_loss: bool = False,
    ):
        """Task for object detection and classification in set prediction scenarios.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Dict of loss function names to weights.
            costs: Dict of cost function names to weights.
            net: Network for classification. Cannot be specified with dim.
            dim: Input dimension for a default Dense network. Cannot be specified with net.
            num_classes: Number of object classes (excluding null).
            class_weights: Weights for each non-null class in the loss.
            null_weight: Weight applied to the null class in the loss.
            mask_queries: Whether to mask queries based on predictions.
            has_intermediate_loss: Whether the task has intermediate loss.
            has_first_layer_loss: Whether the task has first layer loss.
        """
        if has_first_layer_loss and not has_intermediate_loss:
            raise ValueError("has_first_layer_loss=True requires has_intermediate_loss=True")

        super().__init__(has_intermediate_loss=has_intermediate_loss, has_first_layer_loss=has_first_layer_loss)

        if net is not None and dim is not None:
            raise ValueError("Cannot specify both 'net' and 'dim'. Choose one.")
        if net is None and dim is None:
            raise ValueError("Must specify either 'net' or 'dim'.")

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.num_classes = num_classes
        self.mask_queries = mask_queries

        self.output_size = 1 if num_classes == 1 else num_classes + 1
        if net is not None:
            if net.output_size != self.output_size:
                raise ValueError(
                    f"Network output size ({net.output_size}) doesn't match expected size "
                    f"for num_classes={num_classes} (expected {self.output_size})."
                )
            self.net = net
        else:
            assert dim is not None
            self.net = Dense(input_size=dim, output_size=self.output_size)

        loss_weights = torch.ones(self.num_classes + 1, dtype=torch.float32)
        if class_weights is not None:
            if len(class_weights) != self.num_classes:
                raise ValueError(f"Length of class_weights ({len(class_weights)}) does not match number of classes ({self.num_classes})")
            loss_weights[: self.num_classes] = torch.tensor(class_weights, dtype=torch.float32)
        loss_weights[-1] = null_weight
        self.register_buffer("loss_weights", loss_weights)

        self.logits_key = output_object + "_logit"
        self.probs_key = output_object + "_class_prob"
        self.inputs = [input_object + "_embed"]
        self.outputs = [self.logits_key, self.probs_key]

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute classification logits and class probabilities for each object query."""
        x_logits = self.net(x[self.input_object + "_embed"])

        if self.num_classes == 1:
            x_logits = x_logits.squeeze(-1)
            x_sigmoid = torch.sigmoid(x_logits)
            x_probs = torch.stack([x_sigmoid, 1 - x_sigmoid], dim=-1)
        else:
            x_probs = torch.softmax(x_logits, dim=-1)

        return {self.logits_key: x_logits, self.probs_key: x_probs}

    def predict(self, outputs: dict[str, Tensor], threshold: float = 0.5, query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Derive predicted classes, valid flags, and valid probabilities from raw outputs."""
        class_probs = outputs[self.output_object + "_class_prob"].detach()

        valid_prob = 1 - class_probs[..., -1]
        classes = class_probs.argmax(-1)
        valid = classes < self.num_classes

        if query_mask is not None:
            valid = valid & query_mask

        return {
            f"{self.output_object}_class": classes,
            f"{self.output_object}_valid_prob": valid_prob,
            f"{self.output_object}_valid": valid,
        }

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute pairwise classification costs used for bipartite matching."""
        costs = {}

        if self.num_classes == 1:
            output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
            target = targets[self.target_object + "_valid"].to(torch.float32)
        else:
            output = outputs[self.output_object + "_class_prob"].detach().to(torch.float32)
            target = targets[self.target_object + "_class"].long()

        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute the weighted classification loss for binary or multi-class settings."""
        losses = {}
        query_mask = targets.get("query_mask")

        if self.num_classes == 1:
            output = outputs[self.output_object + "_logit"]
            target = targets[self.target_object + "_valid"].float()
            sample_weight = target + self.loss_weights[-1] * (1 - target)

            if query_mask is not None:
                sample_weight = sample_weight * query_mask.float()

            for loss_fn, loss_weight in self.losses.items():
                losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, sample_weight=sample_weight)
        else:
            output = outputs[self.output_object + "_logit"]
            target = targets[self.target_object + "_class"].long()

            for loss_fn, loss_weight in self.losses.items():
                losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=query_mask, weight=self.loss_weights)

        return losses

    def query_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> Tensor | None:
        """Return a boolean mask suppressing low-confidence queries, or None if masking is disabled."""
        if not self.mask_queries:
            return None
        class_probs = outputs[self.output_object + "_class_prob"].detach()
        return class_probs[..., -1] <= (1 - threshold)

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute efficiency, fake rate, and query count metrics for object detection."""
        pred_valid = preds[f"{self.output_object}_valid"].bool()
        true_valid = targets[f"{self.target_object}_valid"].bool()

        query_mask = targets.get("query_mask")
        if query_mask is not None:
            pred_valid = pred_valid & query_mask
            true_valid = true_valid & query_mask

        tp = (pred_valid & true_valid).sum()
        fp = (pred_valid & ~true_valid).sum()

        true_pos = true_valid.sum()
        total_pred = pred_valid.sum()

        eps = torch.tensor(1e-12, device=pred_valid.device)
        eff = tp / torch.maximum(true_pos, eps)
        fr = fp / torch.maximum(total_pred, eps)

        return {
            "num_queries": float(pred_valid.shape[1]) if query_mask is None else float(query_mask.sum()),
            "query_frac_pred_valid": pred_valid.float().mean() if query_mask is None else pred_valid[query_mask].float().mean(),
            "query_eff": eff,
            "query_fr": fr,
        }


class ClassificationTask(Task):
    """Standard classification task for objects already known to be valid."""

    def __init__(
        self,
        name: str,
        input_object: str,
        classes: list[str],
        net: nn.Module,
        output_object: str | None = None,
        target_object: str | None = None,
        class_weights: dict[str, float] | None = None,
        loss_weight: float = 1.0,
        threshold: float = 0.5,
        multilabel: bool = False,
        permute_loss: bool = True,
        has_intermediate_loss: bool = True,
    ):
        """Standard classification task for existing objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            classes: List of class names.
            net: Network for classification.
            output_object: Name of the output object. Defaults to input_object.
            target_object: Name of the target object. Defaults to input_object.
            class_weights: Weights for each class in the loss function.
            loss_weight: Weight for the loss function.
            threshold: Threshold for classification predictions.
            multilabel: Whether this is a multilabel classification.
            permute_loss: Whether to permute loss.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=permute_loss)

        self.name = name
        self.input_object = input_object
        self.output_object = output_object if output_object is not None else input_object
        self.target_object = target_object if target_object is not None else input_object
        self.classes = classes
        self.class_weights = class_weights
        self.loss_weight = loss_weight
        self.threshold = threshold
        self.multilabel = multilabel
        self.net = net

        if self.class_weights is not None:
            self.class_weights_values = torch.tensor([self.class_weights[class_name] for class_name in self.classes])

        self.inputs = [input_object + "_embed"]
        self.outputs = [self.output_object + "_logits"]

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute classification logits from object embeddings."""
        x = self.net(x[f"{self.input_object}_embed"])
        return {f"{self.output_object}_logits": x}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Derive per-class probabilities and binary predictions from logits."""
        logits = outputs[self.output_object + "_logits"].detach()
        result = {}

        if len(self.classes) == 1 and not self.multilabel:
            logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
            probs = torch.nn.functional.sigmoid(logits)
            result[self.output_object + "_" + self.classes[0] + "_prob"] = probs
            result[self.output_object + "_" + self.classes[0]] = probs >= self.threshold
        elif self.multilabel:
            probs = torch.nn.functional.sigmoid(logits)
            for i, class_name in enumerate(self.classes):
                result[self.output_object + "_" + class_name + "_prob"] = probs[..., i]
                result[self.output_object + "_" + class_name] = probs[..., i] >= self.threshold
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.nn.functional.one_hot(torch.argmax(logits, dim=-1), num_classes=len(self.classes)).bool()
            for i, class_name in enumerate(self.classes):
                result[self.output_object + "_" + class_name + "_prob"] = probs[..., i]
                result[self.output_object + "_" + class_name] = predictions[..., i]

        return result

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute cross-entropy or BCE loss over valid objects only."""
        logits = outputs[f"{self.output_object}_logits"]

        if len(self.classes) == 1 and not self.multilabel:
            target = targets[self.target_object + "_" + self.classes[0]].float()
            logits = logits.squeeze(-1) if logits.dim() > target.dim() else logits

            pos_weight = None
            if self.class_weights is not None:
                pos_weight = torch.tensor([self.class_weights[self.classes[0]]], dtype=logits.dtype, device=logits.device)

            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
        elif self.multilabel:
            target = torch.stack([targets[self.target_object + "_" + class_name] for class_name in self.classes], dim=-1).float()

            pos_weight = None
            if self.class_weights is not None:
                pos_weight = self.class_weights_values.type_as(target)

            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none").mean(dim=-1)
        else:
            target = torch.stack([targets[self.target_object + "_" + class_name] for class_name in self.classes], dim=-1).float()

            class_weights = None
            if self.class_weights is not None:
                class_weights = self.class_weights_values.type_as(target)

            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target.view(-1, target.shape[-1]),
                weight=class_weights,
                reduction="none",
            )

        valid_mask = targets[f"{self.target_object}_valid"].view(-1)
        losses = losses.view(-1)[valid_mask]
        return {"bce": self.loss_weight * losses.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute per-class efficiency and purity metrics over valid objects."""
        metrics = {}
        for class_name in self.classes:
            target = targets[f"{self.target_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()
            pred = preds[f"{self.output_object}_{class_name}"][targets[f"{self.target_object}_valid"]].bool()

            true_positives = float((target & pred).sum())
            false_positives = float((~target & pred).sum())

            metrics[f"{class_name}_eff"] = true_positives / target.sum()
            metrics[f"{class_name}_pur"] = true_positives / pred.sum()
            metrics[f"{class_name}_tp"] = true_positives
            metrics[f"{class_name}_fp"] = false_positives

        return metrics


# ---------------------------------------------------------------------------
# Hit filter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


class ObjectHitMaskTask(Task):
    """Task for predicting binary associations between reconstructed objects and constituent hits."""

    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        object_net: nn.Module | None = None,
        constituent_net: nn.Module | None = None,
        null_weight: float = 1.0,
        mask_attn: bool = True,
        target_field: str = "valid",
        logit_scale: float = 1.0,
        pred_threshold: float = 0.5,
        mask_attention_threshold: float | None = None,
        has_intermediate_loss: bool = True,
    ):
        """Task for predicting associations between objects and hits.

        Args:
            name: Name of the task.
            input_constituent: Name of the input constituent type.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Loss functions and their weights.
            costs: Cost functions and their weights.
            dim: Embedding dimension.
            object_net: Network applied to object embeddings to get mask tokens.
            constituent_net: Network applied to constituent embeddings.
            null_weight: Weight for null class.
            mask_attn: Whether to mask attention.
            target_field: Target field name.
            logit_scale: Scale for logits.
            pred_threshold: Prediction threshold.
            mask_attention_threshold: Threshold for attention masking. Defaults to pred_threshold.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.target_field = target_field

        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.constituent_net = constituent_net
        self.object_net = object_net or Dense(dim, dim)
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold
        self.mask_attention_threshold = mask_attention_threshold if mask_attention_threshold is not None else pred_threshold

        self.output_object_hit = output_object + "_" + input_constituent
        self.target_object_hit = target_object + "_" + input_constituent

        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute object-hit association logits via inner product of mask tokens and constituent embeddings."""
        mask_tokens = self.object_net(x[self.input_object + "_embed"])
        xs = x[self.input_constituent + "_embed"]
        if self.constituent_net:
            xs = self.constituent_net(xs)

        object_hit_logit = self.logit_scale * torch.einsum("bnc,bmc->bnm", mask_tokens, xs)

        if (valid_mask := x[f"{self.input_constituent}_valid"]) is not None:
            valid_mask = valid_mask.unsqueeze(-2).expand_as(object_hit_logit)
            object_hit_logit[~valid_mask] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs: dict[str, Tensor], threshold: float | None = None) -> dict[str, Tensor]:
        """Return a boolean cross-attention mask derived from predicted hit associations."""
        if not self.mask_attn:
            return {}
        thresh = threshold if threshold is not None else self.mask_attention_threshold
        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= thresh
        return {self.input_constituent: attn_mask}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return per-association probabilities and binary valid flags."""
        output = {}
        probs = outputs[self.output_object_hit + "_logit"].sigmoid().detach()
        valid = probs >= self.pred_threshold

        if query_mask is not None:
            valid = valid & query_mask.unsqueeze(-1)

        output[self.output_object_hit + "_valid_prob"] = probs
        output[self.output_object_hit + "_valid"] = valid

        return output

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute pairwise mask matching costs for bipartite assignment."""
        output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object_hit + "_" + self.target_field].detach().to(output.dtype)

        hit_pad = targets[self.input_constituent + "_valid"]

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target, input_pad_mask=hit_pad)
        return costs

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute weighted mask loss over valid objects and hit positions."""
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_" + self.target_field].type_as(output)

        hit_pad = targets[self.input_constituent + "_valid"]
        object_pad = targets[self.target_object + "_valid"]

        query_mask = targets.get("query_mask")
        if query_mask is not None:
            object_pad = object_pad & query_mask

        sample_weight = target + self.null_weight * (1 - target)
        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](
                output, target, object_valid_mask=object_pad, input_pad_mask=hit_pad, sample_weight=sample_weight
            )

        return losses


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class RegressionTask(Task):
    """Abstract base class for regression tasks that predict continuous target fields."""

    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        loss: RegressionLossType = "smooth_l1",
        has_intermediate_loss: bool = True,
    ):
        """Base class for regression tasks.

        Args:
            name: Name of the task.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.loss_fn_name = loss
        self.loss_fn = REGRESSION_LOSS_FNS[loss]
        self.k = len(fields)
        self.ndofs = self.k

        self.regression_key = output_object + "_regr"

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute regression predictions by passing embeddings through the task network."""
        latent = self.latent(x)
        return {self.regression_key: latent}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return per-field regression predictions from latent outputs."""
        latent = outputs[self.regression_key]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute mean regression loss over valid objects."""
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.regression_key]

        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)

        return {self.loss_fn_name: self.loss_weight * loss.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute mean absolute and normalised absolute residuals for each field."""
        metrics = {}
        for field in self.fields:
            pred = preds[self.output_object + "_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            abs_err = (pred - target).abs()
            metrics[field + "_abs_res"] = torch.mean(abs_err)
            metrics[field + "_abs_norm_res"] = torch.mean(abs_err / target.abs() + 1e-8)
        return metrics


class GaussianRegressionTask(Task):
    """Abstract base class for regression tasks that output a full Gaussian (mean + precision)."""

    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        has_intermediate_loss: bool = True,
    ):
        """Regression task with Gaussian output distribution.

        Args:
            name: Name of the task.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.cost_weight = cost_weight
        self.k = len(fields)
        self.ndofs = self.k + int(self.k * (self.k + 1) / 2)
        self.likelihood_norm = self.k * 0.5 * math.log(2 * math.pi)

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute Gaussian distribution parameters (mean and upper-triangular precision factor)."""
        latent = self.latent(x)
        k = self.k
        triu_idx = torch.triu_indices(k, k, device=latent.device)

        mu = latent[..., :k]
        u = torch.zeros(latent.size()[:-1] + torch.Size((k, k)), device=latent.device)
        u[..., triu_idx[0, :], triu_idx[1, :]] = latent[..., k:]

        ubar = u.clone()
        ubar[..., torch.arange(k), torch.arange(k)] = torch.exp(u[..., torch.arange(k), torch.arange(k)])

        return {self.output_object + "_mu": mu, self.output_object + "_u": u, self.output_object + "_ubar": ubar}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return per-field mean predictions and full precision matrix elements."""
        preds = outputs
        mu = outputs[self.output_object + "_mu"]
        ubar = outputs[self.output_object + "_ubar"]

        precs = torch.einsum("...kj,...kl->...jl", ubar, ubar)

        for i, field in enumerate(self.fields):
            preds[self.output_object + "_" + field] = mu[..., i]

        for i, field_i in enumerate(self.fields):
            for j, field_j in enumerate(self.fields):
                if i > j:
                    continue
                preds[field_i + "_" + field_j + "_prec"] = precs[..., i, j]

        return preds

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute negative log-likelihood loss under the predicted Gaussian distribution."""
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)

        z = torch.einsum("...ij,...j->...i", outputs[self.output_object + "_ubar"], y - outputs[self.output_object + "_mu"])
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(torch.diagonal(outputs[self.output_object + "_u"], offset=0, dim1=-2, dim2=-1), dim=-1)
        log_likelihood = self.likelihood_norm - 0.5 * zsq + jac

        log_likelihood *= targets[self.target_object + "_valid"].type_as(log_likelihood)
        return {"nll": -self.loss_weight * log_likelihood.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute RMSE, pull mean, and pull std metrics for each regression field."""
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        res = y - preds[self.output_object + "_mu"]
        z = torch.einsum("...ij,...j->...i", preds[self.output_object + "_ubar"], res)

        valid_mask = targets[self.target_object + "_valid"]

        metrics = {}
        for i, field in enumerate(self.fields):
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(res[..., i][valid_mask])))
            metrics[field + "_pull_mean"] = torch.mean(z[..., i][valid_mask])
            metrics[field + "_pull_std"] = torch.std(z[..., i][valid_mask])

        return metrics


class ObjectRegressionTask(RegressionTask):
    """Regression task that operates on per-object embeddings via a dense projection."""

    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        dim: int,
        loss: RegressionLossType = "smooth_l1",
        has_intermediate_loss: bool = True,
    ):
        """Regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [self.regression_key]
        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        """Produce raw latent regression vector from object embeddings."""
        return self.net(x[self.input_object + "_embed"])


class ObjectHitRegressionTask(RegressionTask):
    """Regression task that jointly encodes object and constituent hit embeddings via bilinear projection."""

    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        dim: int,
        loss: RegressionLossType = "smooth_l1",
        has_intermediate_loss: bool = True,
    ):
        """Regression task for object-constituent associations.

        Args:
            name: Name of the task.
            input_constituent: Name of the input constituent type.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
            loss: Type of loss function to use.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight, loss=loss, has_intermediate_loss=has_intermediate_loss)

        self.input_constituent = input_constituent
        self.input_object = input_object

        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.output_object + "_regr"]

        self.dim = dim
        self.dim_per_dof = self.dim // self.ndofs

        self.hit_net = Dense(dim, self.ndofs * self.dim_per_dof)
        self.object_net = Dense(dim, self.ndofs * self.dim_per_dof)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        """Compute per-object per-hit regression latents via factored bilinear projection."""
        x_obj = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_constituent + "_embed"])

        x_obj = x_obj.reshape(x_obj.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))

        x_obj_hit = torch.einsum("...nie,...mie->...nmi", x_obj, x_hit)

        x_obj_hit *= x[self.input_constituent + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
        return x_obj_hit
