import torch
from torch import Tensor, nn

from hepattn.components.dense import Dense
from hepattn.losses import cost_fns, loss_fns
from hepattn.tasks.base import Task


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

        Handles both binary object detection (num_classes=1) and multi-class object
        detection+classification (num_classes>1). Always includes an implicit null class
        for empty object slots. Outputs both logits and class probabilities.

        Class Layout:
        - Valid classes: indices 0, 1, 2, ..., num_classes-1
        - Null class: index num_classes (always the LAST class)
        - For binary case (num_classes=1): [valid_class=0, null_class=1]
        - For multi-class case: [class_0, class_1, ..., class_N-1, null_class=N]

        Args:
            name: Name of the task, used as the key to separate task outputs.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Dict specifying which losses to use. Keys are loss function names and values are loss weights.
            costs: Dict specifying which costs to use. Keys are cost function names and values are cost weights.
            net: Network that will be used for classification. For binary case (num_classes=1), should output 1 logit.
                For multi-class case (num_classes>1), should output num_classes+1 logits. Cannot be specified with dim.
            dim: Input dimension for creating a default Dense network. Cannot be specified with net.
            num_classes: Number of object classes (excluding null). For binary detection, use 1.
            class_weights: Weights for each non-null class in the loss.
            null_weight: Weight applied to the null class in the loss.
            mask_queries: Whether to mask queries based on predictions.
            has_intermediate_loss: Whether the task has intermediate loss.
            has_first_layer_loss: Whether the task has first layer loss (defaults to has_intermediate_loss if not specified).
        """
        if has_first_layer_loss and not has_intermediate_loss:
            raise ValueError("has_first_layer_loss=True requires has_intermediate_loss=True")

        super().__init__(has_intermediate_loss=has_intermediate_loss, has_first_layer_loss=has_first_layer_loss)

        # Validate net and dim arguments
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

        # Create network based on provided arguments
        self.output_size = 1 if num_classes == 1 else num_classes + 1
        if net is not None:
            if net.output_size != self.output_size:
                raise ValueError(
                    f"Network output size ({net.output_size}) doesn't match expected size "
                    f"for num_classes={num_classes} (expected {self.output_size}). "
                    f"For binary case (num_classes=1), net should output 1 logit. "
                    f"For multi-class case, net should output num_classes+1 logits."
                )

            self.net = net
        else:
            assert dim is not None
            self.net = Dense(input_size=dim, output_size=self.output_size)

        # Set up class weights: [class_0, class_1, ..., class_N, null_class]
        loss_weights = torch.ones(self.num_classes + 1, dtype=torch.float32)
        if class_weights is not None:
            if len(class_weights) != self.num_classes:
                raise ValueError(f"Length of class_weights ({len(class_weights)}) does not match number of classes ({self.num_classes})")
            loss_weights[: self.num_classes] = torch.tensor(class_weights, dtype=torch.float32)
        loss_weights[-1] = null_weight  # Last class is the null class
        self.register_buffer("loss_weights", loss_weights)

        # Define semantic output keys as properties
        self.logits_key = output_object + "_logit"
        self.probs_key = output_object + "_class_prob"
        self.inputs = [input_object + "_embed"]
        self.outputs = [self.logits_key, self.probs_key]

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute classification logits and class probabilities for each object query."""
        # Output both logits and class probabilities
        x_logits = self.net(x[self.input_object + "_embed"])

        # Handle both binary and multi-class cases
        if self.num_classes == 1:
            # Convert single logit to 2-class probabilities [valid_prob, null_prob]
            x_logits = x_logits.squeeze(-1)
            x_sigmoid = torch.sigmoid(x_logits)
            x_probs = torch.stack([x_sigmoid, 1 - x_sigmoid], dim=-1)
        else:
            x_probs = torch.softmax(x_logits, dim=-1)

        return {
            self.logits_key: x_logits,
            self.probs_key: x_probs,
        }

    def predict(self, outputs: dict[str, Tensor], threshold: float = 0.5, query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Derive predicted classes, valid flags, and valid probabilities from raw outputs."""
        class_probs = outputs[self.output_object + "_class_prob"].detach()

        valid_prob = 1 - class_probs[..., -1]
        classes = class_probs.argmax(-1)
        valid = classes < self.num_classes

        # Apply query_mask to mark padded queries as invalid
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

        # Get query_mask if present (for masking padded query losses)
        query_mask = targets.get("query_mask")

        if self.num_classes == 1:
            output = outputs[self.output_object + "_logit"]
            target = targets[self.target_object + "_valid"].float()
            sample_weight = target + self.loss_weights[-1] * (1 - target)

            # Mask out padded queries
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

        This task is for scenarios where all input objects are already known to be valid,
        and the goal is to classify them into specific categories. Unlike ObjectClassificationTask,
        this task does NOT handle object detection (valid/invalid) and assumes all inputs
        represent real objects that just need categorization.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            classes: List of class names (no null class - all inputs assumed valid).
            net: Network for classification. Should output len(classes) logits.
            class_weights: Weights for each class in the loss function.
            loss_weight: Weight for the loss function.
            threshold: Threshold for classification predictions.
            multilabel: Whether this is a multilabel classification.
            permute_loss: Whether to permute loss.
            has_intermediate_loss: Whether the task has intermediate loss.
            output_object: Name of the output object. Defaults to input_object if empty string.
            target_object: Name of the target object. Defaults to input_object if empty string.
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

        # Only consider valid targets - flatten both losses and mask
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
