import torch
from torch import Tensor, nn

from hepattn.components.dense import Dense
from hepattn.losses import cost_fns, loss_fns
from hepattn.tasks.base import Task


class ObjectHitMaskTask(Task):
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
            input_constituent: Name of the input constituent type (traditionally hits in tracking).
                For unified decoding, use "key" to access merged embeddings.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Loss functions and their weights.
            costs: Cost functions and their weights.
            dim: Embedding dimension.
            object_net: Get mask tokens from object embeddings
            constituent_net: Get constituent mask tokens from constituent embeddings.
                This is NOT RECOMMENDED - whatever you do, don't use an output activation.
            null_weight: Weight for null class.
            mask_attn: Whether to mask attention.
            target_field: Target field name.
            logit_scale: Scale for logits.
            pred_threshold: Prediction threshold.
            mask_attention_threshold: Threshold for attention masking. Defaults to pred_threshold if None.
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
        if not self.mask_attn:
            return {}

        thresh = threshold if threshold is not None else self.mask_attention_threshold
        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= thresh
        return {self.input_constituent: attn_mask}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        output = {}
        probs = outputs[self.output_object_hit + "_logit"].sigmoid().detach()
        valid = probs >= self.pred_threshold

        if query_mask is not None:
            valid = valid & query_mask.unsqueeze(-1)

        output[self.output_object_hit + "_valid_prob"] = probs
        output[self.output_object_hit + "_valid"] = valid

        return output

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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
