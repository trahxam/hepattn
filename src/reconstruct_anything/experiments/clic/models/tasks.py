import torch
from torch import Tensor, nn

from reconstruct_anything.components.costs import cost_fns
from reconstruct_anything.components.losses import loss_fns
from reconstruct_anything.models.tasks import RegressionLossType, RegressionTask, Task
from reconstruct_anything.utils.masks import topk_attn
from reconstruct_anything.utils.scaling import FeatureScaler


class IncidenceRegressionTask(Task):
    """Task that predicts a soft incidence matrix between objects and their constituent hits."""

    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module,
        node_net: nn.Module | None = None,
        has_intermediate_loss: bool = True,
    ):
        """Incidence regression task.

        Args:
            name: Name of the task.
            input_constituent: Name of the input hit object.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Loss functions and their weights.
            costs: Cost functions and their weights.
            net: Network for object embedding.
            node_net: Network for node embedding.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)
        self.name = name
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.net = net
        self.node_net = node_net if node_net is not None else nn.Identity()

        self.incidence_key = self.output_object + "_incidence"
        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.incidence_key]

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Compute the soft incidence matrix via dot-product between object and constituent embeddings."""
        x_object = self.net(x[self.input_object + "_embed"])
        x_hit = self.node_net(x[self.input_constituent + "_embed"])

        incidence_pred = torch.einsum("bqe,ble->bql", x_object, x_hit)
        incidence_pred = incidence_pred.softmax(dim=1) * x[self.input_constituent + "_valid"].unsqueeze(1).expand_as(incidence_pred)

        return {self.incidence_key: incidence_pred}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return the detached predicted incidence matrix."""
        return {self.output_object + "_incidence": outputs[self.incidence_key].detach()}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute pairwise incidence matching costs for bipartite assignment."""
        output = outputs[self.incidence_key].detach().to(torch.float32)
        target = targets[self.target_object + "_incidence"].to(torch.float32)

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute incidence matrix loss masked by valid constituent and object positions."""
        losses = {}
        output = outputs[self.incidence_key]
        target = targets[self.target_object + "_incidence"].type_as(output)

        node_mask = targets[self.input_constituent + "_valid"].unsqueeze(1).expand_as(output)
        object_mask = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(output)
        mask = node_mask & object_mask
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=mask)

        return losses


class IncidenceBasedRegressionTask(RegressionTask):
    """Regression task that constructs proxy particle features from a predicted incidence matrix."""

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
        scale_dict_path: str,
        net: nn.Module,
        loss: RegressionLossType = "smooth_l1",
        use_incidence: bool = True,
        use_nodes: bool = False,
        has_intermediate_loss: bool = True,
        mode: str = "offset",
        cost: str = "old",
        incidence_task_name: str = "incidence",
        class_prob_task_name: str = "classification",
    ):
        """Construct proxy particles from predicted incidence matrix, and then correct the proxies using a regression.

        Args:
            name: Name of the task.
            input_constituent: Name of the input hit object.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            scale_dict_path: Path to the scale dictionary.
            net: Network for regression.
            loss: Type of loss function to use.
            use_incidence: Whether to use incidence matrix.
            use_nodes: Whether to use node features.
            has_intermediate_loss: Whether the task has intermediate loss.
            mode: Regression mode ('offset' or 'scale').
            cost: Cost mode ('old' or 'new').
            incidence_task_name: Name of the task that produces incidence matrix (default: 'incidence').
            class_prob_task_name: Name of the task that produces class probabilities (default: 'classification').
        """
        super().__init__(
            name=name,
            output_object=output_object,
            target_object=target_object,
            fields=fields,
            loss_weight=loss_weight,
            cost_weight=cost_weight,
            loss=loss,
            has_intermediate_loss=has_intermediate_loss,
        )
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.scaler = FeatureScaler(scale_dict_path=scale_dict_path)
        self.use_incidence = use_incidence
        self.cost_weight = cost_weight
        self.net = net
        self.use_nodes = use_nodes
        self.inputs = [input_object + "_embed"] + [input_constituent + "_" + field for field in fields]
        self.outputs = [output_object + "_regr", output_object + "_proxy_regr"]
        self.mode = mode
        if mode not in {"offset", "scale"}:
            raise ValueError(f"Invalid mode {mode}, must be 'offset' or 'scale'")
        if cost == "old":
            self.cost = self.old_cost
        elif cost == "new":
            self.cost = self.new_cost
        else:
            raise ValueError(f"Invalid cost mode {cost}")

        self.incidence_task_name = incidence_task_name
        self.incidence_output_key = f"{output_object}_incidence"
        self.class_prob_task_name = class_prob_task_name
        self.class_prob_output_key = f"{output_object}_class_prob"

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """Build proxy particle features from the incidence matrix and regress corrections."""
        if self.use_incidence:
            inc = None
            class_probs = None
            if outputs is not None:
                if self.incidence_task_name in outputs:
                    inc = outputs[self.incidence_task_name].get(self.incidence_output_key)
                    if inc is not None:
                        inc = inc.detach()

                if self.class_prob_task_name in outputs:
                    class_probs = outputs[self.class_prob_task_name].get(self.class_prob_output_key)
                    if class_probs is not None:
                        class_probs = class_probs.detach()

            if inc is None or class_probs is None:
                raise RuntimeError(
                    f"When use_incidence=True, both incidence and class_probs must be provided. "
                    f"Got inc={inc is not None}, class_probs={class_probs is not None}. "
                    f"Looking for incidence in task '{self.incidence_task_name}' with key '{self.incidence_output_key}' "
                    f"and class_probs in task '{self.class_prob_task_name}' with key '{self.class_prob_output_key}'"
                )

            proxy_feats, is_charged = self.get_proxy_feats(inc, x["inputs"], class_probs=class_probs)
            input_data = torch.cat(
                [
                    x[self.input_object + "_embed"],
                    proxy_feats,
                    is_charged.unsqueeze(-1),
                ],
                -1,
            )
            if self.use_nodes:
                valid_mask = x[self.input_constituent + "_valid"].unsqueeze(-1)
                masked_embed = valid_mask * x[self.input_constituent + "_embed"]
                node_feats = torch.bmm(inc, masked_embed)
                input_data = torch.cat([input_data, node_feats], dim=-1)
        else:
            input_data = x[self.input_object + "_embed"]
            proxy_feats = torch.zeros_like(input_data[..., : len(self.fields)])
        if self.mode == "offset":
            preds = self.net(input_data) + proxy_feats
        elif self.mode == "scale":
            preds = self.net(input_data) * proxy_feats
        else:
            raise ValueError(f"Invalid mode {self.mode}")
        return {self.output_object + "_regr": preds, self.output_object + "_proxy_regr": proxy_feats}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Return per-field regression predictions and proxy feature values."""
        pflow_regr = outputs[self.output_object + "_regr"]
        proxy_regr = outputs[self.output_object + "_proxy_regr"]
        return {self.output_object + "_" + field: pflow_regr[..., i] for i, field in enumerate(self.fields)} | {
            self.output_object + "_proxy_" + field: proxy_regr[..., i] for i, field in enumerate(self.fields)
        }

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute regression metrics including residuals for proxy particle predictions."""
        metrics = super().metrics(preds, targets)
        for field in self.fields:
            pred = preds[self.output_object + "_proxy_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            abs_err = (pred - target).abs()
            metrics[field + "_proxy_abs_res"] = abs_err.mean()
            metrics[field + "_proxy_abs_norm_res"] = torch.mean(abs_err / target.abs() + 1e-8)
        return metrics

    def old_cost(self, outputs, targets) -> dict[str, Tensor]:
        """Compute dR-based matching cost in eta-phi space."""
        eta_pos = self.fields.index("eta")
        sinphi_pos = self.fields.index("sinphi")
        cosphi_pos = self.fields.index("cosphi")

        pred_phi = torch.atan2(
            outputs[self.output_object + "_regr"][..., sinphi_pos],
            outputs[self.output_object + "_regr"][..., cosphi_pos],
        )[:, :, None]
        pred_eta = outputs[self.output_object + "_regr"][..., eta_pos][:, :, None]
        target_phi = torch.atan2(
            targets[self.target_object + "_sinphi"],
            targets[self.target_object + "_cosphi"],
        )[:, None, :]
        target_eta = targets[self.target_object + "_eta"][:, None, :]
        dphi = (pred_phi - target_phi + torch.pi) % (2 * torch.pi) - torch.pi
        deta = (pred_eta - target_eta) * self.scaler["eta"].scale
        if self.use_pt_match:
            pred_pt = outputs[self.output_object + "_regr"][..., self.pt_pos][:, :, None]
            target_pt = targets[self.target_object + "_pt"][:, None, :]
            pt_cost = (target_pt - pred_pt) ** 2 / (target_pt**2 + 1e-8)
        else:
            pt_cost = 0
        cost = self.cost_weight * torch.sqrt(pt_cost + torch.pow(dphi, 2) + torch.pow(deta, 2))
        return {"regression": cost}

    def new_cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute pairwise regression-based matching cost."""
        output = outputs[self.output_object + "_regr"].detach().to(torch.float32)
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)
        num_objects = output.shape[1]
        num_targets = target.shape[1]

        costs = self.loss_fn(
            output.unsqueeze(2).expand(-1, -1, num_objects, -1),
            target.unsqueeze(1).expand(-1, num_targets, -1, -1),
            reduction="none",
        )

        return {f"regr_{self.loss_fn_name}": self.cost_weight * costs.mean(-1)}

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute regression loss over valid objects only."""
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.output_object + "_regr"]

        mask = targets[self.target_object + "_valid"]
        target = target[mask]
        output = output[mask]

        loss = self.loss_fn(output, target, reduction="mean")
        return {self.loss_fn_name: self.loss_weight * loss}

    def scale_proxy_feats(self, proxy_feats: Tensor):
        """Normalise proxy particle features using the stored feature scaler."""
        return torch.cat([self.scaler[field].transform(proxy_feats[..., i]).unsqueeze(-1) for i, field in enumerate(self.fields)], -1)

    def get_proxy_feats(
        self,
        incidence: Tensor,
        inputs: dict[str, Tensor],
        class_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build proxy particle features for charged and neutral particles from the incidence matrix.

        Args:
            incidence: Soft incidence matrix of shape ``(B, N_objects, N_constituents)``.
            inputs: Raw input feature dictionary.
            class_probs: Class probability tensor used to determine charged/neutral status.

        Returns:
            Tuple of ``(proxy_feats, is_charged)`` where ``proxy_feats`` has shape
            ``(B, N_objects, N_fields)`` and ``is_charged`` is a boolean mask of shape
            ``(B, N_objects)``.
        """
        proxy_feats = torch.cat(
            [inputs[self.input_constituent + "_" + field].unsqueeze(-1) for field in self.fields],
            dim=-1,
        )

        charged_inc = incidence * inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        charged_inc_top2 = (topk_attn(charged_inc, 2, dim=-2) & (charged_inc > 0)).float()
        charged_inc_max = charged_inc.max(-2, keepdim=True)[0]
        charged_inc_new = (charged_inc == charged_inc_max) & (charged_inc > 0)
        zero_track_mask = charged_inc_new.sum(-1, keepdim=True) == 0
        charged_inc = torch.where(zero_track_mask, charged_inc_top2, charged_inc_new)

        is_charged = class_probs.argmax(-1) < 3

        proxy_feats_charged = torch.bmm(charged_inc, proxy_feats)
        proxy_feats_charged[..., 0] = proxy_feats_charged[..., 1] * torch.cosh(proxy_feats_charged[..., 2])
        proxy_feats_charged = self.scale_proxy_feats(proxy_feats_charged) * is_charged.unsqueeze(-1)

        inc_e_weighted = incidence * proxy_feats[..., 0].unsqueeze(1)
        inc_e_weighted *= 1 - inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        inc = inc_e_weighted / (inc_e_weighted.sum(dim=-1, keepdim=True) + 1e-6)

        proxy_feats_neutral = torch.einsum("bnf,bpn->bpf", proxy_feats, inc)
        proxy_feats_neutral[..., 0] = inc_e_weighted.sum(-1)
        proxy_feats_neutral[..., 1] = proxy_feats_neutral[..., 0] / torch.cosh(proxy_feats_neutral[..., 2])

        proxy_feats_neutral = self.scale_proxy_feats(proxy_feats_neutral) * (~is_charged).unsqueeze(-1)
        proxy_feats = proxy_feats_charged + proxy_feats_neutral

        return proxy_feats, is_charged
