from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns
from hepattn.models.task import Task


class PixelTrackTask(Task):
    """Pixel-specific query task with built-in validity classification and optional regression."""

    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        classification_losses: dict[str, float],
        classification_costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        enable_regression: bool = False,
        regression_fields: list[str] | None = None,
        regression_loss_weight: float = 1.0,
        regression_cost_weight: float = 1.0,
        classification_cost_weight: float = 5.0,
        mask_queries: bool = False,
        has_intermediate_loss: bool = True,
        has_first_layer_loss: bool = False,
    ):
        if has_first_layer_loss and not has_intermediate_loss:
            raise ValueError("has_first_layer_loss=True requires has_intermediate_loss=True")

        if enable_regression and not regression_fields:
            raise ValueError("regression_fields must be provided when enable_regression=True")

        super().__init__(has_intermediate_loss=has_intermediate_loss, has_first_layer_loss=has_first_layer_loss)

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.classification_losses = classification_losses
        self.classification_costs = classification_costs
        self.enable_regression = enable_regression
        self.mask_queries = mask_queries
        self.regression_fields = list(regression_fields or [])
        self.regression_loss_weight = regression_loss_weight
        self.regression_cost_weight = regression_cost_weight
        self.classification_cost_weight = classification_cost_weight

        self.logits_key = f"{output_object}_logit"
        self.probs_key = f"{output_object}_class_prob"
        self.mu_key = f"{output_object}_mu"
        self.u_key = f"{output_object}_u"
        self.ubar_key = f"{output_object}_ubar"

        self.classification_net = Dense(input_size=dim, output_size=1)

        loss_weights = torch.ones(2, dtype=torch.float32)
        loss_weights[-1] = null_weight
        self.register_buffer("loss_weights", loss_weights)

        self.inputs = [f"{input_object}_embed"]
        self.outputs = [self.logits_key, self.probs_key]

        self.k = len(self.regression_fields)
        self.ndofs = self.k + int(self.k * (self.k + 1) / 2)
        self.likelihood_norm = self.k * 0.5 * math.log(2 * math.pi)
        self.regression_net = Dense(dim, self.ndofs) if enable_regression else None
        if self.regression_net is not None:
            self.outputs.extend([self.mu_key, self.ubar_key, self.u_key])

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        _ = outputs
        embed = x[f"{self.input_object}_embed"]

        logits = self.classification_net(embed).squeeze(-1)
        valid_prob = torch.sigmoid(logits)
        task_outputs = {
            self.logits_key: logits,
            self.probs_key: torch.stack([valid_prob, 1 - valid_prob], dim=-1),
        }

        if self.regression_net is None:
            return task_outputs

        latent = self.regression_net(embed)
        triu_idx = torch.triu_indices(self.k, self.k, device=latent.device)

        mu = latent[..., : self.k]
        u = torch.zeros(latent.size()[:-1] + torch.Size((self.k, self.k)), device=latent.device)
        u[..., triu_idx[0], triu_idx[1]] = latent[..., self.k :]

        ubar = u.clone()
        diag_idx = torch.arange(self.k, device=latent.device)
        ubar[..., diag_idx, diag_idx] = torch.exp(u[..., diag_idx, diag_idx])

        task_outputs.update(
            {
                self.mu_key: mu,
                self.u_key: u,
                self.ubar_key: ubar,
            }
        )
        return task_outputs

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        class_probs = outputs[self.probs_key].detach()
        classes = class_probs.argmax(-1)
        valid = classes == 0

        if query_mask is not None:
            valid = valid & query_mask

        preds = {
            f"{self.output_object}_class": classes,
            f"{self.output_object}_valid_prob": 1 - class_probs[..., -1],
            f"{self.output_object}_valid": valid,
        }

        if self.regression_net is None:
            return preds

        mu = outputs[self.mu_key]
        u = outputs[self.u_key]
        ubar = outputs[self.ubar_key]
        precs = torch.einsum("...kj,...kl->...jl", ubar, ubar)

        preds.update(
            {
                self.mu_key: mu,
                self.ubar_key: ubar,
                self.u_key: u,
            }
        )
        for i, field in enumerate(self.regression_fields):
            preds[f"{self.output_object}_{field}"] = mu[..., i]

        for i, field_i in enumerate(self.regression_fields):
            for j, field_j in enumerate(self.regression_fields):
                if i > j:
                    continue
                preds[f"{field_i}_{field_j}_prec"] = precs[..., i, j]

        return preds

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        _ = layer_outputs
        target_valid = targets[f"{self.target_object}_valid"].float()

        query_mask = targets.get("query_mask")

        losses = {}

        if self.regression_net is None:
            sample_weight = target_valid + self.loss_weights[-1] * (1 - target_valid)
            if query_mask is not None:
                sample_weight = sample_weight * query_mask.float()
            for loss_fn, loss_weight in self.classification_losses.items():
                losses[loss_fn] = loss_weight * loss_fns[loss_fn](outputs[self.logits_key], target_valid, sample_weight=sample_weight)
            return losses

        # Joint score loss: couples classification and regression without a null distribution.
        # Valid targets:  -(log p_valid + log N(y; mu, Sigma))  — blows up if p_valid -> 0, preventing collapse.
        # Null targets:   -log p_null                           — blows up if p_valid -> 1, preventing collapse.
        log_p_valid = F.logsigmoid(outputs[self.logits_key])
        log_p_null = F.logsigmoid(-outputs[self.logits_key])

        y = torch.stack([targets[f"{self.target_object}_{field}"] for field in self.regression_fields], dim=-1)

        # log N(y; mu, Sigma) using the Cholesky precision parameterisation
        z = torch.einsum("...ij,...j->...i", outputs[self.ubar_key], y - outputs[self.mu_key])
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(torch.diagonal(outputs[self.u_key], offset=0, dim1=-2, dim2=-1), dim=-1)
        log_gaussian = -self.likelihood_norm - 0.5 * zsq + jac

        valid_mask = target_valid.bool()
        loss_per_query = torch.where(valid_mask, -(log_p_valid + log_gaussian), self.loss_weights[-1] * (-log_p_null))

        if query_mask is not None:
            loss_per_query = loss_per_query * query_mask.float()

        losses["joint_nll"] = self.regression_loss_weight * loss_per_query.mean()
        return losses

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        _ = kwargs

        if self.regression_net is None:
            costs = {}
            for cost_fn, cost_weight in self.classification_costs.items():
                costs[cost_fn] = cost_weight * cost_fns[cost_fn](
                    outputs[self.logits_key].detach().to(torch.float32),
                    targets[f"{self.target_object}_valid"].to(torch.float32),
                )
            return costs

        logits = outputs[self.logits_key].detach().to(torch.float32)
        mu = outputs[self.mu_key].detach().to(torch.float32)
        ubar = outputs[self.ubar_key].detach().to(torch.float32)
        u = outputs[self.u_key].detach().to(torch.float32)
        y = torch.stack([targets[f"{self.target_object}_{field}"] for field in self.regression_fields], dim=-1).to(torch.float32)

        num_queries = mu.shape[1]
        num_targets = y.shape[1]

        # Classification cost: gates matching so null-predicting queries can't win on regression alone.
        # Weighted ~4-5x higher than regression so a 1-sigma NLL difference can't override a clear validity signal.
        log_p_valid = F.logsigmoid(logits).unsqueeze(2).expand(-1, -1, num_targets)
        class_cost = -self.classification_cost_weight * log_p_valid

        mu_exp = mu.unsqueeze(2).expand(-1, -1, num_targets, -1)
        ubar_exp = ubar.unsqueeze(2).expand(-1, -1, num_targets, -1, -1)
        u_exp = u.unsqueeze(2).expand(-1, -1, num_targets, -1, -1)
        y_exp = y.unsqueeze(1).expand(-1, num_queries, -1, -1)

        z = torch.einsum("...ij,...j->...i", ubar_exp, y_exp - mu_exp)
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(torch.diagonal(u_exp, offset=0, dim1=-2, dim2=-1), dim=-1)
        log_gaussian = -self.likelihood_norm - 0.5 * zsq + jac
        regr_cost = -self.regression_cost_weight * log_gaussian

        return {"class": class_cost, "nll": regr_cost}

    def query_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> Tensor | None:
        if not self.mask_queries:
            return None

        class_probs = outputs[self.probs_key].detach()
        return class_probs[..., -1] <= (1 - threshold)

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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
        metrics = {
            "num_queries": float(pred_valid.shape[1]) if query_mask is None else float(query_mask.sum()),
            "query_frac_pred_valid": pred_valid.float().mean() if query_mask is None else pred_valid[query_mask].float().mean(),
            "query_eff": tp / torch.maximum(true_pos, eps),
            "query_fr": fp / torch.maximum(total_pred, eps),
        }

        if self.regression_net is None:
            return metrics

        y = torch.stack([targets[f"{self.target_object}_{field}"] for field in self.regression_fields], dim=-1)
        res = y - preds[self.mu_key]
        z = torch.einsum("...ij,...j->...i", preds[self.ubar_key], res)
        valid_mask = targets[f"{self.target_object}_valid"]

        for i, field in enumerate(self.regression_fields):
            metrics[f"{field}_rmse"] = torch.sqrt(torch.mean(torch.square(res[..., i][valid_mask])))
            metrics[f"{field}_pull_mean"] = torch.mean(z[..., i][valid_mask])
            metrics[f"{field}_pull_std"] = torch.std(z[..., i][valid_mask])

        return metrics
