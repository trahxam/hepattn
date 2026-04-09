import math

import torch
from torch import Tensor

from hepattn.components.dense import Dense
from hepattn.tasks.base import REGRESSION_LOSS_FNS, RegressionLossType, Task


class RegressionTask(Task):
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
        latent = self.latent(x)
        return {self.regression_key: latent}

    def predict(self, outputs: dict[str, Tensor], query_mask: Tensor | None = None) -> dict[str, Tensor]:
        latent = outputs[self.regression_key]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.regression_key]

        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        loss = self.loss_fn(output, target, reduction="none")
        loss = torch.mean(loss, dim=-1)

        return {self.loss_fn_name: self.loss_weight * loss.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = {}
        for field in self.fields:
            pred = preds[self.output_object + "_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            abs_err = (pred - target).abs()
            metrics[field + "_abs_res"] = torch.mean(abs_err)
            metrics[field + "_abs_norm_res"] = torch.mean(abs_err / target.abs() + 1e-8)
        return metrics


class GaussianRegressionTask(Task):
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
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)

        z = torch.einsum("...ij,...j->...i", outputs[self.output_object + "_ubar"], y - outputs[self.output_object + "_mu"])
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(torch.diagonal(outputs[self.output_object + "_u"], offset=0, dim1=-2, dim2=-1), dim=-1)
        log_likelihood = self.likelihood_norm - 0.5 * zsq + jac

        log_likelihood *= targets[self.target_object + "_valid"].type_as(log_likelihood)
        return {"nll": -self.loss_weight * log_likelihood.mean()}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
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


class ObjectGaussianRegressionTask(GaussianRegressionTask):
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
    ):
        """Gaussian regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [
            output_object + "_mu",
            output_object + "_ubar",
            output_object + "_u",
        ]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        mu = outputs[self.output_object + "_mu"].to(torch.float32)
        ubar = outputs[self.output_object + "_ubar"].to(torch.float32)
        u = outputs[self.output_object + "_u"].to(torch.float32)
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)

        num_objects = y.shape[1]
        mu = mu.unsqueeze(2).expand(-1, -1, num_objects, -1)
        ubar = ubar.unsqueeze(2).expand(-1, -1, num_objects, -1, -1)
        u = u.unsqueeze(2).expand(-1, -1, num_objects, -1, -1)
        diagu = torch.diagonal(u, offset=0, dim1=-2, dim2=-1)
        y = y.unsqueeze(1).expand(-1, num_objects, -1, -1)

        z = torch.einsum("...ij,...j->...i", ubar, y - mu)
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(diagu, dim=-1)

        log_likelihood = self.likelihood_norm - 0.5 * zsq + jac
        log_likelihood *= targets[f"{self.target_object}_valid"].unsqueeze(1).type_as(log_likelihood)
        costs = -log_likelihood

        return {"nll": self.cost_weight * costs}


class ObjectRegressionTask(RegressionTask):
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
        return self.net(x[self.input_object + "_embed"])


class ObjectHitRegressionTask(RegressionTask):
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
            input_constituent: Name of the input constituent type (e.g. hits in tracking).
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
        x_obj = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_constituent + "_embed"])

        x_obj = x_obj.reshape(x_obj.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))

        x_obj_hit = torch.einsum("...nie,...mie->...nmi", x_obj, x_hit)

        x_obj_hit *= x[self.input_constituent + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
        return x_obj_hit
