from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns


class Task(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the forward pass of the task."""

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return predictions from model outputs."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute loss between outputs and targets."""

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs, **kwargs):
        return {}


class ObjectValidTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
    ):
        """Task used for classifying whether object candidates / seeds should be
        taken as reconstructed / pred objects or not.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_object : str
            Name of the input object feature
        output_object : str
            Name of the output object feature which will denote if the predicted object slot is used or not.
        target_object: str
            Name of the target object feature that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        dim : int
            Embedding dimension of the input features.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight

        # Internal
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_logit"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Network projects the embedding down into a scalar
        x_logit = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_logit": x_logit.squeeze(-1)}

    def predict(self, outputs, threshold=0.5):
        # Objects that have a predicted probability aove the threshold are marked as predicted to exist
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_logit"].detach()
        target = targets[self.target_object + "_valid"].type_as(output)
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        losses = {}
        output = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        weight = target + self.null_weight * (1 - target)
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=weight)
        return losses


class HitFilterTask(Task):
    def __init__(
        self,
        name: str,
        hit_name: str,
        target_field: str,
        dim: int,
        threshold: float = 0.1,
    ):
        """Task used for classifying whether hits belong to reconstructable objects or not."""
        super().__init__()

        self.name = name
        self.hit_name = hit_name
        self.target_field = target_field
        self.dim = dim
        self.threshold = threshold

        # Internal
        self.input_objects = [f"{hit_name}_embed"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_logit = self.net(x[f"{self.hit_name}_embed"])
        return {f"{self.hit_name}_logit": x_logit.squeeze(-1)}

    def predict(self, outputs: dict) -> dict:
        return {f"{self.hit_name}_{self.target_field}": outputs[f"{self.hit_name}_logit"].detach().sigmoid() >= self.threshold}

    def loss(self, outputs: dict, targets: dict) -> dict:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        output = outputs[f"{self.hit_name}_logit"]
        target = targets[f"{self.hit_name}_{self.target_field}"].type_as(output)

        # Calculate the BCE loss with class weighting
        weight = 1 / target.float().mean()
        loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
        return {f"{self.hit_name}_bce": loss}


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
    ):
        super().__init__()

        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight

        self.output_object_hit = output_object + "_" + input_hit
        self.target_object_hit = target_object + "_" + input_hit
        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        self.hit_net = Dense(dim, dim)
        self.object_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the hits and objects
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = x[self.input_hit + "_embed"]

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = torch.einsum("bnc,bmc->bnm", x_object, x_hit)

        # Zero out entries for any hit slots that are not valid
        object_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs, threshold=0.1):
        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() > threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"].detach()
        target = targets[self.target_object_hit + "_valid"].type_as(output)

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)

            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_valid"].type_as(output)

        # Build a padding mask for object-hit pairs
        hit_pad = targets[self.input_hit + "_valid"].unsqueeze(-2).expand_as(target)
        object_pad = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(target)
        # An object-hit is valid slot if both its object and hit are valid slots
        # TODO: Maybe calling this a mask is confusing since true entries are
        object_hit_mask = object_pad & hit_pad

        weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            loss = loss_fns[loss_fn](output, target, mask=object_hit_mask, weight=weight)
            losses[loss_fn] = loss_weight * loss
        return losses


class RegressionTask(Task):
    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
    ):
        super().__init__()

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.k = len(fields)
        # For standard regression number of DoFs is just the number of targets
        self.ndofs = self.k

    def forward(self, x: dict[str, Tensor], pads: None | dict[str, Tensor] = None) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x, pads=pads)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs):
        # Split the regression vectior into the separate fields
        latent = outputs[self.output_object + "_regr"]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs, data):
        targets = torch.stack([data[self.target_object + "_" + field] for field in self.fields], dim=-1)
        loss = torch.nn.functional.smooth_l1_loss(outputs[self.output_object + "_regr"], targets, reduction="none")

        # Compute average loss over all the features
        loss = torch.mean(loss, dim=-1)
        # Compute the regression loss only for valid objects
        loss = loss[data[self.target_object + "_valid"]]
        # Take the average loss and apply the task weight
        return {"smooth_l1": self.loss_weight * loss.mean()}

    def metrics(self, preds, data):
        metrics = {}
        for field in self.fields:
            # Get the error between the prediction and target for this field
            err = preds[self.output_object + "_" + field] - data[self.target_object + "_" + field]
            # Select the error only for valid objects
            err = err[data[self.target_object + "_valid"]]
            # Compute the RMSE and log it
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(err)))

        return metrics


class GaussianRegressionTask(Task):
    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
    ):
        super().__init__()

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.k = len(fields)
        # For multivaraite gaussian case we have extra DoFs from the variance and covariance terms
        self.ndofs = self.k + int(self.k * (self.k + 1) / 2)

    def forward(self, x: dict[str, Tensor], pads: None | dict[str, Tensor] = None) -> dict[str, Tensor]:
        latent = self.latent(x, pads=pads)
        k = self.k
        triu_idx = torch.triu_indices(k, k, device=latent.device)

        # Mean vector
        mu = latent[..., :k]
        # Upper-diagonal Cholesky decomposition of the precision matrix
        U = torch.zeros(latent.size()[:-1] + torch.Size((k, k)), device=latent.device)
        U[..., triu_idx[0, :], triu_idx[1, :]] = latent[..., k:]

        Ubar = U.clone()
        # Make sure the diagonal entries are positive (as variance is always positive)
        Ubar[..., torch.arange(k), torch.arange(k)] = torch.exp(U[..., torch.arange(k), torch.arange(k)])

        return {self.output_object + "_mu": mu, self.output_object + "_U": U, self.output_object + "_Ubar": Ubar}

    def predict(self, outputs):
        preds = outputs
        Ubar = outputs[self.target_object + "_Ubar"]

        # Calculate the precision matrix
        precs = torch.einsum("...kj,...kl->...jl", Ubar, Ubar)

        # Get the predicted mean for each field
        for i, field in enumerate(self.fields):
            preds[self.output_object + "_" + field] = mu[..., i]

        # Get the predicted precision for each field and the predicted covariance / coprecision
        for i, field_i in enumerate(self.field):
            for j, field_j in enumerate(self.fields):
                if i > j:
                    continue
                preds[field_i + "_" + field_j + "_prec"] = precs[..., i, j]

        return preds

    def loss(self, outputs, data):
        targets = torch.stack([data[self.target_object + "_" + field] for field in self.fields], dim=-1)

        # Compute the standardised score vector between the targets and the predicted distribution paramaters
        z = torch.einsum("...ij,...j->...i", outputs[self.output_object + "_Ubar"], targets - outputs[self.output_object + "_mu"])
        # Compute the NLL from the score vector
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(torch.diagonal(outputs[self.output_object + "_U"], offset=0, dim1=-2, dim2=-1), dim=-1)
        nll = -0.5 * zsq + jac

        # Only compute NLL for valid objects or object-hit pairs
        nll = nll[data[self.target_object + "_valid"]]
        # Take the average and apply the task weight
        nll = -self.loss_weight * nll.mean()

        return {"nll": nll}

    def metrics(self, preds, data):
        targets = torch.stack([data[self.target_object + "_" + field] for field in self.fields], dim=-1)
        errors = targets - preds[self.name + "_mu"]

        # Project back to the standard normal
        z = torch.einsum("...ij,...j->...i", preds[self.name + "_Ubar"], errors)

        # Select only values that havea valid target
        valid_mask = data[self.target_object + "_valid"]

        metrics = {}
        for i, field in enumerate(self.fields):
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(errors[..., i][valid_mask])))
            # The mean and standard deviation of the pulls to check predictions are calibrated
            metrics[field + "_pull_mean"] = torch.mean(z[..., i][valid_mask])
            metrics[field + "_pull_std"] = torch.std(z[..., i][valid_mask])

        return metrics


class ObjectRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        embed_dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight)

        self.input_object = input_object

        self.input_objects = [input_object + "_embed"]
        self.output_objects = [output_object + "_regr"]
        self.embed_dim = embed_dim
        self.net = Dense(self.embed_dim, self.ndofs)

    def latent(self, x: dict[str, Tensor], pads: None | dict[str, Tensor] = None) -> Tensor:
        return self.net(x[self.input_object + "_embed"])


class ObjectHitRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_object: str,
        input_hit: str,
        output_object_hit: str,
        target_object_hit: str,
        fields: list[str],
        loss_weight: float,
        embed_dim: int,
    ):
        super().__init__(name, output_object_hit, target_object_hit, fields, loss_weight)

        self.input_object = input_object
        self.input_hit = input_hit

        self.input_objects = [input_object + "_embed", input_hit + "_embed"]
        self.output_objects = [output_object_hit + "_regr"]

        # The embedding dimension is split up equally between each degree of freedom
        self.embed_dim = embed_dim
        self.embed_dim_per_dof = self.embed_dim // self.ndofs
        self.object_net = Dense(self.embed_dim, self.embed_dim_per_dof * self.ndofs)
        self.hit_net = Dense(self.embed_dim, self.embed_dim_per_dof * self.ndofs)

    def latent(self, x: dict[str, Tensor], pads: None | dict[str, Tensor] = None) -> Tensor:
        # Embed the hits and objects and reshape so we have a separate embedding for each DoF
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        x_object = x_object.reshape(x_object.size()[:-1] + torch.Size((self.ndofs, self.embed_dim_per_dof)))  # Shape BNDE
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.embed_dim_per_dof)))  # Shape BMDE

        # Take the dot product between the hits and objects over the last embedding dimension so we are left
        # with just a scalar for each degree of freedom
        x_object_hit = torch.einsum("...nie,...mie->...nmi", x_object, x_hit)  # Shape BNMD

        # If padding data is provided, use it to zero out predictions for any hit slots that are not valid
        if pads is not None:
            # Shape of padding goes BM -> B1M -> B1M1 -> BNMD
            pad = pads[self.input_hit + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_object_hit).as_type(x_object_hit)
            x_object_hit = x_object_hit * pad
        return x_object_hit
