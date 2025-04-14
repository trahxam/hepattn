import torch
from torch import Tensor, nn
from typing import Dict
from abc import ABC, abstractmethod

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns


class Task(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute the forward pass of the task."""
        pass

    @abstractmethod
    def predict(self, outputs: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        """Return predictions from model outputs."""
        pass

    @abstractmethod
    def loss(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Compute loss between outputs and targets."""
        pass

    def cost(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
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
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].sigmoid() >= threshold}

    def cost(self, outputs, targets):
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](outputs[self.output_object + "_logit"], targets[self.target_object + "_valid"].float())
            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        losses = {}
        target = targets[self.target_object + "_valid"].float()
        weight = target + self.null_weight * (1 - target)
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](outputs[self.output_object + "_logit"], target, mask=None, weight=weight)
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
        return {f"{self.hit_name}_{self.target_field}": outputs[f"{self.hit_name}_logit"].sigmoid() >= self.threshold}

    def loss(self, outputs: dict, targets: dict) -> dict:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        target = targets[f"{self.hit_name}_{self.target_field}"]
        output = outputs[f"{self.hit_name}_logit"]

        # Calculate the BCE loss with class weighting
        weight = 1 / target.float().mean()
        loss = nn.functional.binary_cross_entropy_with_logits(output, target.type_as(output), pos_weight=weight)
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
        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() < threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].sigmoid() >= threshold}

    def cost(self, outputs, targets):
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](
                outputs[self.output_object_hit + "_logit"], targets[self.target_object_hit + "_valid"].float()
            )

            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        target = targets[self.target_object_hit + "_valid"].float()
        # Build a padding mask for object-hit pairs
        hit_pad = targets[self.input_hit + "_valid"].unsqueeze(-2).expand_as(target)
        object_pad = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(target)
        # An object-hit is valid slot if both its object and hit are valid slots
        # TODO: Maybe calling this a mask is confusing since true entries are
        object_hit_mask = object_pad & hit_pad

        weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            loss = loss_fns[loss_fn](outputs[self.output_object_hit + "_logit"], target, mask=object_hit_mask, weight=weight)
            losses[loss_fn] = loss_weight * loss
        return losses
