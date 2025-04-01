import torch
import numpy as np

from typing import List, Dict
from torch import Tensor, nn

from hepattn.models.loss import loss_fns, cost_fns
from hepattn.models.dense import Dense
    

class TrackValidTask(nn.Module):
    def __init__(
        self,
        name: str,
        input_track: str,
        output_track: str,
        target_track: str,
        losses: dict[str, float],
        costs: dict[str, float],
        embed_dim: int,
        null_weight: float = 1.0,
    ):
        """ Task used for classifying whether track candidates / seeds should be
        taken as reconstructed / pred tracks or not.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_track : str
            Name of the input track feature 
        output_track : str
            Name of the output track feature which will denote if the predicted track slot is used or not.
        target_track: str
            Name of the target track feature that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        embed_dim : int
            Embedding dimension of the input features.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

        self.name = name
        self.input_track = input_track
        self.output_track = output_track
        self.target_track = target_track
        self.losses = losses
        self.costs = costs
        self.embed_dim = embed_dim
        self.null_weight = null_weight

        # Internal
        self.input_features = [input_track + "_embed"]
        self.output_features = [output_track + "_logit"]
        self.net = Dense(embed_dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """ Yields a probability denoting whether the model thinks whether a track
        slot should be used or not. 

        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing the embeding that is used for the task.
        pads : dict[str, Tensor]
            Optional dictionary containing a padding tensor. Not used for the track valid task.

        Returns
        -------
        outputs : dict[str, Tensor]
            Dictionary containing the output track probabilites.
        """
        # Network projects the embedding down into a scalar
        x_track = self.net(x[self.input_track + "_embed"])
        return {self.output_track + "_logit": x_track.squeeze(-1),}

    def predict(self, outputs, threshold=0.5):
        """ Performs a cut on the output probability to predict whether the output
        track slot should be used or not.

        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing the outputs from the forward pass of the task.
        threshold : float
            Float indicating the threshold value above which output probabilies correspond to a
            psoitive predicton / a track slot is marked as predicted to be used.

        Returns
        -------
        outputs : dict[str, Tensor]
            Dictionary containing the output track predictions of whether a track slot is used or not.
        """
        # Tracks that have a predicted probability aove the threshold are marked as predicted to exist
        return {self.output_track + "_valid": outputs[self.output_track + "_logit"].sigmoid() >= threshold}

    def cost(self, outputs, targets):
        """ Produces a dict of cost matrices which consist of the loss between each possible
        predicted and true track pair.

        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing the outputs from the forward pass of the task.
        data : float
            Data containing grund truth / target tracks.

        Returns
        -------
        costs : dict[str, Tensor]
            A dictionary of cost tensors. Each cost tensor is of shape (batch, num pred tracks, num true tracks),
            where each entry [i,j,k] denotes the loss between the jth pred track and kth true track for the ith sample.
        """
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](outputs[self.output_track + "_logit"], targets[self.target_track + "_valid"].float())
            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_track + "_valid"].unsqueeze(1).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        """ Calculates the loss between a set of predicted and true tracks.

        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing the outputs from the forward pass of the task.
        data : float
            Data containing grund truth / target tracks.

        Returns
        -------
        loss : dict[str, Tensor]
            A dictionary of losses. Each loss is the scalar average loss over the batch from a given loss function.
        """
        losses = {}
        target = targets[self.target_track + "_valid"].float()
        weight = target + self.null_weight * (1 - target)
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](outputs[self.output_track + "_logit"], target, mask=None, weight=weight)
        return losses


class TrackHitValidTask(nn.Module):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_track: str,
        output_track: str,
        target_track: str,
        losses: dict[str, float],
        costs: dict[str, float],
        embed_dim: int,
        null_weight: float = 1.0,
    ):
        """ Takes a set of input hits and input tracks and classifies whether each
        hit and track pair is assocoiated to eachother or not. To do this it embeds
        each hit and track separately, and then takes the sigmoid of the dot product
        between each track-hit pair to produce a probability.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_track : str
            Name of the input track feature.
        input_hit : str
            Name of the input hit featrue.
        output_track_hit : str
            Name of the output track-hit feature pair that we want to predict asisgnment on.
        target_track: str
            Name of the target track-hit feature pair that we want to predict asisgnment on.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        embed_dim : int
            Embedding dimension of the input features.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

        self.name = name
        self.input_hit = input_hit
        self.input_track = input_track
        self.output_track = output_track
        self.target_track = target_track
        self.losses = losses
        self.costs = costs
        self.embed_dim = embed_dim
        self.null_weight = null_weight

        self.output_track_hit = output_track + "_" + input_hit
        self.target_track_hit = target_track + "_" + input_hit
        self.input_features = [input_track + "_embed", input_hit + "_embed"]
        self.output_features = [self.output_track_hit + "_logit"]
        self.hit_net = Dense(embed_dim, embed_dim)
        self.track_net = Dense(embed_dim, embed_dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the hits and tracks
        x_track = self.track_net(x[self.input_track + "_embed"])
        x_hit = x[self.input_hit + "_embed"]

        # Track-hit probability is the dot product between the hit and track embedding
        track_hit_logit = torch.einsum("bnc,bmc->bnm", x_track, x_hit)

        # Zero out entries for any hit slots that are not valid
        track_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(track_hit_logit)] = torch.finfo(track_hit_logit.dtype).min

        return {self.output_track_hit + "_logit": track_hit_logit}
    
    def attn_mask(self, outputs, threshold=0.1):
        attn_mask = (outputs[self.output_track_hit + "_logit"].detach().sigmoid() < threshold)

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        """ Performs a cut on the output probability to predict whether each of the input tracks should be 
        assigned to each of the input hits.

        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing the outputs from the forward pass of the task.
        threshold : float
            Float indicating the threshold value above which output probabilies 
            should imply a trach-hit assignment.

        Returns
        -------
        outputs : dict[str, Tensor]
            Dictionary containing the output track predictions of whether each
            track is assigned to each hit or not.
        """
        # Track-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_track_hit + "_valid": outputs[self.output_track_hit + "_logit"].sigmoid() >= threshold}

    def cost(self, outputs, targets):
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](outputs[self.output_track_hit + "_logit"], targets[self.target_track_hit + "_valid"].float())

            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_track + "_valid"].unsqueeze(1).expand_as(costs[cost_fn])] = 1e6
        return costs
        
    def loss(self, outputs, targets):
        target = targets[self.target_track_hit + "_valid"].float()
        # Build a padding mask for track-hit pairs
        hit_pad = targets[self.input_hit + "_valid"].unsqueeze(-2).expand_as(target)
        track_pad = targets[self.target_track + "_valid"].unsqueeze(-1).expand_as(target)
        # A track-hit is valid slot if both its track and hit are valid slots
        # TODO: Maybe calling this a mask is confusing since true entries are 
        track_hit_mask = track_pad & hit_pad
        
        
        weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            loss = loss_fns[loss_fn](outputs[self.output_track_hit + "_logit"], target, mask=None, weight=weight)
            losses[loss_fn] = loss_weight * loss
        return losses
