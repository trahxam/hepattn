import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.task import Task
from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns, mask_focal_loss


class TopEventTask(Task):
    def __init__(
        self,
        name: str,
        dim: int,
        has_intermediate_loss: bool = True,
        mask_attn: bool = False,
    ):
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.dim = dim
        self.mask_attn = mask_attn

        self.mask_net = Dense(dim, dim)

        self.inputs = ["jet_embed"]
        self.outputs = ["top_jet_logit"]

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        query_embed = x["query_embed"]
        
        # Produce mask tokens
        top_mask_embed = self.mask_net(query_embed)
        jet_mask_embed = x["jet_embed"]

        # Object-hit probability is the dot product between the hit and object embedding
        top_jet_logit = torch.einsum("bnc,bmc->bnm", top_mask_embed, jet_mask_embed)

        # Zero out entries for any padded input constituents
        if (valid_mask := x["jet_valid"]) is not None:
            valid_mask = valid_mask.unsqueeze(-2).expand_as(top_jet_logit)
            top_jet_logit[~valid_mask] = torch.finfo(top_jet_logit.dtype).min

        return {"top_jet_logit": top_jet_logit}


    def attn_mask(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.mask_attn:
            return {}

        thresh = 0.5
        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= thresh

        return {"jet": attn_mask}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        output = {}

        probs = outputs["top_jet_logit"].sigmoid().detach()

        output["top_jet_prob"] = probs
        output["top_jet_valid"] = probs >= 0.5

        return output

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs["top_jet_logit"].detach().to(torch.float32)
        target = targets["top_jet_valid"].detach().to(output.dtype)

        input_pad_mask = targets["jet_valid"]

        costs = {}
        costs["mask_bce"] = cost_fns["mask_bce"](output, target, input_pad_mask=input_pad_mask)
        costs["mask_focal"] = cost_fns["mask_focal"](output, target, input_pad_mask=input_pad_mask)
        costs["mask_dice"] = cost_fns["mask_dice"](output, target, input_pad_mask=input_pad_mask)

        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs["top_jet_logit"]
        target = targets["top_jet_valid"].to(output.dtype)

        input_pad_mask = targets["jet_valid"]
        object_pad = targets["top_valid"]

        losses = {}
        losses["mask_bce"] = loss_fns["mask_bce"](output, target, object_valid_mask=object_pad, input_pad_mask=input_pad_mask)
        losses["mask_focal"] = loss_fns["mask_focal"](output, target, object_valid_mask=object_pad, input_pad_mask=input_pad_mask)
        losses["mask_dice"] = loss_fns["mask_dice"](output, target, object_valid_mask=object_pad, input_pad_mask=input_pad_mask)

        return losses

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = {}

        mask = torch.full_like(targets["top_valid"].any(-1), True)

        top_valid = targets["top_valid"]

        true_jet_valid = targets["top_jet_valid"]
        pred_jet_valid = preds["top_jet_valid"]

        both_jet_valid = true_jet_valid == pred_jet_valid 

        top_reconstructed = both_jet_valid.all(-1)

        metrics["top_eff_all"] = top_reconstructed.float().sum() / top_valid.float().sum()

        return metrics