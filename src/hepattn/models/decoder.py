"""
Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from hepattn.models import CrossAttention, Dense, SelfAttention
from hepattn.models.loss import MFLoss


class MaskDecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, mask_attention: bool, bidirectional_ca: bool) -> None:
        super().__init__()
        self.step = 0

        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca

        self.q_ca = CrossAttention(dim=dim, n_heads=n_heads)
        self.q_sa = SelfAttention(dim=dim, n_heads=n_heads)
        self.q_dense = Dense(dim)
        if bidirectional_ca:
            self.kv_ca = CrossAttention(dim=dim, n_heads=n_heads)
            self.kv_dense = Dense(dim)

    def forward(self, q: Tensor, kv: Tensor, mask_net: nn.Module, kv_mask: Tensor | None = None) -> Tensor:
        # q are object queries, kv are hit embeddings
        attn_mask = None

        # if we want to do mask attention
        if self.mask_attention:
            scores = get_masks(kv, q, mask_net, kv_mask).detach()

            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = scores.sigmoid() < 0.1

            # if the attn mask is completely invalid for a given query, allow it to attend everywhere
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        # update queries with cross attention from nodes
        q += self.q_ca(q, kv, kv_mask=kv_mask, attn_mask=attn_mask)

        # update queries with self attention
        q += self.q_sa(q)

        # dense update
        q += self.q_dense(q)

        # update nodes with cross attention from queries and dense layer
        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(1, 2)
            kv += self.kv_ca(kv, q, q_mask=kv_mask, attn_mask=attn_mask)
            kv += self.kv_dense(kv)

        return q, kv, attn_mask.transpose(1, 2).detach().cpu().numpy() if attn_mask is not None else None


class MaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        md_config: Mapping,
        mask_net: nn.Module,
        num_objects: int,
        class_net: nn.Module | None = None,
        aux_loss: bool = False,
    ):
        super().__init__()
        self.aux_loss = aux_loss

        self.inital_q = nn.Parameter(torch.empty((num_objects, embed_dim)))
        nn.init.normal_(self.inital_q)

        self.layers = nn.ModuleList([MaskDecoderLayer(embed_dim, **md_config) for _ in range(num_layers)])

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.class_net = class_net
        self.mask_net = mask_net
        self.step = 0

    def get_preds(self, queries: Tensor, mask_tokens: Tensor, input_pad_mask: Tensor | None = None):
        # get mask predictions from queries and mask tokens
        pred_masks = get_masks(mask_tokens, queries, self.mask_net, input_pad_mask)  # [..., mask.squeeze(0)] when padding is enabled
        # get class predictions from queries

        if self.class_net is None:
            return {"masks": pred_masks}

        class_logits = self.class_net(queries)
        if class_logits.shape[-1] == 1:
            class_probs = class_logits.sigmoid()
            class_probs = torch.cat([1 - class_probs, class_probs], dim=-1)
        else:
            class_probs = class_logits.softmax(-1)

        return {"class_logits": class_logits, "class_probs": class_probs, "masks": pred_masks}

    def forward(self, x: Tensor, mask: Tensor = None):
        # apply norm
        q = self.norm1(self.inital_q.expand(x.shape[0], -1, -1))
        x = self.norm2(x)

        intermediate_outputs: list | None = [] if self.aux_loss else None
        for i, layer in enumerate(self.layers):
            if self.aux_loss:
                assert intermediate_outputs is not None
                intermediate_outputs.append({"queries": q, **self.get_preds(q, x, mask)})
            q, x, attn_mask = layer(q, x, mask_net=self.mask_net, kv_mask=mask)

            if self.log_figure is not None and attn_mask is not None and self.step % 1000 == 0 and (i == 0 or i == len(self.layers) - 1):
                plt.figure(constrained_layout=True, dpi=300)
                plt.imshow(attn_mask[0], aspect="auto")
                self.log_figure(f"local_ca_mask_step{self.step}_layer{i}", plt, step=self.step)
        self.step += 1

        preds = {"queries": q, "x": x, **self.get_preds(q, x, mask)}
        if self.aux_loss:
            preds["intermediate_outputs"] = intermediate_outputs
        return preds


def get_masks(x: Tensor, q: Tensor, mask_net: nn.Module, input_pad_mask: Tensor | None = None):
    mask_tokens = mask_net(q)
    pred_masks = torch.einsum("bqe,ble->bql", mask_tokens, x)
    if input_pad_mask is not None:
        pred_masks[input_pad_mask.unsqueeze(1).expand_as(pred_masks)] = torch.finfo(pred_masks.dtype).min
    return pred_masks


class MaskFormer(nn.Module):
    def __init__(
        self,
        loss_config: Mapping,
        init_nets: ModuleList,
        mask_decoder: nn.Module,
        encoder: nn.Module | None = None,
        pool_net: nn.Module | None = None,
        tasks: nn.ModuleList | None = None,
    ):
        super().__init__()

        self.init_nets = init_nets
        self.encoder = encoder
        self.mask_decoder = mask_decoder
        self.pool_net = pool_net
        if tasks is None:
            tasks = []
        self.tasks = tasks

        # setup loss
        self.loss = MFLoss(**loss_config, tasks=tasks)

        # check init nets have the same output embedding size
        sizes = {list(init_net.parameters())[-1].shape[0] for init_net in self.init_nets}
        assert len(sizes) == 1

    def forward(self, inputs: dict, mask: dict, labels: dict | None = None):
        # initial embeddings
        embed_x = {}
        for init_net in self.init_nets:
            embed_x[init_net.name] = init_net(inputs)

        assert len(self.init_nets) == 1
        embed_x = next(iter(embed_x.values()))
        input_pad_mask = next(iter(mask.values()))

        # encoder/decoder
        if self.encoder is not None:
            embed_x = self.encoder(embed_x, mask=input_pad_mask)
        preds = {"embed_x": embed_x}

        # get mask and flavour predictions
        preds.update(self.mask_decoder(embed_x, input_pad_mask))

        # get the loss, updating the preds and labels with the best matching
        do_loss_matching = True  # set to false for inference timings
        if do_loss_matching:
            preds, labels, loss = self.loss(preds, labels)
        else:
            loss = {}  # disable the bipartite matching
            for task in self.tasks:
                if task.input_type == "queries":
                    preds.update(task(preds, labels))

        # pool the node embeddings
        if self.pool_net is not None:
            preds.update(self.pool_net(preds))

        # configurable tasks here
        for task in self.tasks:
            if task.input_type == "queries":
                task_loss = task.get_loss(preds, labels)
            else:
                task_preds, task_loss = task(preds, labels, input_pad_mask)
                preds.update(task_preds)
            loss.update(task_loss)

        return preds, loss
