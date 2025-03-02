"""
Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

from functools import partial

import torch
from torch import Tensor, nn

from hepattn.models import Attention, Dense, LayerNorm
from hepattn.models.transformer import Residual


class MaskformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        mask_attention: bool = True,
        bidirectional_ca: bool = True,
    ) -> None:
        super().__init__()

        self.mask_attention = mask_attention
        self.bidirectional_ca = bidirectional_ca

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim, norm=LayerNorm)
        self.q_ca = residual(Attention(dim, **attn_kwargs))
        self.q_sa = residual(Attention(dim, **attn_kwargs))
        self.q_dense = residual(Dense(dim, **dense_kwargs))
        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, **attn_kwargs))
            self.kv_dense = residual(Dense(dim, **dense_kwargs))

    def forward(self, q: Tensor, kv: Tensor, attn_mask: Tensor | None = None, kv_mask: Tensor | None = None) -> Tensor:
        assert kv_mask is None, "KV mask is not yet supported"

        # q are object queries, kv are hit embeddings
        # if we want to do mask attention
        if self.mask_attention:
            assert attn_mask is not None, "attn mask must be provided for mask attention"
            attn_mask = attn_mask.detach()
            # If a BoolTensor is provided, positions with `True` are not allowed to attend while `False` values will be unchanged.
            # if the attn mask is completely invalid for a given query, allow it to attend everywhere
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        else:
            attn_mask = None

        # update queries
        q = self.q_ca(q, k=kv, v=kv, attn_mask=attn_mask)  # needs input pad mask
        q = self.q_sa(q)
        q = self.q_dense(q)

        # update inputs
        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(2, 3)
            kv = self.kv_ca(kv, k=q, v=q, attn_mask=attn_mask)  # needs input pad mask
            kv = self.kv_dense(kv)

        return q, kv


class MaskformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_objects: int,
        md_config: dict,
        mask_net: nn.Module,
        class_net: nn.Module,
        intermediate_loss: bool = False,
        mask_threshold: float = 0.1,
    ):
        super().__init__()
        self.intermediate_loss = intermediate_loss
        self.mask_threshold = mask_threshold
        self.class_net = class_net
        self.mask_net = mask_net

        # initialize queries
        self.initial_q = nn.Parameter(torch.randn(num_objects, dim))

        # initialize layers
        self.layers = nn.ModuleList([MaskformerDecoderLayer(dim, **md_config) for _ in range(num_layers)])

    def forward(self, x: Tensor, input_pad_mask: Tensor = None):
        assert input_pad_mask is None, "Input padding is not yet supported"

        # broadcast queries to batch size
        q = self.initial_q.expand(x.shape[0], -1, -1)

        # Sanity checks
        assert q.shape[0] == x.shape[0], f"Batch size mismatch: q ({q.shape[0]}) vs x ({x.shape[0]})"
        assert q.shape[2] == x.shape[2], f"Feature dimension mismatch: q ({q.shape[2]}) vs x ({x.shape[2]})"

        # for each layer
        intermediate_outputs = []
        for layer in self.layers:
            # compute the attention mask for the current layer
            mask_logits = self.mask_pred(q, x, input_pad_mask)
            attn_mask = mask_logits.sigmoid() < self.mask_threshold
            attn_mask = attn_mask.unsqueeze(1)  # add head dimension

            # keep track of intermediate outputs
            if self.intermediate_loss:
                intermediate_outputs.append({"q": q.detach(), **self.class_pred(q), "mask_logits": mask_logits})

            # update queries and inputs
            q, x = layer(q, x, attn_mask=attn_mask)

        # construct final outputs
        preds = {"q": q, "x": x, **self.class_pred(q), "mask_logits": self.mask_pred(q, x, input_pad_mask)}
        if self.intermediate_loss:
            preds["intermediate_outputs"] = intermediate_outputs

        return preds

    def class_pred(self, q: Tensor):
        out = {}
        # get class predictions from queries
        if self.class_net is not None:
            class_logits = self.class_net(q)
            out["class_logits"] = class_logits
            if class_logits.shape[-1] == 1:  # binary classification
                class_probs = class_logits.sigmoid()
                class_probs = torch.cat([1 - class_probs, class_probs], dim=-1)
            else:  # multi-class classification
                class_probs = class_logits.softmax(-1)
            out["class_probs"] = class_probs.detach()

        return out

    def mask_pred(self, q: Tensor, x: Tensor, input_pad_mask: Tensor | None = None):
        if q.dim() != 3 or x.dim() != 3:
            raise ValueError(f"Expected 3D tensors, got q: {q.dim()}D, x: {x.dim()}D")

        # compute masks
        mask_tokens = self.mask_net(q)
        pred_masks = mask_tokens @ x.transpose(1, 2)

        # apply input padding to mask
        if input_pad_mask is not None:
            pred_masks[input_pad_mask.unsqueeze(1).expand_as(pred_masks)] = torch.finfo(pred_masks.dtype).min

        return pred_masks
