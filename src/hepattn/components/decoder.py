from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.components.attention import Attention
from hepattn.components.dense import Dense
from hepattn.components.encoder import Residual
from hepattn.components.norm import get_hybrid_norm_config


class KMeansCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        update: str = "mean",
        value_proj: bool = False,
        mask_attn: bool = False,
        respect_attn_mask: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert update in {"sum", "mean"}
        self.dim = dim
        self.update = update
        self.respect_attn_mask = mask_attn or respect_attn_mask
        self.eps = eps
        self.v_proj = nn.Linear(dim, dim, bias=False) if value_proj else None

    def forward(
        self,
        q: Tensor,  # (B, N, D)
        k: Tensor | None = None,  # (B, M, D)
        v: Tensor | None = None,  # (B, M, D)
        attn_mask: Tensor | None = None,  # (B, N, M) bool
        q_mask: Tensor | None = None,  # (B, N) bool
        kv_mask: Tensor | None = None,  # (B, M) bool
        logits: Tensor | None = None,  # (B, N, M)
        **kwargs,
    ) -> Tensor:
        if v is None:
            raise ValueError("KMeansCrossAttention requires v (values).")

        neg_inf = float("-inf")

        if logits is None:
            if k is None:
                raise ValueError("Provide either logits or k.")
            logits = q @ k.transpose(-2, -1)

        if q_mask is not None:
            logits = logits.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
        if self.respect_attn_mask and attn_mask is not None:
            logits = logits.masked_fill(~attn_mask, neg_inf)
        if kv_mask is not None:
            logits = logits.masked_fill(~kv_mask.unsqueeze(-2), neg_inf)

        max_val, idx = logits.max(dim=-2)  # (B, M), (B, M)
        valid = torch.isfinite(max_val)  # (B, M)

        vv = v if self.v_proj is None else self.v_proj(v)  # (B, M, D)
        vv = vv * valid.unsqueeze(-1).to(vv.dtype)

        B, M, D = vv.shape
        N = logits.shape[-2]

        out = vv.new_zeros((B, N, D))
        out.scatter_add_(1, idx.unsqueeze(-1).expand(B, M, D), vv)

        if self.update == "mean":
            counts = vv.new_zeros((B, N))
            counts.scatter_add_(1, idx, valid.to(vv.dtype))
            out = out / (counts.clamp_min(1.0).unsqueeze(-1) + self.eps)

        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        bidirectional_ca: bool = True,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        cross_attn_mode: Literal["softmax", "kmeans"] = "softmax",
        kmeans_kwargs: dict | None = None,
    ) -> None:
        """Initialize a cross-attention decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            bidirectional_ca: Enable bidirectional cross-attention.
            qkv_norm: Apply normalization to QKV in attention.
            hybrid_norm: Enable hybrid normalization from 2503.04598.
            cross_attn_mode: "softmax" (standard attention) or "kmeans" (kMaX-style hard assignment update).
            kmeans_kwargs: Optional kwargs passed to KMeansCrossAttention when cross_attn_mode="kmeans".
        """
        super().__init__()
        self.dim = dim
        self.bidirectional_ca = bidirectional_ca
        self.cross_attn_mode = cross_attn_mode

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs = attn_kwargs or {}
        self.attn_type = attn_kwargs.get("attn_type", "torch")
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim)

        if self.cross_attn_mode == "kmeans":
            kmeans_kwargs = kmeans_kwargs or {}
            self.q_ca = residual(KMeansCrossAttention(dim, **kmeans_kwargs), norm=attn_norm)
        else:
            self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)

        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
            self.kv_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        query_posenc: Tensor | None = None,
        key_posenc: Tensor | None = None,
        attn_mask_transpose: Tensor | None = None,
        logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for the decoder layer.

        Args:
            q: Query embeddings.
            kv: Key/value embeddings.
            attn_mask: Optional attention mask (B, N, M) for q<-kv, or blockmask for flex.
            q_mask: Optional query mask (B, N).
            kv_mask: Optional key/value mask (B, M).
            query_posenc: Optional query positional encoding.
            key_posenc: Optional key positional encoding.
            attn_mask_transpose: Optional transposed attention mask for flex attention.
            logits: If cross_attn_mode="kmeans", dense logits (B, N, M).

        Returns:
            tuple[Tensor, Tensor]: Updated (q, kv).
        """
        q_pe = q if query_posenc is None else q + query_posenc
        kv_pe = kv if key_posenc is None else kv + key_posenc

        if self.cross_attn_mode == "kmeans":
            q = self.q_ca(
                q_pe,
                k=kv_pe,
                v=kv,
                attn_mask=attn_mask,
                q_mask=q_mask,
                kv_mask=kv_mask,
                logits=logits,
            )
        else:
            q = self.q_ca(q_pe, k=kv_pe, v=kv, attn_mask=attn_mask, q_mask=q_mask, kv_mask=kv_mask)

        q = self.q_dense(q)
        q = self.q_sa(q, k=q, v=q, q_mask=q_mask)

        # Update key/constituent embeddings with the query/object embeddings
        if self.bidirectional_ca:
            if attn_mask is not None:
                if self.attn_type == "flex":
                    assert attn_mask_transpose is not None, "attn_mask_transpose must be provided for flex attention"
                attn_mask = attn_mask_transpose if attn_mask_transpose is not None else attn_mask.transpose(-2, -1)

            q_pe = q if query_posenc is None else q + query_posenc
            kv_pe = kv if key_posenc is None else kv + key_posenc

            kv = self.kv_ca(kv_pe, k=q_pe, v=q, attn_mask=attn_mask, q_mask=kv_mask, kv_mask=q_mask)
            kv = self.kv_dense(kv)

        return q, kv

    def set_backend(self, attn_type: str) -> None:
        """Set the backend for the attention layers.

        Args:
            attn_type: Attention implementation type to use.
        """
        if self.cross_attn_mode != "kmeans":
            self.q_ca.fn.set_backend(attn_type)
        self.q_sa.fn.set_backend(attn_type)

        if self.bidirectional_ca:
            self.kv_ca.fn.set_backend(attn_type)
