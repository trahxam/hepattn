import torch
from torch import Tensor, nn


class KMeansCrossAttention(nn.Module):
    def __init__(self, dim: int, update="mean", value_proj=False,
                 mask_attn=False, eps=1e-6):
        super().__init__()
        assert update in {"sum", "mean"}
        self.dim = dim
        self.update = update
        self.mask_attn = mask_attn
        self.eps = eps
        self.v_proj = nn.Linear(dim, dim, bias=False) if value_proj else None

    def forward(
        self,
        q: Tensor,                       # (B, N, D)
        k: Tensor | None = None,         # (B, M, D)
        v: Tensor | None = None,         # (B, M, D)
        attn_mask: Tensor | None = None,  # (B, N, M) bool
        q_mask: Tensor | None = None,    # (B, N) bool
        kv_mask: Tensor | None = None,   # (B, M) bool
        affinity_logits: Tensor | None = None,  # (B, N, M)
        **kwargs,
    ) -> Tensor:
        if v is None:
            raise ValueError("KMeansCrossAttention requires v (values).")

        # logits: (B, N, M)
        if affinity_logits is None:
            if k is None:
                raise ValueError("Provide either affinity_logits or k.")
            logits = q @ k.transpose(-2, -1)
        else:
            logits = affinity_logits

        # Mask to -inf
        neg_inf = float("-inf")
        if q_mask is not None:
            logits = logits.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
        if kv_mask is not None:
            logits = logits.masked_fill(~kv_mask.unsqueeze(-2), neg_inf)
        if self.mask_attn and (attn_mask is not None):
            logits = logits.masked_fill(~attn_mask, neg_inf)

        # One pass gives both argmax indices and max values
        max_val, idx = logits.max(dim=-2)          # max over N -> (B, M), (B, M)
        valid = torch.isfinite(max_val)            # (B, M) tokens with any allowed query

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
