import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_mask, flex_attention


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.

    Adapted from https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/modeling/transformer.py
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        flex: bool = True,
        torch_compile: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "num_heads must divide dim."
        assert not (flex and not torch_compile), "must compile with flex."

        self.dim = dim
        self.num_heads = num_heads
        self.flex = flex
        self.attn = flex_attention if flex else F.scaled_dot_product_attention
        if torch_compile:
            self.attn = torch.compile(self.attn, dynamic=False)

        self.q_proj = nn.Linear(dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, dim, bias=bias)

    def separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        x = x.unflatten(-1, (num_heads, -1))  # B S D -> B S H Dh
        return x.transpose(-3, -2)  # B S H Dh -> B H S Dh

    def recombine_heads(self, x: Tensor) -> Tensor:
        x = x.transpose(-3, -2)  # B H S Dh -> B S H Dh
        return x.flatten(-2)  # B S H Dh -> B S D

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: BlockMask | BoolTensor | None = None) -> Tensor:
        q_len = q.shape[-2]
        kv_len = k.shape[-2]

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # Attention
        if self.flex:
            out = self.attn(q, k, v, block_mask=mask)
        else:
            mask = None
            if mask is not None:
                # materialise full mask
                mask = create_mask(mask, 1, 1, q_len, kv_len, device=q.device)
            out = self.attn(q, k, v, attn_mask=mask)

        # Get output
        out = self.recombine_heads(out)
        return self.out_proj(out)
