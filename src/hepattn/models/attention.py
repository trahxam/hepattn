import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch import BoolTensor, Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

ATTN_TYPES = {
    "torch": F.scaled_dot_product_attention,
    "flex": flex_attention,
    "flash": flash_attn_func,
}


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
        attn_type: str = "torch",
        torch_compile: bool = False,
        window_size: int | None = None,
        value_residual: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "num_heads must divide dim."
        assert attn_type in ATTN_TYPES, f"Invalid attention type: {attn_type}"
        assert window_size is None or attn_type == "flash", "Window size can only be specified for flash attention"

        self.dim = dim
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.attn = ATTN_TYPES[attn_type]
        self.window_size = None
        self.value_residual = value_residual

        if attn_type == "flash":
            self.window_size = (window_size // 2, window_size // 2) if window_size is not None else (-1, -1)
        if torch_compile or attn_type == "flex":
            self.attn = torch.compile(self.attn, dynamic=True)

        self.q_proj = nn.Linear(dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, dim, bias=bias)

        if self.value_residual:
            self.value_residual_mix = nn.Sequential(nn.Linear(dim, num_heads), nn.Sigmoid())

    def separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        x = x.unflatten(-1, (num_heads, -1))  # B S D -> B S H Dh
        if self.attn_type != "flash":
            x = x.transpose(-3, -2)  # B S H Dh -> B H S Dh
        return x

    def recombine_heads(self, x: Tensor) -> Tensor:
        if self.attn_type != "flash":
            x = x.transpose(-3, -2)  # B H S Dh -> B S H Dh
        return x.flatten(-2)  # B S H Dh -> B S D

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        mask: BlockMask | BoolTensor | None = None,
        initial_values: dict | None = None,
    ) -> Tensor:
        # Default to self-attention
        k = k or q
        v = v or q

        # Mix for value residual
        mix = None
        if self.value_residual:
            mix = self.value_residual_mix(q)
            if self.attn_type != "flash":
                mix = mix.transpose(-1, -2)
            mix = mix.unsqueeze(-1)

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # Residual connection with initial values
        if self.value_residual and not initial_values:
            initial_values["v"] = v
        elif self.value_residual:
            v = v * mix + initial_values["v"] * (1.0 - mix)

        # Fused attention
        if self.attn_type == "flex":
            out = self.attn(q, k, v, block_mask=mask)
        elif self.attn_type == "torch":
            out = self.attn(q, k, v, attn_mask=mask)
        elif self.attn_type == "flash":
            out = self.attn(q, k, v, window_size=self.window_size)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        # Get output
        out = self.recombine_heads(out)
        return self.out_proj(out)
