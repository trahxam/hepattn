import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from torch import BoolTensor, Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, _score_mod_signature, flex_attention

from hepattn.models.norm import LayerNorm

ATTN_TYPES = {
    "torch": F.scaled_dot_product_attention,
    "flex": flex_attention,
    "flash": flash_attn_func,
    "flash-varlen": flash_attn_varlen_func,
}

VARLEN_ATTN_TYPES = [
    # TODO: Implement kv masking for torch
    # "torch",
    "flash-varlen",
]

ATTN_MASK_ATTN_TYPES = [
    "torch",
    "flex",
]

WINDOW_ATTN_TYPES = [
    "flash",
    "flash-varlen",
]

# For now basically just defines which attention types expect (B, S, H, Dh) instead of (B, H, S, Dh)
FLASH_ATTN_TYPES = [
    "flash",
    "flash-varlen",
]


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        window_size: int | None = None,
        query_window_size: int | None = None,
        value_residual: bool = False,
        qkv_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "num_heads must divide dim."
        assert attn_type in ATTN_TYPES, f"Invalid attention type: {attn_type}"
        assert window_size is None or attn_type in WINDOW_ATTN_TYPES, f"Window size can only be specified for {WINDOW_ATTN_TYPES}"

        self.dim = dim
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.attn = ATTN_TYPES[attn_type]
        self.window_size = None
        self.value_residual = value_residual
        self.qkv_norm = qkv_norm

        if attn_type in FLASH_ATTN_TYPES:
            # TODO: Will need to change when supporting window with flex
            self.window_size = (window_size // 2, window_size // 2) if window_size is not None else (-1, -1)
        if torch_compile or attn_type == "flex":
            self.attn = torch.compile(self.attn, dynamic=True)

        self.q_proj = nn.Linear(dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, dim, bias=bias)

        if self.value_residual:
            self.value_residual_mix = nn.Sequential(nn.Linear(dim, num_heads), nn.Sigmoid())

        if self.qkv_norm:
            self.q_norm = LayerNorm(dim)
            self.k_norm = LayerNorm(dim)
            self.v_norm = LayerNorm(dim)

    def separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        x = x.unflatten(-1, (num_heads, -1))  # B S D -> B S H Dh
        if self.attn_type not in FLASH_ATTN_TYPES:
            x = x.transpose(-3, -2)  # B S H Dh -> B H S Dh
        return x

    def recombine_heads(self, x: Tensor) -> Tensor:
        if self.attn_type not in FLASH_ATTN_TYPES:
            x = x.transpose(-3, -2)  # B H S Dh -> B S H Dh
        return x.flatten(-2)  # B S H Dh -> B S D

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BlockMask | BoolTensor | None = None,
        score_mod: _score_mod_signature | None = None,
        initial_values: dict | None = None,
    ) -> Tensor:
        """
        Multi-head attention forward pass.

        Parameters
        ----------
        q : Tensor
            Queries tensor of shape (B, S, D).
        k : Tensor, optional
            Keys tensor of shape (B, S, D). If None, defaults to q.
        v : Tensor, optional
            Values tensor of shape (B, S, D). If None, defaults to q.
        attn_mask : BlockMask | BoolTensor, optional
            Attention mask to apply. If None, no mask is applied.
        score_mod : _score_mod_signature, optional
            Score modifier function for flex attention. If None, no score modifier is applied.
        initial_values : dict, optional
            Initial values for value residual connection.
        """
        # Default to self-attention
        k = k if k is not None else q
        v = v if v is not None else q

        # Mix for value residual
        mix = None
        if self.value_residual:
            mix = self.value_residual_mix(q)
            mix = mix.unsqueeze(-1)
            if self.attn_type not in FLASH_ATTN_TYPES:
                mix = mix.transpose(-2, -3)

        # Input projections, shape is (batch, seq, dim)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Normalize queries, keys, and values
        if self.qkv_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)

        # Separate into heads, output shape is (batch, head, seq, head_dim)
        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        # Residual connection with initial values
        if self.value_residual:
            if not initial_values:
                initial_values["v"] = v
            else:
                v = v * mix + initial_values["v"] * (1.0 - mix)

        # Check that the specified attention backend actualy supports kv masking / jagged inputs
        if kv_mask is not None:
            msg = f"Only the backends {VARLEN_ATTN_TYPES} support kv masking"
            assert self.attn_type in VARLEN_ATTN_TYPES, msg

        if attn_mask is not None:
            msg = f"Only the backends {ATTN_MASK_ATTN_TYPES} support attention masking"
            assert self.attn_type in ATTN_MASK_ATTN_TYPES, msg

        # Fused attention
        if self.attn_type == "flex":
            out = self.attn(q, k, v, block_mask=attn_mask, score_mod=score_mod)

        elif self.attn_type == "torch":
            # Have to expand the attention mask so that it is broadcasted over the head dimension
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(-3)

            out = self.attn(q, k, v, attn_mask=attn_mask)

        elif self.attn_type == "flash":
            out = self.attn(q, k, v, window_size=self.window_size)
            
        elif self.attn_type == "flash-varlen":
            # TODO: Implement a packed version for the self attention case

            assert q_mask is None, "query mask not currently supported"
            q_mask = torch.full((q.shape[0], q.shape[1]), True, dtype=torch.bool, device=q.device)

            # If no kv mask is provided, all kv are valid
            if kv_mask is None:
                kv_mask = torch.full((k.shape[0], k.shape[1]), True, dtype=torch.bool, device=q.device)

            q_lens = q_mask.sum(dim=1, dtype=torch.int32)
            kv_lens = kv_mask.sum(dim=1, dtype=torch.int32)

            # q has shape (B, S, H, Dh)
            H = q.shape[-2]
            Dh = q.shape[-1]
            B = q.shape[0]

            max_seqlen_q = q.shape[-3]
            max_seqlen_k = k.shape[-3]

            q_flat = q[q_mask].reshape(-1, H, Dh)
            k_flat = k[kv_mask].reshape(-1, H, Dh)
            v_flat = v[kv_mask].reshape(-1, H, Dh)

            cu_seqlens_q = torch.nn.functional.pad(q_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            cu_seqlens_k = torch.nn.functional.pad(kv_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))

            out = self.attn(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                window_size=self.window_size,
                )
            
            # Reshape to (B, S, H, Dh)
            out = out.reshape(B, max_seqlen_q, H, Dh)

        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        # Get output
        out = self.recombine_heads(out)
        return self.out_proj(out)
