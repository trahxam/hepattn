from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from hepattn.flex import relative_position, relative_position_wrapped
from hepattn.flex.sliding_window import sliding_window_mask, sliding_window_mask_wrapped
from hepattn.models import Attention, Dense, LayerNorm

create_block_mask = torch.compile(create_block_mask, dynamic=True)

SCORE_MODS = {
    "relative_position": relative_position,
    "relative_position_wrapped": relative_position_wrapped,
}


class DropPath(nn.Module):
    """Randomly drop layers: https://arxiv.org/abs/1603.09382"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(mask_shape, dtype=x.dtype, device=x.device)
        return x * mask.floor().div(keep_prob)


class LayerScale(nn.Module):
    """Learn residual strength: https://arxiv.org/abs/2103.17239"""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Residual(nn.Module):
    """Neatly wrap x = x + drop(scale * fn(norm(x)))"""

    def __init__(
        self,
        fn: nn.Module,
        norm: nn.Module | None = None,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        dim: int = 0,
    ) -> None:
        """Parameters
        ----------
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        norm : str, optional
            The normalization layer.
        layer_scale : float | None, optional
            The initial value for the layer_scale. If None, then no layer_scale is applied.
        drop_path : float, optional
            The drop path rate.
        dim : int
            The dimension of the input and output.
        """
        super().__init__()
        if norm is None:
            norm = LayerNorm
        self.fn = fn
        self.norm = norm(dim)
        self.ls = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.dp = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.dp(self.ls(self.fn(self.norm(x), **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: nn.Module = None,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        value_residual: bool = False,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ) -> None:
        """Encoder layer: self-attention -> feed-forward.

        Parameters
        ----------
        dim : int
            Dimension of the embeddings.
        norm : str, optional
            The normalization layer.
        drop_path : float, optional
            Drop path rate.
        layer_scale : float | None, optional
            Initial layer_scale value.
        dense_kwargs : dict | None, optional
            Keyword arguments for dense layer.
        attn_kwargs : dict | None, optional
            Keyword arguments for self-attention layer.
        """
        super().__init__()

        if attn_kwargs is None:
            attn_kwargs = {}
        if dense_kwargs is None:
            dense_kwargs = {}
        if norm is None:
            norm = LayerNorm

        self.dim = dim
        self.value_residual = value_residual
        residual = partial(Residual, dim=dim, norm=norm, layer_scale=layer_scale, drop_path=drop_path)
        self.attn = residual(Attention(self.dim, **attn_kwargs))
        self.dense = residual(Dense(self.dim, **dense_kwargs))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        score_mod: str | None = None,
        value_residual: bool = False,
        **layer_kwargs,
    ) -> None:
        """Transformer encoder.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        dim : int
            Dimension of the embeddings at each layer.
        window_size : int | None, optional
            The window size for the sliding window.
        value_residual : bool, optional
            Whether to use value residual.
        kwargs : dict
            Keyword arguments for EncoderLayer.
        """
        super().__init__()

        assert not window_wrap or window_size, "Window size must be set if window wrap is True."
        assert attn_type != "flex" or score_mod is None, "Score mod is only supported with flex attention."
        layer_kwargs = layer_kwargs or {}

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = attn_type
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.score_mod = SCORE_MODS[score_mod] if score_mod else None
        self.value_residual = value_residual

        # handle masking
        self.mask_mod = None
        self.q_len = None

        # handle attention
        attn_kwargs = layer_kwargs.get("attn_kwargs", None) or {}
        attn_kwargs["attn_type"] = attn_type
        if value_residual:
            attn_kwargs["value_residual"] = True
        if attn_type == "flash" and window_size is not None:
            attn_kwargs["window_size"] = window_size
        layer_kwargs["attn_kwargs"] = attn_kwargs

        self.layers = torch.nn.ModuleList([EncoderLayer(dim=dim, **layer_kwargs) for _ in range(num_layers)])

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # Concatenate dictionary values
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)

        # Initialise sliding window mask
        if self.mask_mod is None and self.attn_type != "flash" and self.window_size:
            self.q_len = torch.tensor([1], device=x.device)
            self.mask_mod = (
                sliding_window_mask(self.window_size) if not self.window_wrap else sliding_window_mask_wrapped(self.window_size, self.q_len)
            )

        # Handle masking
        mask = None
        q_len = x.shape[-2]
        if self.attn_type == "torch" and self.mask_mod:
            mask = create_mask(self.mask_mod, 1, 1, q_len, q_len, device=x.device)
        elif self.attn_type == "flex" and self.mask_mod:
            self.q_len[0] = q_len
            mask = create_block_mask(self.mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, device=x.device)

        # Add wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = torch.cat([x[:, -self.window_size // 2 :], x, x[:, : self.window_size // 2]], dim=1)

        # Apply layers
        initial_values = {} if self.value_residual else None
        for layer in self.layers:
            x = layer(x, mask=mask, score_mod=self.score_mod, initial_values=initial_values, **kwargs)

        # Remove wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = x[:, self.window_size // 2 : -self.window_size // 2]

        return x
