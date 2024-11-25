from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from hepattn.flex.sliding_window import sliding_window_mask
from hepattn.models import Attention, Dense, LayerNorm

create_block_mask = torch.compile(create_block_mask)


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

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x + self.dp(self.ls(self.fn(self.norm(x), *args, **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: nn.Module = None,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
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
        residual = partial(Residual, dim=dim, norm=norm, layer_scale=layer_scale, drop_path=drop_path)
        self.attn = residual(Attention(self.dim, **attn_kwargs))
        self.dense = residual(Dense(self.dim, **dense_kwargs))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, x, x, **kwargs))


class Encoder(nn.Module):
    def __init__(self, num_layers: int, dim: int, window_size: int | None = None, **kwargs) -> None:
        """Transformer encoder.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        dim : int
            Dimension of the embeddings at each layer.
        window_size : int | None, optional
            The window size for the sliding window.
        kwargs : dict
            Keyword arguments for EncoderLayer.
        """
        super().__init__()

        if kwargs is None:
            kwargs = {"attn_kwargs": {"attn_type": "torch"}}
        elif "attn_kwargs" not in kwargs:
            kwargs["attn_kwargs"] = {"attn_type": "torch"}

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = kwargs["attn_kwargs"]["attn_type"]

        self.layers = torch.nn.ModuleList([EncoderLayer(dim=dim, **kwargs) for _ in range(num_layers)])

        if self.attn_type == "flash":
            kwargs["attn_kwargs"]["window_size"] = window_size
            self.mask_mod = None
        else:
            self.mask_mod = sliding_window_mask(10)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if isinstance(x, dict):
            x = torch.cat(list(x.values()), dim=1)

        mask = None
        q_len = x.shape[-2]
        if self.attn_type == "torch":
            mask = create_mask(self.mask_mod, 1, 1, q_len, q_len, device=x.device)
        elif self.attn_type == "flex":
            mask = create_block_mask(self.mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=q_len, device=x.device)

        for layer in self.layers:
            x = layer(x, mask=mask, **kwargs)

        return x
