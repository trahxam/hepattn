import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation function.

    Splits the last dimension in half and applies SiLU-gated multiplication,
    as described in https://arxiv.org/abs/2002.05202.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU activation."""
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * F.silu(x2)
