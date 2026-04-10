import torch
from torch import Tensor, nn

from hepattn.components.dense import Dense


class Pooling(nn.Module):
    def __init__(self, dim: int, pool_net: nn.Module | None = None) -> None:
        """Initialize the Pooling module.

        Args:
            dim: Dimensionality of the input embeddings.
            pool_net: Optional network applied to input entities before pooling.
        """
        super().__init__()

        self.dim = dim
        self.weight_net = Dense(dim, 1)
        self.pool_net = pool_net

    def forward(self, x: Tensor, x_valid: Tensor) -> Tensor:
        """Pool input embeddings into a single output embedding via learned attention weights.

        Args:
            x: Input embeddings of shape (..., N, D).
            x_valid: Boolean mask of shape (..., N). True for valid (non-padded) entries.

        Returns:
            Pooled embedding of shape (..., D).
        """
        if self.pool_net is not None:
            x = self.pool_net(x)  # (..., N, E) -> (..., N, E)
        # Calculate a weight that will be used to pool the new embeddings (..., N, E) -> (..., N, 1)
        w = self.weight_net(x).squeeze(-1)  # (..., N)
        # Set weights of padded entries to zero and make sure they sum to one
        w = w.masked_fill(~x_valid, -torch.inf)
        w = torch.softmax(w, dim=-1)  # (..., N)
        w = w.masked_fill(~x_valid, 0.0)
        # Weighted sum of all the embeddings (..., N, E) -> (..., E)
        return torch.sum(x * w.unsqueeze(-1), dim=-2)
