import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense


class Pooling(nn.Module):
    def __init__(
        self,
        input_object: str,
        output_object: str,
        dim: int,
        pool_net: nn.Module | None = None
    ) -> None:
        """
        A pooling module that applies optional transformation and weighted aggregation
        over input objects.

        Parameters
        ----------
        input_object : str
            Name of the input object.
        output_object : str
            Name of the output object.
        dim : int
            Dimensionality of the input embeddings.
        pool_net : nn.Module, optional
            Optional network applied to input objects before pooling.
        """

        super().__init__()

        self.input_object = input_object
        self.output_object = output_object
        self.dim = dim
        self.weight_net = Dense(dim, 1)
        self.pool_net = pool_net

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        x = inputs[self.input_object + "_embed"] # (..., N, E)
        if self.pool_net is not None:
            x = self.pool_net(x) # (..., N, E) -> (..., N, E)
        # Calculate a weight that will be used to pool the new embeddings (..., N, E) -> (..., N, 1)
        w = self.weight_net(x)
        # Set weights of padded entries to zero and make sure they sum to one
        w = w.masked_fill(~inputs[self.input_object  + "_valid"], -torch.inf)
        w = torch.softmax(w, dim=-2)  # (..., N, 1)
        # Weighted sum of all the embeddings (..., N, E) -> (..., E)
        x = torch.sum(x * w, dim=-2)
        return {self.output_object + "_embed": x}