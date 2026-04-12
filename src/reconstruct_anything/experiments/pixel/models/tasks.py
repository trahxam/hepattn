import torch
from torch import Tensor

from reconstruct_anything.components.dense import Dense
from reconstruct_anything.models.tasks import GaussianRegressionTask


class ObjectGaussianRegressionTask(GaussianRegressionTask):
    """Gaussian regression task that operates on per-object embeddings."""

    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        cost_weight: float,
        dim: int,
    ):
        """Gaussian regression task for objects.

        Args:
            name: Name of the task.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            fields: List of fields to regress.
            loss_weight: Weight for the loss function.
            cost_weight: Weight for the cost function.
            dim: Embedding dimension.
        """
        super().__init__(name, output_object, target_object, fields, loss_weight, cost_weight)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [
            output_object + "_mu",
            output_object + "_ubar",
            output_object + "_u",
        ]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor]) -> Tensor:
        """Produce raw latent vector from object embeddings."""
        return self.net(x[self.input_object + "_embed"])

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute pairwise negative log-likelihood costs for bipartite matching."""
        mu = outputs[self.output_object + "_mu"].to(torch.float32)
        ubar = outputs[self.output_object + "_ubar"].to(torch.float32)
        u = outputs[self.output_object + "_u"].to(torch.float32)
        y = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)

        num_objects = y.shape[1]
        mu = mu.unsqueeze(2).expand(-1, -1, num_objects, -1)
        ubar = ubar.unsqueeze(2).expand(-1, -1, num_objects, -1, -1)
        u = u.unsqueeze(2).expand(-1, -1, num_objects, -1, -1)
        diagu = torch.diagonal(u, offset=0, dim1=-2, dim2=-1)
        y = y.unsqueeze(1).expand(-1, num_objects, -1, -1)

        z = torch.einsum("...ij,...j->...i", ubar, y - mu)
        zsq = torch.einsum("...i,...i->...", z, z)
        jac = torch.sum(diagu, dim=-1)

        log_likelihood = self.likelihood_norm - 0.5 * zsq + jac
        log_likelihood *= targets[f"{self.target_object}_valid"].unsqueeze(1).type_as(log_likelihood)
        costs = -log_likelihood

        return {"nll": self.cost_weight * costs}
