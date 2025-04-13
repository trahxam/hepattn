import torch
from torch import nn


def get_omegas(alpha, dim, **kwargs):
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), 100, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, 100, **kwargs)
    return omega_1, omega_2


def pos_enc_symmetric(xs, dim, alpha=1000):
    """Symmetric positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns
    -------
    torch.Tensor
        Symmetric positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs.sin() * omega_1).sin()
    p2 = (xs.cos() * omega_2).sin()
    return torch.cat((p1, p2), dim=-1)


def pos_enc(xs, dim, alpha=1000):
    """Positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns
    -------
    torch.Tensor
        Positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


class PositionEncoder(nn.Module):
    def __init__(self, variables: list[str], dim: int, alpha=1000):
        """Positional encoder.

        Parameters
        ----------
        variables : list[str]
            List of variables to apply the positional encoding to.
        """
        super().__init__()

        self.variables = variables
        self.dim = dim
        self.alpha = alpha
        self.SYM_VARS = {"phi"}

        self.per_input_dim = self.dim // len(self.variables)
        self.remainder_dim = self.dim % len(self.variables)

    def forward(self, inputs: dict):
        """Apply positional encoding to the inputs.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.

        Returns
        -------
        torch.Tensor
            Positional encoding of the input variables.
        """
        encodings = []
        for var in self.variables:
            pos_enc_fn = pos_enc_symmetric if var in self.SYM_VARS else pos_enc
            encodings.append(pos_enc_fn(inputs[var], self.per_input_dim, self.alpha))
        if self.remainder_dim:
            encodings.append(torch.zeros_like(encodings[0])[..., : self.remainder_dim])
        encodings = torch.cat(encodings, dim=-1)
        return encodings
