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
    def __init__(self, input_name: str, fields: list[str], sym_fields: list[str], dim: int, alpha=1000):
        """Positional encoder.

        Parameters
        ----------
        input_name : str
            The name of the input object that will be encoded.
        fields : list[str]
            List of fields belonging to the object to apply the positional encoding to.
        fields : list[str]
            List of fields that should use a rotationally symmetric positional encoding.
        dim : int
            Dimension to project the positional encoding into.
        alpha : float
            Scaling factor hyperparamater for the positional encoding.
        """
        super().__init__()

        self.input_name = input_name
        self.fields = fields
        self.sym_fields = sym_fields
        self.dim = dim
        self.alpha = alpha

        self.per_input_dim = self.dim // len(self.fields)
        self.remainder_dim = self.dim % len(self.fields)

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
        for field in self.fields:
            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc
            encodings.append(pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha))
        if self.remainder_dim:
            # Make sure to allow for arbitrary batch shape
            encodings.append(torch.zeros_like(encodings[0])[...,:self.remainder_dim])
        encodings = torch.cat(encodings, dim=-1)
        return encodings
    

class RandomFourierFeatureEncoder(nn.Module):
    """
    An implementation of Gaussian Fourier positional encoding.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    see https://arxiv.org/abs/2006.10739

    Parameters
        ----------
        inputs : dict
            Dictionary of inputs.

        Returns
        -------
        torch.Tensor
            Positional encoding of the input variables.
    """
    def __init__(self, input_name: str, fields: list[str], dim: int, sigma: float = 10.0, train: bool = False):
        super().__init__()

        assert dim % 2 == 0, "Embedding dimension must be divisible by 2."

        self.input_name = input_name
        self.fields = fields
        self.dim = dim
        self.encoding_dim = int(self.dim / 2)
        self.B = torch.randn([len(self.fields), self.encoding_dim]) * sigma

        if train:
            self.B = nn.Parameter(self.B)

    def forward(self, inputs: dict):
        pos = torch.stack([inputs[f"{self.input_name}_{field}"] for field in self.fields], dim=-1)
        pos_enc = torch.matmul(pos, self.B.to(pos.device))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc
