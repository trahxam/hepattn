import math

import torch
from torch import Tensor, nn


def get_omegas(alpha, dim, base, **kwargs):
    """Compute logarithmically-spaced angular frequency pairs for positional encoding."""
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), base, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, base, **kwargs)
    return omega_1, omega_2


def pos_enc_symmetric(xs, dim, alpha=1000, base=100):
    """Compute a rotationally symmetric positional encoding.

    Args:
        xs: Input tensor of positions.
        dim: Dimension of the positional encoding.
        alpha: Scaling factor for the angular frequencies.
        base: Base for the logarithmic frequency scale.

    Returns:
        Symmetric positional encoding tensor.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, base, **kwargs)
    p1 = (xs.sin() * omega_1).sin()
    p2 = (xs.cos() * omega_2).sin()
    return torch.cat((p1, p2), dim=-1)


def pos_enc(xs, dim, alpha=1000, base=100):
    """Compute a standard sinusoidal positional encoding.

    Args:
        xs: Input tensor of positions.
        dim: Dimension of the positional encoding.
        alpha: Scaling factor for the angular frequencies.
        base: Base for the logarithmic frequency scale.

    Returns:
        Positional encoding tensor.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, base, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


class PositionEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000, base=100):
        """Initialize the PositionEncoder.

        Args:
            input_name: Name of the input object to encode (e.g. 'hit').
            fields: List of fields on the input object to apply positional encoding to.
            dim: Total output dimension of the positional encoding.
            sym_fields: Fields that should use rotationally symmetric positional encoding.
            alpha: Scaling factor hyperparameter for the angular frequencies.
            base: Base for the logarithmic frequency scale.
        """
        super().__init__()

        self.input_name = input_name
        self.fields = fields
        self.sym_fields = sym_fields or []
        self.dim = dim
        self.alpha = alpha
        self.base = base

        self.per_input_dim = self.dim // len(self.fields)
        self.remainder_dim = self.dim % len(self.fields)

    def forward(self, inputs: dict):
        """Apply positional encoding to the inputs.

        Args:
            inputs: Dictionary of input tensors keyed by ``{input_name}_{field}``.

        Returns:
            Concatenated positional encoding over all fields.
        """
        encodings = []
        for field in self.fields:
            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc
            encodings.append(pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha, self.base))
        if self.remainder_dim:
            encodings.append(torch.zeros_like(encodings[0])[..., : self.remainder_dim])
        return torch.cat(encodings, dim=-1)


class FourierPositionEncoder(nn.Module):
    """An implementation of Gaussian Fourier positional encoding.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    see https://arxiv.org/abs/2006.10739
    """

    def __init__(self, input_name: str, dim: int, fields: list[str], scale: float = 1) -> None:
        super().__init__()
        assert scale > 0
        assert dim % 2 == 0, "Dimension must be even"
        self.input_name = input_name
        self.fields = fields
        self.B = torch.nn.parameter.Buffer(scale * torch.randn((len(fields), dim // 2)))
        self.pi = torch.tensor(math.pi)

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """Apply Gaussian Fourier positional encoding to the inputs."""
        xs = torch.cat([inputs[f"{self.input_name}_{field}"].unsqueeze(-1) for field in self.fields], dim=-1)
        xs = 2 * self.pi * xs
        xs @= self.B
        return torch.cat([torch.sin(xs), torch.cos(xs)], dim=-1)
