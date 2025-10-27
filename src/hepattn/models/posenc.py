# ruff: noqa: E741

import math

import torch
from torch import Tensor, nn

from hepattn.utils.spherical_harmonics_basis import spherical_harmonic as spherical_harmonic_analytic
from hepattn.utils.spherical_harmonics_closed_form import spherical_harmonic as spherical_harmonic_closed_form


def get_omegas(alpha, dim, base, **kwargs):
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), base, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, base, **kwargs)
    return omega_1, omega_2


def pos_enc_symmetric(xs, dim, alpha=1000, base=100):
    """Symmetric positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns:
    -------
    torch.Tensor
        Symmetric positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, base, **kwargs)
    p1 = (xs.sin() * omega_1).sin()
    p2 = (xs.cos() * omega_2).sin()
    return torch.cat((p1, p2), dim=-1)


def pos_enc(xs, dim, alpha=1000, base=100):
    """Positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns:
    -------
    torch.Tensor
        Positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, base, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


class PositionEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000, base=100):
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
        base : float
            Base for the logarithmic scale.
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

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.

        Returns:
        -------
        torch.Tensor
            Positional encoding of the input variables.
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
        xs = torch.cat([inputs[f"{self.input_name}_{field}"].unsqueeze(-1) for field in self.fields], dim=-1)
        xs = 2 * self.pi * xs
        xs @= self.B
        return torch.cat([torch.sin(xs), torch.cos(xs)], dim=-1)


class SphericalHarmonicEncoder(nn.Module):
    def __init__(
        self,
        input_name: str,
        dim: int,
        legendre_polys: int = 12,
        harmonics_calculation="closed-form",
        phi_field="phi",
        theta_field="theta",
        radius_field="r",
    ):
        """Encodes spherical coordinates by combining spherical harmonics (angular) and sinusoidal (radial) encodings.

        Args:
            input_name (str): Prefix for input dict keys (e.g. "hit" -> "hit_phi").
            dim (int): Total embedding dimension.
            legendre_polys (int): Number of Legendre polynomials (angular resolution).
            harmonics_calculation (str): "analytic" (fast, up to l≈50) or "closed-form" (slower, general).
            phi_field (str): Field name suffix for azimuthal angle. Default: "phi".
            theta_field (str): Field name suffix for polar angle. Default: "theta".
            radius_field (str): Field name suffix for radius. Default: "r".

        Inputs:
            inputs (dict[str, Tensor]): Must contain
                f"{input_name}_{phi_field}",
                f"{input_name}_{theta_field}",
                f"{input_name}_{radius_field}".
        """
        super().__init__()
        self.input_name = input_name
        self.L = int(legendre_polys)
        self.M = int(legendre_polys)
        self.dim = dim

        self.phi_field = phi_field
        self.theta_field = theta_field
        self.radius_field = radius_field

        if harmonics_calculation == "closed-form":
            self.spherical_harmonic = spherical_harmonic_closed_form
        elif harmonics_calculation == "analytic":
            self.spherical_harmonic = spherical_harmonic_analytic

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        phi = inputs[f"{self.input_name}_{self.phi_field}"]
        theta = inputs[f"{self.input_name}_{self.theta_field}"]
        radius = inputs[f"{self.input_name}_{self.radius_field}"]

        # Calculate the angular embedding
        harmonics = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self.spherical_harmonic(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                harmonics.append(y)

        # Create the angular embedding
        angular = torch.stack(harmonics, dim=-1)

        # Use the remainder of the dimension for the radial embedding
        radial = pos_enc(radius, dim=self.dim - angular.shape[-1])

        # Combine the angular and radial embeddings
        return torch.cat([angular, radial], dim=-1)
