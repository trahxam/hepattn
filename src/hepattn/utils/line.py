"""Energy-weighted mean η/φ fit for calorimeter clusters."""
from __future__ import annotations

import torch


def weighted_mean_eta_phi(
    eta: torch.Tensor,
    phi: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy-weighted mean pseudorapidity and azimuthal angle.

    φ uses circular statistics (mean of unit vectors on the circle) to
    handle the ±π discontinuity correctly.

    Args:
        eta: pseudorapidity of each hit, shape (B, K)
        phi: azimuthal angle [rad] of each hit, shape (B, K)
        w:   non-negative energy weights, shape (B, K)
        eps: small regulariser for zero-weight cases

    Returns:
        eta_mean:       (B,) energy-weighted mean η
        phi_mean:       (B,) energy-weighted mean φ in [−π, π]
        fit_successful: (B,) bool — True where sum(w) > eps
    """
    sum_w = w.sum(dim=-1)
    fit_successful = sum_w > eps
    sum_w_safe = sum_w.clamp_min(eps)

    eta_mean = (w * eta).sum(dim=-1) / sum_w_safe

    # Circular mean avoids discontinuity at ±π
    sin_mean = (w * torch.sin(phi)).sum(dim=-1) / sum_w_safe
    cos_mean = (w * torch.cos(phi)).sum(dim=-1) / sum_w_safe
    phi_mean = torch.atan2(sin_mean, cos_mean)

    return eta_mean, phi_mean, fit_successful
