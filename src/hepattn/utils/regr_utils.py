"""Pure-numeric helpers extracted from ObjectPropertyRegressionTask.

All functions are module-level (no self). Callers pass whatever trainable state /
config they hold. Keeps the task class focused on model / loss logic.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from torch import Tensor


def pchip_coeffs(log_centers_np: np.ndarray, values_np: np.ndarray) -> Tensor:
    """Compute PCHIP cubic Hermite spline coefficients for interpolation."""
    pchip = PchipInterpolator(log_centers_np, values_np)
    return torch.tensor(pchip.c, dtype=torch.float32)


def bin_lookup(seed_pt: Tensor, values: Tensor, bin_edges: Tensor) -> Tensor:
    """Hard-step per-bin lookup via torch.bucketize.

    Args:
        seed_pt: (B, N) lookup keys.
        values: (num_bins,) per-bin statistics.
        bin_edges: (num_bins + 1,) edges; use self.corr_bin_edges[1:-1] (open ends).

    Returns:
        (B, N) per-bin values for each track.
    """
    bin_idx = torch.bucketize(seed_pt, bin_edges)
    return values[bin_idx]


def bin_interp(
    seed_pt: Tensor,
    values: Tensor,
    log_centers: Tensor,
    *,
    mode: str,
    pchip_c: Tensor | None = None,
    bin_edges: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Per-bin value lookup with step / linear / cubic (PCHIP) interpolation.

    Args:
        seed_pt: (B, N) lookup keys (linear pT).
        values: (num_bins,) per-bin statistics.
        log_centers: (num_bins,) log(bin_center) for this param.
        mode: 'step' | 'linear' | 'cubic'.
        pchip_c: (4, num_bins-1) PCHIP coefficients (cubic mode).
        bin_edges: open-ended bin edges for step mode (required if mode='step').
        eps: numerical clamp.
    """
    if mode == "step":
        assert bin_edges is not None, "bin_edges required for step mode"
        return bin_lookup(seed_pt, values, bin_edges)

    log_pt = torch.log(seed_pt.clamp(min=eps))
    log_pt_clamped = log_pt.clamp(log_centers[0], log_centers[-1])

    if mode == "linear":
        idx_right = torch.searchsorted(log_centers, log_pt_clamped).clamp(max=len(log_centers) - 1)
        idx_left = (idx_right - 1).clamp(min=0)
        denom = (log_centers[idx_right] - log_centers[idx_left]).clamp(min=eps)
        t = ((log_pt_clamped - log_centers[idx_left]) / denom).clamp(0.0, 1.0)
        return values[idx_left] * (1 - t) + values[idx_right] * t

    if mode == "cubic":
        assert pchip_c is not None, "pchip_c required for cubic mode"
        idx = torch.searchsorted(log_centers[:-1], log_pt_clamped).clamp(1, len(log_centers) - 1) - 1
        dx = log_pt_clamped - log_centers[idx]
        c = pchip_c
        return c[0][idx] * dx**3 + c[1][idx] * dx**2 + c[2][idx] * dx + c[3][idx]

    msg = f"Unknown interp_mode: {mode}"
    raise ValueError(msg)


def compute_calo_direction_split(
    x: dict[str, Tensor],
    calo_masks: dict[str, Tensor],
    *,
    calo_constituents: list[str],
    calo_energy_field: str,
    input_object: str,  # unused but kept for call-site parity
    eps: float,
) -> dict[str, Tensor]:
    """Per-constituent calo centroid direction + shower shape features.

    Returns dict with per-constituent (eta, phi, E, mean_depth, depth_rms, trans_rms, max_hit_frac).
    """
    del input_object  # signature parity with combined variant
    result: dict[str, Tensor] = {}

    for constituent in calo_constituents:
        mask_calo = calo_masks[constituent]
        energy = x["inputs"][f"{constituent}_{calo_energy_field}"]
        pos_x = x["inputs"][f"{constituent}_pos.x"]
        pos_y = x["inputs"][f"{constituent}_pos.y"]
        pos_z = x["inputs"][f"{constituent}_pos.z"]

        w = mask_calo.float() * energy.unsqueeze(1)
        total_E = w.sum(-1)
        safe_E = total_E.clamp(min=eps)

        max_hit_E = w.max(-1).values
        max_hit_frac = max_hit_E / safe_E
        max_hit_frac = max_hit_frac.masked_fill(total_E < eps, 0.0)

        cx = (w * pos_x.unsqueeze(1)).sum(-1) / safe_E
        cy = (w * pos_y.unsqueeze(1)).sum(-1) / safe_E
        cz = (w * pos_z.unsqueeze(1)).sum(-1) / safe_E

        r_T_c = torch.sqrt(cx**2 + cy**2).clamp(min=eps)
        eta = torch.asinh(cz / r_T_c)
        phi = torch.atan2(cy, cx)

        no_hits = total_E < eps
        eta = eta.masked_fill(no_hits, 0.0)
        phi = phi.masked_fill(no_hits, 0.0)

        r_T_hits = torch.sqrt(pos_x**2 + pos_y**2)
        mean_depth = (w * r_T_hits.unsqueeze(1)).sum(-1) / safe_E
        depth_diff_sq = (r_T_hits.unsqueeze(1) - mean_depth.unsqueeze(-1)) ** 2
        depth_rms = torch.sqrt((w * depth_diff_sq).sum(-1) / safe_E)

        hit_eta = torch.asinh(pos_z / torch.sqrt(pos_x**2 + pos_y**2).clamp(min=eps))
        hit_phi = torch.atan2(pos_y, pos_x)
        d_eta = hit_eta.unsqueeze(1) - eta.unsqueeze(-1)
        d_phi = torch.atan2(
            torch.sin(hit_phi.unsqueeze(1) - phi.unsqueeze(-1)),
            torch.cos(hit_phi.unsqueeze(1) - phi.unsqueeze(-1)),
        )
        d_R_sq = d_eta**2 + d_phi**2
        trans_rms = torch.sqrt((w * d_R_sq).sum(-1) / safe_E)

        mean_depth = mean_depth.masked_fill(no_hits, 0.0)
        depth_rms = depth_rms.masked_fill(no_hits, 0.0)
        trans_rms = trans_rms.masked_fill(no_hits, 0.0)

        prefix = "ecal" if constituent.startswith("e") else "hcal"
        result[f"{prefix}_eta"] = eta
        result[f"{prefix}_phi"] = phi
        result[f"{prefix}_E"] = total_E
        result[f"{prefix}_mean_depth"] = mean_depth
        result[f"{prefix}_depth_rms"] = depth_rms
        result[f"{prefix}_trans_rms"] = trans_rms
        result[f"{prefix}_max_hit_frac"] = max_hit_frac

    return result


def compute_calo_direction(
    x: dict[str, Tensor],
    calo_masks: dict[str, Tensor],
    batch_size: int,
    num_queries: int,
    *,
    calo_constituents: list[str],
    calo_energy_field: str,
    input_object: str,
    eps: float,
) -> Tensor:
    """Energy-weighted calo centroid direction (eta, phi) across all calo constituents."""
    query_embed = x[input_object + "_embed"]
    weighted_x = query_embed.new_zeros(batch_size, num_queries)
    weighted_y = query_embed.new_zeros(batch_size, num_queries)
    weighted_z = query_embed.new_zeros(batch_size, num_queries)
    total_weight = query_embed.new_zeros(batch_size, num_queries)

    for constituent in calo_constituents:
        mask_calo = calo_masks[constituent]
        energy = x["inputs"][f"{constituent}_{calo_energy_field}"]
        pos_x = x["inputs"][f"{constituent}_pos.x"]
        pos_y = x["inputs"][f"{constituent}_pos.y"]
        pos_z = x["inputs"][f"{constituent}_pos.z"]

        w = mask_calo.float() * energy.unsqueeze(1)
        weighted_x += (w * pos_x.unsqueeze(1)).sum(-1)
        weighted_y += (w * pos_y.unsqueeze(1)).sum(-1)
        weighted_z += (w * pos_z.unsqueeze(1)).sum(-1)
        total_weight += w.sum(-1)

    safe_weight = total_weight.clamp(min=eps)
    cx = weighted_x / safe_weight
    cy = weighted_y / safe_weight
    cz = weighted_z / safe_weight

    cs = torch.sqrt(cx * cx + cy * cy + cz * cz + eps)
    ctheta = torch.acos((cz / cs).clamp(-1 + eps, 1 - eps))
    ceta = -torch.log(torch.tan(ctheta / 2).clamp(min=eps))
    ceta = ceta.clamp(-4.0, 4.0)
    cphi = torch.atan2(cy, cx)

    no_calo = total_weight < eps
    ceta = ceta.masked_fill(no_calo, 0.0)
    cphi = cphi.masked_fill(no_calo, 0.0)

    return torch.stack([ceta, cphi], dim=-1)
