"""Evaluate MaskFormer tracking predictions by fitting helices to predicted
hits and comparing pT resolution against truth and SiTrack baselines.

Produces four plots:
  1. Residual histograms (1/pT residuals, linear + log y)
  2. pT distributions (truth vs fit pT)
  3. Residual vs truth pT: IQR + median + mean
  4. Hit efficiency & purity vs truth pT with Bayesian binomial error bands
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from hepattn.utils.helix_fit import helix_seed_3pt

plt.rcParams["text.usetex"] = bool(shutil.which("latex"))
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """Unwrap angles along the last dimension."""
    two_pi = 2.0 * torch.pi
    dphi = phi[..., 1:] - phi[..., :-1]
    step = torch.where(
        dphi > torch.pi,
        -two_pi,
        torch.where(dphi < -torch.pi, two_pi, torch.zeros_like(dphi)),
    )
    offset = torch.cumsum(step, dim=-1)
    phi_unwrapped = phi.clone()
    phi_unwrapped[..., 1:] = phi_unwrapped[..., 1:] + offset
    return phi_unwrapped


def fit_helix(points_xyz: torch.Tensor):
    """Fit 3D helices using differentiable, closed-form least squares.

    Parameters
    ----------
    points_xyz : Tensor
        (N, 3) single helix or (B, N, 3) batch.

    Returns:
    -------
    dict with R, d0, z0, phi0, xc, yc, tan_lambda, alpha, beta, phi_dca.

    Raises:
        ValueError: If points_xyz does not have 2 or 3 dimensions.
    """
    squeeze_batch = False
    if points_xyz.ndim == 2:
        points_xyz = points_xyz.unsqueeze(0)
        squeeze_batch = True
    elif points_xyz.ndim != 3:
        raise ValueError("points_xyz must have shape (N, 3) or (B, N, 3)")

    pts = points_xyz.to(dtype=torch.float64)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]

    # Circle fit in (x, y)
    ones = torch.ones_like(x)
    A = torch.stack([x, y, ones], dim=-1)
    b = -(x**2 + y**2)
    sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution
    D, E, F = sol.squeeze(-1).unbind(-1)

    xc = -D / 2.0
    yc = -E / 2.0
    R = torch.sqrt((D**2 + E**2) / 4.0 - F)

    # Transverse parameters (d0, phi0)
    C = torch.stack([xc, yc], dim=-1)
    c = torch.linalg.norm(C, dim=-1)
    uC = C / c.unsqueeze(-1)
    x_dca = xc - R * uC[..., 0]
    y_dca = yc - R * uC[..., 1]
    d0 = c - R
    phi_dca_raw = torch.atan2(y_dca - yc, x_dca - xc)
    phi0 = torch.remainder(phi_dca_raw + 0.5 * torch.pi + torch.pi, 2.0 * torch.pi) - torch.pi

    # z(phi) linear fit with unwrapped phi
    phi_hits_raw = torch.atan2(y - yc.unsqueeze(-1), x - xc.unsqueeze(-1))
    phi_hits = unwrap_angle(phi_hits_raw)
    phi_mean = phi_hits.mean(dim=-1)
    z_mean = z.mean(dim=-1)
    cov = ((phi_hits - phi_mean.unsqueeze(-1)) * (z - z_mean.unsqueeze(-1))).mean(dim=-1)
    var_phi = ((phi_hits - phi_mean.unsqueeze(-1)) ** 2).mean(dim=-1)
    alpha = cov / var_phi
    beta = z_mean - alpha * phi_mean
    tan_lambda = alpha / R

    two_pi = 2.0 * torch.pi
    k = torch.round((phi_mean - phi_dca_raw) / two_pi)
    phi_dca_unwrapped = phi_dca_raw + two_pi * k
    z0 = alpha * phi_dca_unwrapped + beta

    out = {
        "R": R,
        "d0": d0,
        "z0": z0,
        "phi0": phi0,
        "xc": xc,
        "yc": yc,
        "tan_lambda": tan_lambda,
        "alpha": alpha,
        "beta": beta,
        "phi_dca": phi_dca_unwrapped,
    }
    if squeeze_batch:
        return {k: v[0] for k, v in out.items()}
    return out


def fit_circle_nl(
    hits_xyz: torch.Tensor,
    *,
    max_iters: int = 15,
    lm_lambda: float = 1e-3,
    eps: float = 1e-12,
):
    """Kasa algebraic circle fit + Levenberg-Marquardt geometric refinement.

    Returns (r_kasa, center_kasa_xy, r_refined, center_refined_xy).

    Raises:
        ValueError: If hits_xyz has wrong shape or fewer than 3 hits.
    """
    if hits_xyz.ndim != 2 or hits_xyz.shape[1] != 3:
        raise ValueError(f"hits_xyz must have shape (M, 3), got {tuple(hits_xyz.shape)}")
    if hits_xyz.shape[0] < 3:
        raise ValueError("Need at least 3 hits to fit a circle.")

    device = hits_xyz.device
    dtype = hits_xyz.dtype
    x = hits_xyz[:, 0]
    y = hits_xyz[:, 1]

    # Algebraic (Kasa) fit
    A = torch.stack([x, y, torch.ones_like(x)], dim=1)
    b = -(x * x + y * y)
    beta = torch.linalg.lstsq(A, b).solution
    D, E, F = beta[0], beta[1], beta[2]

    a0 = -0.5 * D
    c0 = -0.5 * E
    r2_0 = a0 * a0 + c0 * c0 - F
    r0 = torch.sqrt(torch.clamp(r2_0, min=eps))

    r_kasa = r0
    center_kasa_xy = torch.stack([a0, c0])

    # Geometric (LM) refinement
    a, c, r = a0, c0, r0
    lam = torch.as_tensor(lm_lambda, device=device, dtype=dtype)

    def sse(a_, c_, r_):
        di = torch.sqrt((x - a_) ** 2 + (y - c_) ** 2)
        e = di - r_
        return (e * e).sum()

    prev_sse = sse(a, c, r)

    for _ in range(max_iters):
        dx = x - a
        dy = y - c
        di = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=eps))
        e = di - r
        J = torch.stack([-dx / di, -dy / di, -torch.ones_like(di)], dim=1)
        H = J.T @ J
        g = J.T @ e
        H_damped = H + lam * torch.eye(3, device=device, dtype=dtype)
        delta = torch.linalg.solve(H_damped, -g)

        a_new = a + delta[0]
        c_new = c + delta[1]
        r_new = torch.clamp(r + delta[2], min=eps)
        new_sse = sse(a_new, c_new, r_new)

        if new_sse < prev_sse:
            a, c, r = a_new, c_new, r_new
            prev_sse = new_sse
            lam = torch.clamp(lam / 10.0, min=torch.as_tensor(1e-12, device=device, dtype=dtype))
            if torch.norm(delta) / (torch.norm(torch.stack([a, c, r])) + eps) < 1e-6:
                break
        else:
            lam = torch.clamp(lam * 10.0, max=torch.as_tensor(1e12, device=device, dtype=dtype))

    r_refined = r
    center_refined_xy = torch.stack([a, c])
    return r_kasa, center_kasa_xy, r_refined, center_refined_xy


# ---------------------------------------------------------------------------
# Coordinate matching
# ---------------------------------------------------------------------------


def count_xyz_matches(px, py, pz, tx, ty, tz):
    """Count how many predicted (px,py,pz) coords appear in truth (tx,ty,tz)."""
    mx = np.isin(px, tx)
    my = np.isin(py, ty)
    mz = np.isin(pz, tz)
    return int(np.sum(mx & my & mz))


# ---------------------------------------------------------------------------
# Binned statistics helpers
# ---------------------------------------------------------------------------


def per_bin_stats(truth_pt, r, edges, min_count=100):
    """Mean / median / IQR per log-spaced pT bin."""
    nb = len(edges) - 1
    idx = np.digitize(truth_pt, edges) - 1
    valid = (idx >= 0) & (idx < nb) & np.isfinite(r) & np.isfinite(truth_pt)

    mean = np.full(nb, np.nan)
    median = np.full(nb, np.nan)
    q1 = np.full(nb, np.nan)
    q3 = np.full(nb, np.nan)
    cnt = np.zeros(nb, dtype=int)

    for i in range(nb):
        m = valid & (idx == i)
        rr = r[m]
        cnt[i] = rr.size
        if rr.size > 0:
            mean[i] = np.mean(rr)
            median[i] = np.median(rr)
            q1[i] = np.quantile(rr, 0.25)
            q3[i] = np.quantile(rr, 0.75)

    iqr = q3 - q1
    keep = cnt >= min_count
    return mean, median, q1, q3, iqr, cnt, keep


def bayesian_binomial_error(k, n):
    """Posterior std for Beta(k+1, n-k+1)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) / (n + 2)) ** 2)


def binned_ratio_with_band(truth_pt, matched, denom, edges, min_count=100):
    """Ratio-of-sums in each pT bin with Bayesian binomial error band."""
    truth_pt = np.asarray(truth_pt, dtype=float)
    matched = np.asarray(matched, dtype=float)
    denom = np.asarray(denom, dtype=float)

    nbins = len(edges) - 1
    idx = np.digitize(truth_pt, edges) - 1
    valid = np.isfinite(truth_pt) & np.isfinite(matched) & np.isfinite(denom) & (idx >= 0) & (idx < nbins)

    sum_m = np.zeros(nbins, dtype=float)
    sum_d = np.zeros(nbins, dtype=float)
    n_tracks = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask = valid & (idx == i)
        n_tracks[i] = mask.sum()
        if n_tracks[i] > 0:
            sum_m[i] = matched[mask].sum()
            sum_d[i] = denom[mask].sum()

    ratio = np.divide(sum_m, sum_d, out=np.full(nbins, np.nan), where=(sum_d > 0))
    sig = np.full(nbins, np.nan)
    ok = (sum_d > 0) & (sum_m >= 0) & (sum_m <= sum_d)
    sig[ok] = bayesian_binomial_error(sum_m[ok], sum_d[ok])

    lo = np.clip(ratio - sig, 0.0, 1.0)
    hi = np.clip(ratio + sig, 0.0, 1.0)
    keep = (n_tracks >= min_count) & np.isfinite(ratio) & np.isfinite(lo) & np.isfinite(hi)
    return ratio, lo, hi, keep, n_tracks


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(eval_path: str, eff_cut: float, pur_cut: float, min_hits: int, max_fit_hits: int):
    """Iterate over all events in the H5 file and collect fit results.

    MF and SiTrack are evaluated independently — each loop selects its own
    set of truth particles, so their truth_pt arrays may differ in length.
    """
    eval_file = h5py.File(eval_path, "r")

    # Auto-detect optional data sources
    first_sid = next(iter(eval_file.keys()))
    has_sitrack = f"{first_sid}/targets/sitrack_particle_valid" in eval_file
    has_corrhead = f"{first_sid}/preds/final/flow_track_fit" in eval_file
    print(f"Auto-detected: SiTrack={has_sitrack}, CorrHead={has_corrhead}")

    # Accumulators — MF (independent of SiTrack)
    mf_truth_pt = []
    mf_kasa_pt = []
    mf_nl_pt = []
    mf_3pt_pt = []
    mf_num_truth_hits = []
    mf_num_pred_hits = []
    mf_num_matched_hits = []
    mf_eff = []
    mf_pur = []

    # Accumulators — CorrHead (optional, inside Loop 1)
    mf_corrhead_pt = []

    # Accumulators — SiTrack (independent of MF)
    sitrack_truth_pt = []
    sitrack_pt_list = []
    sitrack_eff = []
    sitrack_pur = []
    sitrack_num_truth_hits = []
    sitrack_num_sitrack_hits = []
    sitrack_num_matched_hits = []

    counter = 0

    for sample_id in eval_file:
        if counter % 100 == 0:
            print(f"Processing event {counter} ({sample_id})")

        # MF mask and truth
        valid_mask = eval_file[f"{sample_id}/targets/particle_valid"][:]
        pred_valid_mask = eval_file[f"{sample_id}/preds/final"]["flow_valid/flow_valid"][:]

        # Detector trkr hits
        trkr_x = eval_file[f"{sample_id}/inputs/trkr_pos.x"][:]
        trkr_y = eval_file[f"{sample_id}/inputs/trkr_pos.y"][:]
        trkr_z = eval_file[f"{sample_id}/inputs/trkr_pos.z"][:]
        trkr_t = eval_file[f"{sample_id}/inputs/trkr_time"][:]

        # Detector vtxd hits
        vtxd_x = eval_file[f"{sample_id}/inputs/vtxd_pos.x"][:]
        vtxd_y = eval_file[f"{sample_id}/inputs/vtxd_pos.y"][:]
        vtxd_z = eval_file[f"{sample_id}/inputs/vtxd_pos.z"][:]
        vtxd_t = eval_file[f"{sample_id}/inputs/vtxd_time"][:]

        # ---------------------------------------------------------------
        # Loop 1: MF predictions (no SiTrack filter)
        # ---------------------------------------------------------------
        mf_valid = valid_mask & pred_valid_mask

        mf_truth_trkr_mask = eval_file[f"{sample_id}/targets/particle_trkr_valid"][:][mf_valid]
        mf_pred_trkr_mask = eval_file[f"{sample_id}/preds/final/flow_trkr_assignment/flow_trkr_valid"][:][mf_valid]
        mf_truth_vtxd_mask = eval_file[f"{sample_id}/targets/particle_vtxd_valid"][:][mf_valid]
        mf_pred_vtxd_mask = eval_file[f"{sample_id}/preds/final/flow_vtxd_assignment/flow_vtxd_valid"][:][mf_valid]

        # CorrHead: read rinv for this sample if available
        if has_corrhead:
            corrhead_rinv = eval_file[f"{sample_id}/preds/final/flow_track_fit/flow_mom.rinv"][:][mf_valid]

        track_kasa_r, track_nl_r, track_3pt_r = [], [], []
        truth_pt_ev, truth_hits_ev, pred_hits_ev, matched_ev, eff_ev, pur_ev = [], [], [], [], [], []
        mf_corrhead_pt_ev = []
        truth_mom = eval_file[f"{sample_id}/targets/particle_mom.r"][mf_valid]

        for idx, (trkr_pred_mask, trkr_truth_mask, vtxd_pred_mask, vtxd_truth_mask) in enumerate(
            zip(mf_pred_trkr_mask, mf_truth_trkr_mask, mf_pred_vtxd_mask, mf_truth_vtxd_mask, strict=False)
        ):
            # Sort hits by time
            trkr_pred_time = np.argsort(trkr_t[0][trkr_pred_mask])
            trkr_truth_time = np.argsort(trkr_t[0][trkr_truth_mask])

            matched_pred_trkr_x = trkr_x[0][trkr_pred_mask][trkr_pred_time]
            matched_pred_trkr_y = trkr_y[0][trkr_pred_mask][trkr_pred_time]
            matched_pred_trkr_z = trkr_z[0][trkr_pred_mask][trkr_pred_time]
            matched_truth_trkr_x = trkr_x[0][trkr_truth_mask][trkr_truth_time]
            matched_truth_trkr_y = trkr_y[0][trkr_truth_mask][trkr_truth_time]
            matched_truth_trkr_z = trkr_z[0][trkr_truth_mask][trkr_truth_time]

            vtxd_pred_time = np.argsort(vtxd_t[0][vtxd_pred_mask])
            vtxd_truth_time = np.argsort(vtxd_t[0][vtxd_truth_mask])

            matched_pred_vtxd_x = vtxd_x[0][vtxd_pred_mask][vtxd_pred_time]
            matched_pred_vtxd_y = vtxd_y[0][vtxd_pred_mask][vtxd_pred_time]
            matched_pred_vtxd_z = vtxd_z[0][vtxd_pred_mask][vtxd_pred_time]
            matched_truth_vtxd_x = vtxd_x[0][vtxd_truth_mask][vtxd_truth_time]
            matched_truth_vtxd_y = vtxd_y[0][vtxd_truth_mask][vtxd_truth_time]
            matched_truth_vtxd_z = vtxd_z[0][vtxd_truth_mask][vtxd_truth_time]

            # Require enough combined hits
            if (len(matched_pred_trkr_x) + len(matched_pred_vtxd_x)) < min_hits:
                continue

            trkr_n_matched = count_xyz_matches(
                matched_pred_trkr_x,
                matched_pred_trkr_y,
                matched_pred_trkr_z,
                matched_truth_trkr_x,
                matched_truth_trkr_y,
                matched_truth_trkr_z,
            )
            vtxd_n_matched = count_xyz_matches(
                matched_pred_vtxd_x,
                matched_pred_vtxd_y,
                matched_pred_vtxd_z,
                matched_truth_vtxd_x,
                matched_truth_vtxd_y,
                matched_truth_vtxd_z,
            )

            n_truth_hit = len(matched_truth_trkr_x) + len(matched_truth_vtxd_x)
            n_pred_hit = len(matched_pred_trkr_x) + len(matched_pred_vtxd_x)
            n_matched = trkr_n_matched + vtxd_n_matched

            if n_truth_hit == 0:
                print(f"Matched truth track has zero hits | sample_id:{sample_id} | matched valid particle idx:{idx} | pred has hits:{n_pred_hit}")
                continue

            eff = n_matched / n_truth_hit
            pur = n_matched / n_pred_hit

            if eff < eff_cut or pur < pur_cut:
                continue

            # Circle fits on predicted hits
            x_all = np.concatenate([matched_pred_vtxd_x, matched_pred_trkr_x])
            y_all = np.concatenate([matched_pred_vtxd_y, matched_pred_trkr_y])
            z_all = np.concatenate([matched_pred_vtxd_z, matched_pred_trkr_z])

            xt = torch.tensor(x_all[:max_fit_hits])
            yt = torch.tensor(y_all[:max_fit_hits])
            zt = torch.tensor(z_all[:max_fit_hits])
            track = torch.stack([xt, yt, zt], dim=-1)

            # Kasa + NL circle fits
            try:
                r_kasa, _center_kasa, r_refined, _center_nl = fit_circle_nl(track)
            except torch.linalg.LinAlgError:
                continue
            track_kasa_r.append(r_kasa.item())
            track_nl_r.append(r_refined.item())

            # 3-point seed circle fit
            valid_3pt = torch.ones(xt.shape[0], dtype=torch.bool)
            seed_3pt = helix_seed_3pt(xt, yt, zt, valid_3pt, min_hits=3)
            track_3pt_r.append(seed_3pt["R"].item())

            truth_pt_ev.append(truth_mom[idx])
            truth_hits_ev.append(n_truth_hit)
            pred_hits_ev.append(n_pred_hit)
            matched_ev.append(n_matched)
            eff_ev.append(eff)
            pur_ev.append(pur)

            if has_corrhead:
                inv_pt_ch = corrhead_rinv[idx]
                mf_corrhead_pt_ev.append(1.0 / abs(inv_pt_ch) if abs(inv_pt_ch) > 1e-8 else np.nan)

        # Finalise MF pT arrays for this event
        track_kasa_r = np.array(track_kasa_r, dtype=object)
        kasa_pT = 0.3 * 2.0 * track_kasa_r
        track_nl_r = np.array(track_nl_r, dtype=object)
        nl_pT = 0.3 * 2.0 * track_nl_r
        track_3pt_r = np.array(track_3pt_r, dtype=object)
        three_pt_pT = 0.3 * 2.0 * track_3pt_r

        mf_truth_pt.extend(truth_pt_ev)
        mf_kasa_pt.extend(kasa_pT)
        mf_nl_pt.extend(nl_pT)
        mf_3pt_pt.extend(three_pt_pT)
        mf_num_truth_hits.extend(truth_hits_ev)
        mf_num_pred_hits.extend(pred_hits_ev)
        mf_num_matched_hits.extend(matched_ev)
        mf_eff.extend(eff_ev)
        mf_pur.extend(pur_ev)
        if has_corrhead:
            mf_corrhead_pt.extend(mf_corrhead_pt_ev)

        # ---------------------------------------------------------------
        # Loop 2: SiTrack (independent of MF)
        # ---------------------------------------------------------------
        if has_sitrack:
            sitrack_valid_mask = eval_file[f"{sample_id}/targets/sitrack_valid"][:]
            sitrack_particle_valid = eval_file[f"{sample_id}/targets/sitrack_particle_valid"][:]
            sitrack_trkr_mask_all = eval_file[f"{sample_id}/targets/sitrack_trkr_valid"][:]
            sitrack_vtxd_mask_all = eval_file[f"{sample_id}/targets/sitrack_vtxd_valid"][:]

            omega_all = eval_file[f"{sample_id}/targets/sitrack_state.omega"][:]
            omega0 = omega_all[0, :, 0]

            all_truth_trkr_mask = eval_file[f"{sample_id}/targets/particle_trkr_valid"][:]
            all_truth_vtxd_mask = eval_file[f"{sample_id}/targets/particle_vtxd_valid"][:]
            truth_mom_all = eval_file[f"{sample_id}/targets/particle_mom.r"][:]

            # For uniqueness: count how many valid sitracks link to each truth particle
            deg_truth = sitrack_particle_valid[0].T[:, sitrack_valid_mask[0]].sum(-1)

            for t_idx in range(sitrack_valid_mask.shape[1]):
                if not sitrack_valid_mask[0, t_idx]:
                    continue

                # Find linked truth particle(s)
                linked_particles = np.nonzero(sitrack_particle_valid[0, t_idx])[0]
                if linked_particles.size != 1:
                    continue
                p = int(linked_particles[0])

                # Skip if truth particle isn't valid or has multiple sitracks
                if not valid_mask[0, p]:
                    continue
                if deg_truth[p] != 1:
                    continue

                om = omega0[t_idx]
                if om == 0 or np.isnan(om):
                    continue
                pt_sitr = (0.0003 * 2.0) / np.abs(om)

                # Truth hit coords for this particle
                trkr_truth_mask = all_truth_trkr_mask[0, p].astype(bool)
                vtxd_truth_mask = all_truth_vtxd_mask[0, p].astype(bool)

                trkr_truth_time = np.argsort(trkr_t[0][trkr_truth_mask])
                truth_trkr_x = trkr_x[0][trkr_truth_mask][trkr_truth_time]
                truth_trkr_y = trkr_y[0][trkr_truth_mask][trkr_truth_time]
                truth_trkr_z = trkr_z[0][trkr_truth_mask][trkr_truth_time]

                vtxd_truth_time = np.argsort(vtxd_t[0][vtxd_truth_mask])
                truth_vtxd_x = vtxd_x[0][vtxd_truth_mask][vtxd_truth_time]
                truth_vtxd_y = vtxd_y[0][vtxd_truth_mask][vtxd_truth_time]
                truth_vtxd_z = vtxd_z[0][vtxd_truth_mask][vtxd_truth_time]

                n_truth_hit = len(truth_trkr_x) + len(truth_vtxd_x)
                if n_truth_hit == 0:
                    continue

                # Sitrack hit coords
                trkr_sitr_mask = sitrack_trkr_mask_all[0, t_idx].astype(bool)
                vtxd_sitr_mask = sitrack_vtxd_mask_all[0, t_idx].astype(bool)

                trkr_sitr_time = np.argsort(trkr_t[0][trkr_sitr_mask])
                trkr_sitr_x = trkr_x[0][trkr_sitr_mask][trkr_sitr_time]
                trkr_sitr_y = trkr_y[0][trkr_sitr_mask][trkr_sitr_time]
                trkr_sitr_z = trkr_z[0][trkr_sitr_mask][trkr_sitr_time]

                vtxd_sitr_time = np.argsort(vtxd_t[0][vtxd_sitr_mask])
                vtxd_sitr_x = vtxd_x[0][vtxd_sitr_mask][vtxd_sitr_time]
                vtxd_sitr_y = vtxd_y[0][vtxd_sitr_mask][vtxd_sitr_time]
                vtxd_sitr_z = vtxd_z[0][vtxd_sitr_mask][vtxd_sitr_time]

                trkr_sitr_match = count_xyz_matches(
                    trkr_sitr_x,
                    trkr_sitr_y,
                    trkr_sitr_z,
                    truth_trkr_x,
                    truth_trkr_y,
                    truth_trkr_z,
                )
                vtxd_sitr_match = count_xyz_matches(
                    vtxd_sitr_x,
                    vtxd_sitr_y,
                    vtxd_sitr_z,
                    truth_vtxd_x,
                    truth_vtxd_y,
                    truth_vtxd_z,
                )

                n_sitr_hit = len(trkr_sitr_x) + len(vtxd_sitr_x)
                n_sitr_match = trkr_sitr_match + vtxd_sitr_match

                if n_sitr_hit > 0:
                    sitrack_truth_pt.append(truth_mom_all[0, p])
                    sitrack_pt_list.append(pt_sitr)
                    sitrack_eff.append(n_sitr_match / n_truth_hit)
                    sitrack_pur.append(n_sitr_match / n_sitr_hit)
                    sitrack_num_truth_hits.append(n_truth_hit)
                    sitrack_num_sitrack_hits.append(n_sitr_hit)
                    sitrack_num_matched_hits.append(n_sitr_match)

        counter += 1

    eval_file.close()
    summary = f"Done — processed {counter} events, {len(mf_truth_pt)} MF tracks"
    if has_sitrack:
        summary += f", {len(sitrack_truth_pt)} SiTrack tracks"
    if has_corrhead:
        summary += f", {len(mf_corrhead_pt)} CorrHead tracks"
    print(summary)

    return {
        "mf_truth_pt": np.asarray(mf_truth_pt, dtype=float),
        "mf_kasa_pt": np.asarray(mf_kasa_pt, dtype=float),
        "mf_nl_pt": np.asarray(mf_nl_pt, dtype=float),
        "mf_3pt_pt": np.asarray(mf_3pt_pt, dtype=float),
        "mf_num_truth_hits": np.asarray(mf_num_truth_hits, dtype=float),
        "mf_num_pred_hits": np.asarray(mf_num_pred_hits, dtype=float),
        "mf_num_matched_hits": np.asarray(mf_num_matched_hits, dtype=float),
        "mf_eff": np.asarray(mf_eff, dtype=float),
        "mf_pur": np.asarray(mf_pur, dtype=float),
        "mf_corrhead_pt": np.asarray(mf_corrhead_pt, dtype=float),
        "has_sitrack": has_sitrack,
        "has_corrhead": has_corrhead,
        "sitrack_truth_pt": np.asarray(sitrack_truth_pt, dtype=float),
        "sitrack_pt": np.asarray(sitrack_pt_list, dtype=float),
        "sitrack_eff": np.asarray(sitrack_eff, dtype=float),
        "sitrack_pur": np.asarray(sitrack_pur, dtype=float),
        "sitrack_num_truth_hits": np.asarray(sitrack_num_truth_hits, dtype=float),
        "sitrack_num_sitrack_hits": np.asarray(sitrack_num_sitrack_hits, dtype=float),
        "sitrack_num_matched_hits": np.asarray(sitrack_num_matched_hits, dtype=float),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Shared binning
XMIN, XMAX, NBINS = 0.1, 100, 30


def _log_bins():
    edges = np.logspace(np.log10(XMIN), np.log10(XMAX), NBINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    xerr = 0.5 * widths
    return edges, centers, widths, xerr


# Colour palette
C_KASA = "blue"
C_NL = "#90CAF9"
C_3PT = "green"
C_ST = "orange"
C_CH = "#D55E00"  # Okabe-Ito vermilion for correction head


def plot_residual_histograms(d: dict, output_dir: Path, eff_cut: float, pur_cut: float):
    """Plot 1: 1/pT residual histograms (linear + log y)."""
    mf_truth = d["mf_truth_pt"]
    resid_kasa = 1.0 / d["mf_kasa_pt"] - 1.0 / mf_truth
    resid_nl = 1.0 / d["mf_nl_pt"] - 1.0 / mf_truth
    resid_3pt = 1.0 / d["mf_3pt_pt"] - 1.0 / mf_truth

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    for ax, nbins, yscale, ylabel in [(ax1, 50, "linear", "Counts"), (ax2, 32, "log", "Counts (log)")]:
        ax.hist(np.clip(resid_kasa, -1, 1), histtype="step", bins=nbins, range=(-1, 1), label="MF Kasa", color=C_KASA)
        ax.hist(np.clip(resid_nl, -1, 1), histtype="step", bins=nbins, range=(-1, 1), label="MF NL", color=C_NL)
        ax.hist(np.clip(resid_3pt, -1, 1), histtype="step", bins=nbins, range=(-1, 1), label="MF 3pt", color=C_3PT)
        if d["has_sitrack"]:
            st_truth = d["sitrack_truth_pt"]
            resid_st = 1.0 / d["sitrack_pt"] - 1.0 / st_truth
            ax.hist(np.clip(resid_st, -1, 1), histtype="step", bins=nbins, range=(-1, 1), label="SiTrack", color=C_ST)
        if d["has_corrhead"]:
            resid_ch = 1.0 / d["mf_corrhead_pt"] - 1.0 / mf_truth
            ax.hist(np.clip(resid_ch, -1, 1), histtype="step", bins=nbins, range=(-1, 1), label="CorrHead", color=C_CH)
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(r"$1/p_T^{reco} - 1/p_T^{truth}$ [GeV$^{-1}$]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Residual $1/p_T$ ({yscale})")
        ax.legend()

    fig.suptitle(rf"$1/p_T$ residual histograms (eff cut={eff_cut}, pur cut={pur_cut})")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_histograms.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved residual_histograms.png")


def plot_pt_distributions(d: dict, output_dir: Path, eff_cut: float, pur_cut: float):
    """Plot 2: pT distributions (truth vs fit pT)."""
    all_truth_pt = d["mf_truth_pt"]
    all_fit_pt = d["mf_kasa_pt"]
    all_fit_nl_pt = d["mf_nl_pt"]
    all_fit_3pt_pt = d["mf_3pt_pt"]

    xrng = (XMIN, XMAX)
    log_edges = np.logspace(np.log10(xrng[0]), np.log10(xrng[1]), NBINS + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    for ax, xlog, ylog in [(ax1, True, True), (ax2, True, False), (ax3, False, True)]:
        bins = log_edges if xlog else NBINS
        rng = None if xlog else xrng
        ax.hist(all_truth_pt, histtype="step", bins=bins, range=rng, label="truth")
        ax.hist(all_fit_pt, histtype="step", bins=bins, range=rng, label="MF Kasa", color=C_KASA)
        ax.hist(all_fit_nl_pt, histtype="step", bins=bins, range=rng, label="MF NL", color=C_NL)
        ax.hist(all_fit_3pt_pt, histtype="step", bins=bins, range=rng, label="MF 3pt", color=C_3PT)
        if d["has_corrhead"]:
            ax.hist(d["mf_corrhead_pt"], histtype="step", bins=bins, range=rng, label="CorrHead", color=C_CH)
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel("Counts")
        ax.legend()

    fig.suptitle(rf"$p_T$ distributions (eff cut={eff_cut}, pur cut={pur_cut})")
    fig.tight_layout()
    fig.savefig(output_dir / "pt_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved pt_distributions.png")


def _plot_residual_vs_pt_panel(
    d: dict,
    output_dir: Path,
    eff_cut: float,
    pur_cut: float,
    xmin: float,
    xmax: float,
    nbins: int,
    filename: str,
    title_suffix: str = "",
    min_count: int = 10,
):
    """Shared logic for residual-vs-pT plots with configurable range and binning."""
    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    xerr = 0.5 * widths

    mf_truth = d["mf_truth_pt"]
    resid = 1.0 / d["mf_kasa_pt"] - 1.0 / mf_truth
    nl_resid = 1.0 / d["mf_nl_pt"] - 1.0 / mf_truth
    resid_3pt = 1.0 / d["mf_3pt_pt"] - 1.0 / mf_truth

    k_mean, k_median, k_q1, _k_q3, k_iqr, _k_count, k_keep = per_bin_stats(mf_truth, resid, edges, min_count=min_count)
    nl_mean, nl_median, nl_q1, _nl_q3, nl_iqr, _nl_count, nl_keep = per_bin_stats(mf_truth, nl_resid, edges, min_count=min_count)
    s3_mean, s3_median, s3_q1, _s3_q3, s3_iqr, _s3_count, s3_keep = per_bin_stats(mf_truth, resid_3pt, edges, min_count=min_count)

    if d["has_sitrack"]:
        st_truth = d["sitrack_truth_pt"]
        st_resid = 1.0 / d["sitrack_pt"] - 1.0 / st_truth
        st_mean, st_median, st_q1, _st_q3, st_iqr, _st_count, st_keep = per_bin_stats(st_truth, st_resid, edges, min_count=min_count)

    if d["has_corrhead"]:
        ch_resid = 1.0 / d["mf_corrhead_pt"] - 1.0 / mf_truth
        ch_mean, ch_median, ch_q1, _ch_q3, ch_iqr, _ch_count, ch_keep = per_bin_stats(mf_truth, ch_resid, edges, min_count=min_count)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)

    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(r"$p_T^{truth}$ [GeV]")
        ax.axhline(0.0, linewidth=1, linestyle="--", alpha=0.7)

    # Left: IQR bands + medians
    axL.set_title(r"Residual vs truth $p_T$ (IQR + median)")
    axL.set_ylabel(r"$1/p_T^{reco} - 1/p_T^{truth}$ [GeV$^{-1}$]")

    axL.bar(
        centers[k_keep],
        k_iqr[k_keep],
        bottom=k_q1[k_keep],
        width=widths[k_keep] * 0.9935,
        align="center",
        alpha=0.20,
        linewidth=0,
        color=C_KASA,
        label="IQR (MF Kasa)",
    )
    axL.bar(
        centers[nl_keep],
        nl_iqr[nl_keep],
        bottom=nl_q1[nl_keep],
        width=widths[nl_keep] * 0.9935,
        align="center",
        alpha=0.20,
        linewidth=0,
        color=C_NL,
        label="IQR (MF NL)",
    )
    axL.bar(
        centers[s3_keep],
        s3_iqr[s3_keep],
        bottom=s3_q1[s3_keep],
        width=widths[s3_keep] * 0.9935,
        align="center",
        alpha=0.20,
        color=C_3PT,
        edgecolor=C_3PT,
        linewidth=1.0,
        linestyle=":",
        label="IQR (MF 3pt)",
    )
    if d["has_sitrack"]:
        axL.bar(
            centers[st_keep],
            st_iqr[st_keep],
            bottom=st_q1[st_keep],
            width=widths[st_keep] * 0.9935,
            align="center",
            alpha=0.20,
            linewidth=0,
            color=C_ST,
            label="IQR (SiTrack)",
        )
    if d["has_corrhead"]:
        axL.bar(
            centers[ch_keep],
            ch_iqr[ch_keep],
            bottom=ch_q1[ch_keep],
            width=widths[ch_keep] * 0.9935,
            align="center",
            alpha=0.20,
            linewidth=0,
            color=C_CH,
            label="IQR (CorrHead)",
        )

    axL.errorbar(
        centers[k_keep],
        k_median[k_keep],
        xerr=xerr[k_keep],
        fmt="none",
        capsize=0,
        elinewidth=2,
        color=C_KASA,
        ecolor=C_KASA,
        label="Median (MF Kasa)",
    )
    axL.errorbar(
        centers[nl_keep], nl_median[nl_keep], xerr=xerr[nl_keep], fmt="none", capsize=0, elinewidth=2, color=C_NL, ecolor=C_NL, label="Median (MF NL)"
    )
    axL.errorbar(
        centers[s3_keep],
        s3_median[s3_keep],
        xerr=xerr[s3_keep],
        fmt="none",
        capsize=0,
        elinewidth=2,
        color=C_3PT,
        ecolor=C_3PT,
        label="Median (MF 3pt)",
    )
    if d["has_sitrack"]:
        axL.errorbar(
            centers[st_keep],
            st_median[st_keep],
            xerr=xerr[st_keep],
            fmt="none",
            capsize=0,
            elinewidth=2,
            color=C_ST,
            ecolor=C_ST,
            label="Median (SiTrack)",
        )
    if d["has_corrhead"]:
        axL.errorbar(
            centers[ch_keep],
            ch_median[ch_keep],
            xerr=xerr[ch_keep],
            fmt="none",
            capsize=0,
            elinewidth=2,
            color=C_CH,
            ecolor=C_CH,
            label="Median (CorrHead)",
        )

    axL.legend(loc="upper right", frameon=True)

    # Right: Means
    axR.set_title(r"Residual vs truth $p_T$ (mean)")
    axR.set_ylabel(r"Mean of $1/p_T^{reco} - 1/p_T^{truth}$")

    axR.errorbar(
        centers[k_keep], k_mean[k_keep], xerr=xerr[k_keep], fmt="none", capsize=0, elinewidth=2, color=C_KASA, ecolor=C_KASA, label="Mean (MF Kasa)"
    )
    axR.errorbar(
        centers[nl_keep], nl_mean[nl_keep], xerr=xerr[nl_keep], fmt="none", capsize=0, elinewidth=2, color=C_NL, ecolor=C_NL, label="Mean (MF NL)"
    )
    axR.errorbar(
        centers[s3_keep], s3_mean[s3_keep], xerr=xerr[s3_keep], fmt="none", capsize=0, elinewidth=2, color=C_3PT, ecolor=C_3PT, label="Mean (MF 3pt)"
    )
    if d["has_sitrack"]:
        axR.errorbar(
            centers[st_keep],
            st_mean[st_keep],
            xerr=xerr[st_keep],
            fmt="none",
            capsize=0,
            elinewidth=2,
            color=C_ST,
            ecolor=C_ST,
            label="Mean (SiTrack)",
        )
    if d["has_corrhead"]:
        axR.errorbar(
            centers[ch_keep],
            ch_mean[ch_keep],
            xerr=xerr[ch_keep],
            fmt="none",
            capsize=0,
            elinewidth=2,
            color=C_CH,
            ecolor=C_CH,
            label="Mean (CorrHead)",
        )

    axR.legend(loc="upper right", frameon=True)

    suptitle = rf"$1/p_T$ residual vs truth $p_T$ (eff cut={eff_cut}, pur cut={pur_cut})"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_residual_vs_pt(d: dict, output_dir: Path, eff_cut: float, pur_cut: float):
    """Plot 3: Residual vs truth pT — IQR + median (left) and mean (right).

    Produces four variants: full range, low-pT, mid-pT, and high-pT zooms.
    """
    _plot_residual_vs_pt_panel(d, output_dir, eff_cut, pur_cut, xmin=XMIN, xmax=XMAX, nbins=NBINS, filename="residual_vs_pt.png")
    _plot_residual_vs_pt_panel(
        d, output_dir, eff_cut, pur_cut, xmin=0.1, xmax=1.0, nbins=10, filename="residual_vs_pt_low.png", title_suffix="low $p_T$"
    )
    _plot_residual_vs_pt_panel(
        d, output_dir, eff_cut, pur_cut, xmin=1.0, xmax=10.0, nbins=10, filename="residual_vs_pt_mid.png", title_suffix="mid $p_T$"
    )
    _plot_residual_vs_pt_panel(
        d, output_dir, eff_cut, pur_cut, xmin=10.0, xmax=100.0, nbins=10, filename="residual_vs_pt_high.png", title_suffix="high $p_T$"
    )


def plot_eff_pur_vs_pt(d: dict, output_dir: Path, eff_cut: float, pur_cut: float):
    """Plot 4: Hit efficiency and purity vs truth pT with Bayesian error bands."""
    edges, centers, bin_widths, xerr = _log_bins()
    min_count = 100
    mf_color = "tab:blue"
    st_color = "tab:orange"

    truth_mf = d["mf_truth_pt"]
    m_mf = d["mf_num_matched_hits"]
    t_mf = d["mf_num_truth_hits"]
    p_mf = d["mf_num_pred_hits"]

    eff_mf, eff_mf_lo, eff_mf_hi, keep_eff_mf, _ = binned_ratio_with_band(truth_mf, m_mf, t_mf, edges, min_count=min_count)
    pur_mf, pur_mf_lo, pur_mf_hi, keep_pur_mf, _ = binned_ratio_with_band(truth_mf, m_mf, p_mf, edges, min_count=min_count)

    if d["has_sitrack"]:
        truth_st = d["sitrack_truth_pt"]
        m_st = d["sitrack_num_matched_hits"]
        t_st = d["sitrack_num_truth_hits"]
        p_st = d["sitrack_num_sitrack_hits"]

        eff_st, eff_st_lo, eff_st_hi, keep_eff_st, _ = binned_ratio_with_band(truth_st, m_st, t_st, edges, min_count=min_count)
        pur_st, pur_st_lo, pur_st_hi, keep_pur_st, _ = binned_ratio_with_band(truth_st, m_st, p_st, edges, min_count=min_count)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlim(XMIN, XMAX)
        ax.set_xlabel(r"$p_T^{truth}$ [GeV]")

    # Efficiency
    ax1.set_title(r"Hit efficiency vs truth $p_T$")
    ax1.set_ylabel("Hit efficiency")

    ax1.bar(
        centers[keep_eff_mf],
        (eff_mf_hi - eff_mf_lo)[keep_eff_mf],
        bottom=eff_mf_lo[keep_eff_mf],
        width=bin_widths[keep_eff_mf] * 0.9935,
        align="center",
        alpha=0.18,
        linewidth=0,
        color=mf_color,
        label=r"MF band ($\mu \pm \sigma$)",
    )
    ax1.errorbar(
        centers[keep_eff_mf], eff_mf[keep_eff_mf], xerr=xerr[keep_eff_mf], fmt="none", elinewidth=2, color=mf_color, ecolor=mf_color, label="MF mean"
    )

    if d["has_sitrack"]:
        ax1.bar(
            centers[keep_eff_st],
            (eff_st_hi - eff_st_lo)[keep_eff_st],
            bottom=eff_st_lo[keep_eff_st],
            width=bin_widths[keep_eff_st] * 0.9935,
            align="center",
            alpha=0.18,
            linewidth=0,
            color=st_color,
            label=r"SiTrack band ($\mu \pm \sigma$)",
        )
        ax1.errorbar(
            centers[keep_eff_st],
            eff_st[keep_eff_st],
            xerr=xerr[keep_eff_st],
            fmt="none",
            elinewidth=2,
            color=st_color,
            ecolor=st_color,
            label="SiTrack mean",
        )
    ax1.legend(loc="best")

    # Purity
    ax2.set_title(r"Hit purity vs truth $p_T$")
    ax2.set_ylabel("Hit purity")

    ax2.bar(
        centers[keep_pur_mf],
        (pur_mf_hi - pur_mf_lo)[keep_pur_mf],
        bottom=pur_mf_lo[keep_pur_mf],
        width=bin_widths[keep_pur_mf] * 0.9935,
        align="center",
        alpha=0.18,
        linewidth=0,
        color=mf_color,
        label=r"MF band ($\mu \pm \sigma$)",
    )
    ax2.errorbar(
        centers[keep_pur_mf], pur_mf[keep_pur_mf], xerr=xerr[keep_pur_mf], fmt="none", elinewidth=2, color=mf_color, ecolor=mf_color, label="MF mean"
    )

    if d["has_sitrack"]:
        ax2.bar(
            centers[keep_pur_st],
            (pur_st_hi - pur_st_lo)[keep_pur_st],
            bottom=pur_st_lo[keep_pur_st],
            width=bin_widths[keep_pur_st] * 0.9935,
            align="center",
            alpha=0.18,
            linewidth=0,
            color=st_color,
            label=r"SiTrack band ($\mu \pm \sigma$)",
        )
        ax2.errorbar(
            centers[keep_pur_st],
            pur_st[keep_pur_st],
            xerr=xerr[keep_pur_st],
            fmt="none",
            elinewidth=2,
            color=st_color,
            ecolor=st_color,
            label="SiTrack mean",
        )
    ax2.legend(loc="best")

    fig.suptitle(rf"Hit efficiency \& purity vs truth $p_T$ (eff cut={eff_cut}, pur cut={pur_cut})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "eff_pur_vs_pt.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved eff_pur_vs_pt.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MF tracking predictions: helix fits + pT resolution plots.",
    )
    parser.add_argument("--eval-path", type=str, required=True, help="Path to the H5 eval file")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Directory for saved plots")
    parser.add_argument("--eff-cut", type=float, default=0.0, help="Efficiency threshold")
    parser.add_argument("--pur-cut", type=float, default=0.0, help="Purity threshold")
    parser.add_argument("--min-hits", type=int, default=6, help="Minimum combined hits for circle fit")
    parser.add_argument("--max-fit-hits", type=int, default=16, help="Truncation for circle fit input")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Eval file : {args.eval_path}")
    print(f"Output dir: {output_dir}")
    print(f"Eff cut={args.eff_cut}, Pur cut={args.pur_cut}, min_hits={args.min_hits}, max_fit_hits={args.max_fit_hits}")

    d = run_evaluation(args.eval_path, args.eff_cut, args.pur_cut, args.min_hits, args.max_fit_hits)

    plot_residual_histograms(d, output_dir, args.eff_cut, args.pur_cut)
    plot_pt_distributions(d, output_dir, args.eff_cut, args.pur_cut)
    plot_residual_vs_pt(d, output_dir, args.eff_cut, args.pur_cut)
    plot_eff_pur_vs_pt(d, output_dir, args.eff_cut, args.pur_cut)

    print("All plots saved.")


if __name__ == "__main__":
    main()
