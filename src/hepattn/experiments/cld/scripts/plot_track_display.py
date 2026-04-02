"""CLD single-track event display.

Layout: 3 rows × N_TRACKS columns (one charged particle per column).
  Row 0: x-y transverse view with helix curves
  Row 1: z-r longitudinal view with helix curves
  Row 2: parameter table (truth / Pandora / helix fit)

Each panel shows:
  • Silicon hits  — vtxd as "×", trkr as "+"
  • Truth helix   — red dashed
  • Pandora helix — blue solid   (best IoU match, if found)
  • Helix-fit     — green dotted (naive fit on truth hits)

Usage (from repo root)::

    python src/hepattn/experiments/cld/scripts/plot_track_display.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataset
from hepattn.experiments.cld.event_display import _build_helix_path
from hepattn.utils.helix import _fit_helices_flat, helix_params_to_track_params

plt.rcParams["figure.dpi"] = 300
plt.rcParams["text.usetex"] = True

# ── constants ──────────────────────────────────────────────────────────────
B_FIELD_T     = 2.0   # CLD solenoid [T]
HELIX_CLIP_M  = 1.5   # clip helix paths at this transverse radius [m]
HELIX_N_STEPS = 1024  # path points per helix curve
N_TRACKS      = 24    # separate figures to produce
IOI_MATCH_THRESH = 0.5
MIN_PT_GEV    = 0.5   # only show particles above this pT for readability
N_EVENTS_SCAN = 30    # events to scan when picking example tracks


# ── helpers ────────────────────────────────────────────────────────────────

def _cfg() -> dict:
    p = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"
    return yaml.safe_load(p.read_text())["data"]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _d0_from_vtx(vx: float, vy: float, phi: float) -> float:
    return vx * np.sin(phi) - vy * np.cos(phi)


def _helix_fit_on_hits(
    hx: np.ndarray, hy: np.ndarray, hz: np.ndarray, n_vtxd: int
) -> dict | None:
    """Naive helix fit — same logic as plot_track_residuals.py."""
    if hx.shape[0] < 3:
        return None
    hr = np.sqrt(hx ** 2 + hy ** 2)
    order = np.argsort(hr)
    is_vtxd = np.zeros(hx.shape[0], dtype=bool)
    is_vtxd[:n_vtxd] = True
    is_vtxd_sorted = is_vtxd[order]
    hx, hy, hz = hx[order], hy[order], hz[order]
    x = torch.tensor(hx, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(hy, dtype=torch.float32).unsqueeze(0)
    z = torch.tensor(hz, dtype=torch.float32).unsqueeze(0)
    w = torch.ones_like(x)
    zfit_w = torch.tensor(is_vtxd_sorted.astype(np.float32)).unsqueeze(0)
    R, phi0, eta, d0, z0, ok, cs = _fit_helices_flat(x, y, z, w, zfit_w=zfit_w)
    if not ok[0].item():
        return None
    pt, phi, eta_v, d0_mm, z0_mm = helix_params_to_track_params(
        R[0], phi0[0], eta[0], d0[0], z0[0], B_FIELD_T
    )
    return {
        "pt": float(pt), "phi": float(phi), "eta": float(eta_v),
        "d0_mm": float(d0_mm), "z0_mm": float(z0_mm),
        "charge_sign": float(cs[0]),
    }


def _match_pandora(
    p_vtxd: torch.Tensor, p_trkr: torch.Tensor,
    pan_vtxd: torch.Tensor, pan_trkr: torch.Tensor,
    pan_valid: torch.Tensor, pan_charged: torch.Tensor,
) -> tuple[float, int]:
    p_sihit   = torch.cat([p_vtxd, p_trkr]).float()
    pan_sihit = torch.cat([pan_vtxd, pan_trkr], dim=-1).float()
    inter = (pan_sihit * p_sihit).sum(-1)
    union = (pan_sihit + p_sihit - pan_sihit * p_sihit).sum(-1)
    iou   = inter / union.clamp_min(1e-6)
    iou[~(pan_valid & pan_charged)] = 0.0
    best_iou, best_idx = iou.max(dim=0)
    if best_iou.item() < IOI_MATCH_THRESH:
        return 0.0, -1
    return float(best_iou.item()), int(best_idx.item())


# ── track selection ─────────────────────────────────────────────────────────

TrackRecord = dict  # keys: inputs, targets, p_idx, pt_true


def _scan_tracks(cfg: dict, n_tracks: int, n_events: int) -> list[TrackRecord]:
    """Scan events and collect candidate charged-particle track records.

    Returns *n_tracks* records spread across the pT range of all found
    candidates.
    """
    dataset = CLDDataset(
        dirpath=cfg["test_dir"],
        num_samples=n_events,
        inputs=cfg["inputs"],
        targets=cfg["targets"],
        input_dtype=cfg.get("input_dtype", "float32"),
        target_dtype=cfg.get("target_dtype", "float32"),
        merge_inputs=cfg.get("merge_inputs", {}),
        particle_min_pt=cfg.get("particle_min_pt", 0.01),
        particle_max_abs_eta=cfg.get("particle_max_abs_eta", 4.0),
        include_classes=cfg.get("include_classes"),
        charged_particle_min_num_hits=cfg.get("charged_particle_min_num_hits", {}),
        charged_particle_max_num_hits=cfg.get("charged_particle_max_num_hits", {}),
        particle_cut_veto_min_num_hits=cfg.get("particle_cut_veto_min_num_hits", {}),
        particle_hit_deflection_cuts=cfg.get("particle_hit_deflection_cuts", {}),
        particle_hit_separation_cuts=cfg.get("particle_hit_separation_cuts", {}),
        particle_hit_min_p_ratio=cfg.get("particle_hit_min_p_ratio", {}),
        sampling_seed=cfg.get("sampling_seed", 42),
        fast_file_discovery=True,
        force_pad_sizes=None,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    candidates: list[TrackRecord] = []

    for inputs, targets in tqdm(loader, desc="Scanning events", unit="ev", total=n_events):
        part_valid   = targets["particle_valid"][0].bool()
        part_charged = targets["particle_is_charged"][0].bool()
        pt_all       = _np(targets["particle_mom.r"][0])

        for p_idx in range(int(part_valid.shape[0])):
            if not (part_valid[p_idx] and part_charged[p_idx]):
                continue
            pt = float(pt_all[p_idx])
            if pt < MIN_PT_GEV:
                continue
            p_vtxd = targets["particle_vtxd_valid"][0][p_idx]
            p_trkr = targets["particle_trkr_valid"][0][p_idx]
            if int(p_vtxd.sum()) < 3 or int(p_trkr.sum()) < 3:
                continue
            candidates.append({"inputs": inputs, "targets": targets, "p_idx": p_idx, "pt": pt})

        if len(candidates) >= n_tracks * 20:  # plenty to choose from
            break

    if not candidates:
        return []

    # Pick tracks spread evenly across the pT range
    candidates.sort(key=lambda r: r["pt"])
    if len(candidates) <= n_tracks:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, n_tracks, dtype=int)
    return [candidates[i] for i in picks]


# ── per-column panel drawing ─────────────────────────────────────────────────

def _draw_panel(
    ax_xy: plt.Axes,
    ax_zr: plt.Axes,
    rec: TrackRecord,
) -> None:
    inputs  = rec["inputs"]
    targets = rec["targets"]
    p_idx   = rec["p_idx"]

    # ── truth parameters ──────────────────────────────────────────────────
    pt_t   = float(_np(targets["particle_mom.r"][0])[p_idx])
    eta_t  = float(_np(targets["particle_mom.eta"][0])[p_idx])
    phi_t  = float(_np(targets["particle_mom.phi"][0])[p_idx])
    qopt_t = float(_np(targets["particle_mom.qopt"][0])[p_idx])
    vtx_x  = float(_np(targets["particle_vtx.x"][0])[p_idx]) * 1e-3  # mm → m
    vtx_y  = float(_np(targets["particle_vtx.y"][0])[p_idx]) * 1e-3  # mm → m
    vtx_z  = float(_np(targets["particle_vtx.z"][0])[p_idx]) * 1e-3  # mm → m
    charge_t = float(_np(targets["particle_charge"][0])[p_idx]) if "particle_charge" in targets else np.sign(qopt_t * pt_t)

    # Exact perigee d0/z0 from truth kinematics (valid for all particles).
    q_sign_t = np.sign(qopt_t)
    R_t = abs(pt_t) / (0.3 * B_FIELD_T)
    xc_t = vtx_x + q_sign_t * R_t * np.sin(phi_t)
    yc_t = vtx_y - q_sign_t * R_t * np.cos(phi_t)
    c_t  = np.sqrt(xc_t ** 2 + yc_t ** 2)
    denom_t = c_t + R_t if (c_t + R_t) > 1e-12 else 1.0
    d0_t_m = -q_sign_t * (c_t ** 2 - R_t ** 2) / denom_t
    vtx_r_t = np.sqrt(vtx_x ** 2 + vtx_y ** 2)
    z0_t_m = vtx_z - np.sinh(eta_t) * vtx_r_t

    # ── hit positions ─────────────────────────────────────────────────────
    p_vtxd = targets["particle_vtxd_valid"][0][p_idx]   # (N_vtxd,) bool
    p_trkr = targets["particle_trkr_valid"][0][p_idx]   # (N_trkr,) bool

    vtxd_x = _np(inputs["vtxd_pos.x"][0])
    vtxd_y = _np(inputs["vtxd_pos.y"][0])
    vtxd_z = _np(inputs["vtxd_pos.z"][0])
    trkr_x = _np(inputs["trkr_pos.x"][0])
    trkr_y = _np(inputs["trkr_pos.y"][0])
    trkr_z = _np(inputs["trkr_pos.z"][0])

    vtxd_sel = _np(p_vtxd).astype(bool)
    trkr_sel = _np(p_trkr).astype(bool)
    hx = np.concatenate([vtxd_x[vtxd_sel], trkr_x[trkr_sel]])
    hy = np.concatenate([vtxd_y[vtxd_sel], trkr_y[trkr_sel]])
    hz = np.concatenate([vtxd_z[vtxd_sel], trkr_z[trkr_sel]])
    hr = np.sqrt(hx ** 2 + hy ** 2)

    n_vtxd = int(p_vtxd.sum().item())
    n_trkr = int(p_trkr.sum().item())

    # ── draw hits ─────────────────────────────────────────────────────────
    if vtxd_sel.any():
        ax_xy.scatter(vtxd_x[vtxd_sel], vtxd_y[vtxd_sel],
                      marker="x", color="black", s=18, linewidths=0.8, zorder=5)
        ax_zr.scatter(vtxd_z[vtxd_sel],
                      np.sqrt(vtxd_x[vtxd_sel]**2 + vtxd_y[vtxd_sel]**2),
                      marker="x", color="black", s=18, linewidths=0.8, zorder=5)
    if trkr_sel.any():
        ax_xy.scatter(trkr_x[trkr_sel], trkr_y[trkr_sel],
                      marker="+", color="black", s=28, linewidths=0.8, zorder=5)
        ax_zr.scatter(trkr_z[trkr_sel],
                      np.sqrt(trkr_x[trkr_sel]**2 + trkr_y[trkr_sel]**2),
                      marker="+", color="black", s=28, linewidths=0.8, zorder=5)

    # ── truth helix ───────────────────────────────────────────────────────
    cs_t = torch.tensor(np.sign(charge_t) if charge_t != 0 else 1.0, dtype=torch.float32)
    xt, yt, zt, _ = _build_helix_path(
        torch.tensor(phi_t), torch.tensor(eta_t),
        torch.tensor(max(abs(pt_t), 1e-6)), cs_t,
        torch.tensor(d0_t_m), torch.tensor(z0_t_m),
        B_FIELD_T, HELIX_CLIP_M, n_steps=HELIX_N_STEPS,
    )
    ax_xy.plot(_np(xt), _np(yt), color="tab:red", lw=1.5, ls="--", zorder=4)
    ax_zr.plot(_np(zt), np.sqrt(_np(xt)**2 + _np(yt)**2),
               color="tab:red", lw=1.5, ls="--", zorder=4)

    # ── Pandora match ─────────────────────────────────────────────────────
    pan_valid   = targets["pandora_valid"][0].bool()
    pan_charged = targets["pandora_is_charged"][0].bool()
    pan_idx = _match_pandora(
        p_vtxd, p_trkr,
        targets["pandora_vtxd_valid"][0], targets["pandora_trkr_valid"][0],
        pan_valid, pan_charged,
    )[1]

    theta_t = 2.0 * np.arctan(np.exp(-eta_t))

    pt_pan = eta_pan = theta_pan = phi_pan = d0_pan_mm = z0_pan_mm = qopt_pan = None
    if pan_idx >= 0:
        pt_pan  = float(_np(targets["pandora_mom.r"][0])[pan_idx])
        eta_pan = float(_np(targets["pandora_mom.eta"][0])[pan_idx])
        mx_pan  = float(_np(targets["pandora_mom.x"][0])[pan_idx])
        my_pan  = float(_np(targets["pandora_mom.y"][0])[pan_idx])
        phi_pan = float(np.arctan2(my_pan, mx_pan))
        ch_pan  = float(_np(targets["pandora_charge"][0])[pan_idx])
        qopt_pan = ch_pan / max(abs(pt_pan), 1e-6)
        ref_x = float(_np(targets["pandora_ref.x"][0])[pan_idx]) * 1e-3  # mm → m
        ref_y = float(_np(targets["pandora_ref.y"][0])[pan_idx]) * 1e-3  # mm → m
        ref_z = float(_np(targets["pandora_ref.z"][0])[pan_idx]) * 1e-3  # mm → m
        ref_r_pan = np.sqrt(ref_x ** 2 + ref_y ** 2)
        # d0: circle geometry from reference point (works wherever ref is on the helix)
        q_sign_pan = np.sign(ch_pan) if ch_pan != 0 else 1.0
        R_pan = abs(pt_pan) / (0.3 * B_FIELD_T)
        xc_pan = ref_x + q_sign_pan * R_pan * np.sin(phi_pan)
        yc_pan = ref_y - q_sign_pan * R_pan * np.cos(phi_pan)
        c_pan  = np.sqrt(xc_pan ** 2 + yc_pan ** 2)
        denom_pan = c_pan + R_pan if (c_pan + R_pan) > 1e-12 else 1.0
        d0_pan_m = -q_sign_pan * (c_pan ** 2 - R_pan ** 2) / denom_pan
        # z0: linear extrapolation to r=0 (valid for any point on the helix)
        z0_pan_m = ref_z - np.sinh(eta_pan) * ref_r_pan
        d0_pan_mm = d0_pan_m * 1e3
        z0_pan_mm = z0_pan_m * 1e3
        theta_pan = 2.0 * np.arctan(np.exp(-eta_pan))

        cs_pan = torch.tensor(np.sign(ch_pan) if ch_pan != 0 else 1.0, dtype=torch.float32)
        xp, yp, zp, _ = _build_helix_path(
            torch.tensor(phi_pan), torch.tensor(eta_pan),
            torch.tensor(max(abs(pt_pan), 1e-6)), cs_pan,
            torch.tensor(d0_pan_m), torch.tensor(z0_pan_m),
            B_FIELD_T, HELIX_CLIP_M,
        )
        ax_xy.plot(_np(xp), _np(yp), color="tab:blue", lw=1.5, ls="-", zorder=3)
        ax_zr.plot(_np(zp), np.sqrt(_np(xp)**2 + _np(yp)**2),
                   color="tab:blue", lw=1.5, ls="-", zorder=3)

    # ── helix fit ─────────────────────────────────────────────────────────
    hfit = _helix_fit_on_hits(hx, hy, hz, n_vtxd)
    pt_hf = eta_hf = theta_hf = phi_hf = d0_hf_mm = z0_hf_mm = qopt_hf = None
    if hfit is not None:
        pt_hf   = hfit["pt"]
        eta_hf  = hfit["eta"]
        theta_hf = 2.0 * np.arctan(np.exp(-eta_hf))
        phi_hf  = hfit["phi"]
        d0_hf_mm = hfit["d0_mm"]
        z0_hf_mm = hfit["z0_mm"]
        qopt_hf = hfit["charge_sign"] / max(abs(pt_hf), 1e-6)
        cs_hf   = torch.tensor(hfit["charge_sign"], dtype=torch.float32)
        xh, yh, zh, _ = _build_helix_path(
            torch.tensor(phi_hf), torch.tensor(eta_hf),
            torch.tensor(max(abs(pt_hf), 1e-6)), cs_hf,
            torch.tensor(d0_hf_mm * 1e-3), torch.tensor(z0_hf_mm * 1e-3),
            B_FIELD_T, HELIX_CLIP_M,
        )
        ax_xy.plot(_np(xh), _np(yh), color="tab:green", lw=1.5, ls=":", zorder=3)
        ax_zr.plot(_np(zh), np.sqrt(_np(xh)**2 + _np(yh)**2),
                   color="tab:green", lw=1.5, ls=":", zorder=3)

    # ── compact parameter annotations ─────────────────────────────────────
    def _f(v, fmt=".3f"):
        return format(v, fmt) if v is not None else "---"

    d0_t_mm = d0_t_m * 1e3
    z0_t_mm = z0_t_m * 1e3

    _hdr = r"\textbf{T / P / H}"
    ax_xy.text(
        0.03, 0.04,
        _hdr + "\n"
        + rf"$p_T$ [GeV]: {_f(pt_t)} / {_f(pt_pan)} / {_f(pt_hf)}" + "\n"
        + rf"$\phi$ [rad]: {_f(phi_t,'+.3f')} / {_f(phi_pan,'+.3f')} / {_f(phi_hf,'+.3f')}" + "\n"
        + rf"$d_0$ [mm]: {_f(d0_t_mm,'+.2f')} / {_f(d0_pan_mm,'+.2f')} / {_f(d0_hf_mm,'+.2f')}",
        transform=ax_xy.transAxes, fontsize=5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, lw=0.4),
    )
    ax_zr.text(
        0.97, 0.04,
        _hdr + "\n"
        + rf"$\theta$ [rad]: {_f(theta_t,'.3f')} / {_f(theta_pan,'.3f')} / {_f(theta_hf,'.3f')}" + "\n"
        + rf"$q/p_T$: {_f(qopt_t,'+.3f')} / {_f(qopt_pan,'+.3f')} / {_f(qopt_hf,'+.3f')}" + "\n"
        + rf"$z_0$ [mm]: {_f(z0_t_mm,'+.2f')} / {_f(z0_pan_mm,'+.2f')} / {_f(z0_hf_mm,'+.2f')}" + "\n"
        + rf"$N_\mathrm{{si}}$: {n_vtxd + n_trkr}",
        transform=ax_zr.transAxes, fontsize=5,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, lw=0.4),
    )

    # ── cosmetics ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color="tab:red",   lw=1.5, ls="--", label="Truth"),
    ]
    if pan_idx >= 0:
        legend_handles.append(
            Line2D([0], [0], color="tab:blue", lw=1.5, ls="-",  label="Pandora"))
    if hfit is not None:
        legend_handles.append(
            Line2D([0], [0], color="tab:green", lw=1.5, ls=":", label="Helix fit"))
    ax_xy.legend(handles=legend_handles, fontsize=5.0, framealpha=0.8,
                 loc="upper left", handlelength=1.8)

    # Tracker boundary circle
    theta_c = np.linspace(0, 2 * np.pi, 256)
    ax_xy.plot(HELIX_CLIP_M * np.cos(theta_c), HELIX_CLIP_M * np.sin(theta_c),
               color="gray", lw=0.5, ls=":", zorder=0)
    ax_xy.set_aspect("equal")

    ax_xy.set_xlabel("$x$ [m]", fontsize=7)
    ax_xy.set_ylabel("$y$ [m]", fontsize=7)
    ax_xy.tick_params(labelsize=6)
    ax_xy.grid(True, alpha=0.2, lw=0.4)

    ax_zr.set_xlabel("$z$ [m]", fontsize=7)
    ax_zr.set_ylabel("$r$ [m]", fontsize=7)
    ax_zr.set_ylim(bottom=0)
    ax_zr.tick_params(labelsize=6)
    ax_zr.grid(True, alpha=0.2, lw=0.4)


def _draw_perigee_panel(
    ax: plt.Axes,
    rec: TrackRecord,
    col: int,
) -> None:
    """x-y view rotated/shifted so the fit perigee is the origin and fit phi=0."""
    inputs  = rec["inputs"]
    targets = rec["targets"]
    p_idx   = rec["p_idx"]

    pt_t   = float(_np(targets["particle_mom.r"][0])[p_idx])
    eta_t  = float(_np(targets["particle_mom.eta"][0])[p_idx])
    phi_t  = float(_np(targets["particle_mom.phi"][0])[p_idx])
    qopt_t = float(_np(targets["particle_mom.qopt"][0])[p_idx])
    vtx_x  = float(_np(targets["particle_vtx.x"][0])[p_idx]) * 1e-3
    vtx_y  = float(_np(targets["particle_vtx.y"][0])[p_idx]) * 1e-3
    vtx_z  = float(_np(targets["particle_vtx.z"][0])[p_idx]) * 1e-3
    q_sign_t = np.sign(qopt_t)

    p_vtxd   = targets["particle_vtxd_valid"][0][p_idx]
    p_trkr   = targets["particle_trkr_valid"][0][p_idx]
    vtxd_x   = _np(inputs["vtxd_pos.x"][0]); vtxd_y = _np(inputs["vtxd_pos.y"][0]); vtxd_z = _np(inputs["vtxd_pos.z"][0])
    trkr_x   = _np(inputs["trkr_pos.x"][0]); trkr_y = _np(inputs["trkr_pos.y"][0]); trkr_z = _np(inputs["trkr_pos.z"][0])
    vtxd_sel = _np(p_vtxd).astype(bool)
    trkr_sel = _np(p_trkr).astype(bool)
    hx = np.concatenate([vtxd_x[vtxd_sel], trkr_x[trkr_sel]])
    hy = np.concatenate([vtxd_y[vtxd_sel], trkr_y[trkr_sel]])
    hz = np.concatenate([vtxd_z[vtxd_sel], trkr_z[trkr_sel]])
    n_vtxd = int(p_vtxd.sum().item())

    hfit = _helix_fit_on_hits(hx, hy, hz, n_vtxd)
    if hfit is not None:
        ref_phi  = hfit["phi"]
        ref_eta  = hfit["eta"]
        ref_pt   = max(abs(hfit["pt"]), 1e-6)
        ref_d0_m = hfit["d0_mm"] * 1e-3
        ref_z0_m = hfit["z0_mm"] * 1e-3
        ref_cs   = float(hfit["charge_sign"])
        frame_label = "Helix fit"
    else:
        R_t = abs(pt_t) / (0.3 * B_FIELD_T)
        xc_t = vtx_x + q_sign_t * R_t * np.sin(phi_t)
        yc_t = vtx_y - q_sign_t * R_t * np.cos(phi_t)
        c_t  = np.sqrt(xc_t**2 + yc_t**2)
        ref_d0_m = -q_sign_t * (c_t**2 - R_t**2) / (c_t + R_t) if (c_t + R_t) > 1e-12 else 0.0
        vtx_r_t  = np.sqrt(vtx_x**2 + vtx_y**2)
        ref_z0_m = vtx_z - np.sinh(eta_t) * vtx_r_t
        ref_phi, ref_eta = phi_t, eta_t
        ref_pt   = max(abs(pt_t), 1e-6)
        ref_cs   = float(q_sign_t)
        frame_label = "Truth (fit failed)"

    peri_x = -ref_d0_m * np.sin(ref_phi)
    peri_y =  ref_d0_m * np.cos(ref_phi)

    def _tr2d(x, y):
        dx = np.asarray(x, float) - peri_x
        dy = np.asarray(y, float) - peri_y
        cp, sp = np.cos(ref_phi), np.sin(ref_phi)
        return dx * cp + dy * sp, -dx * sp + dy * cp

    # Particle hits
    if vtxd_sel.any():
        rx, ry = _tr2d(vtxd_x[vtxd_sel], vtxd_y[vtxd_sel])
        ax.scatter(rx, ry, marker="x", color="black", s=18, linewidths=0.8, zorder=5)
    if trkr_sel.any():
        rx, ry = _tr2d(trkr_x[trkr_sel], trkr_y[trkr_sel])
        ax.scatter(rx, ry, marker="+", color="black", s=28, linewidths=0.8, zorder=5)

    # Helix fit
    cs_ref = torch.tensor(ref_cs, dtype=torch.float32)
    xref, yref, zref, _ = _build_helix_path(
        torch.tensor(ref_phi, dtype=torch.float32),
        torch.tensor(ref_eta,  dtype=torch.float32),
        torch.tensor(ref_pt,   dtype=torch.float32),
        cs_ref,
        torch.tensor(ref_d0_m, dtype=torch.float32),
        torch.tensor(ref_z0_m, dtype=torch.float32),
        B_FIELD_T, HELIX_CLIP_M, n_steps=HELIX_N_STEPS,
    )
    rx, ry = _tr2d(_np(xref), _np(yref))
    ax.plot(rx, ry, color="tab:green", lw=1.5, ls=":", zorder=3, label=frame_label)

    # Truth helix
    R_t  = abs(pt_t) / (0.3 * B_FIELD_T)
    xc_t = vtx_x + q_sign_t * R_t * np.sin(phi_t)
    yc_t = vtx_y - q_sign_t * R_t * np.cos(phi_t)
    c_t  = np.sqrt(xc_t**2 + yc_t**2)
    d0_t_m  = -q_sign_t * (c_t**2 - R_t**2) / (c_t + R_t) if (c_t + R_t) > 1e-12 else 0.0
    vtx_r_t = np.sqrt(vtx_x**2 + vtx_y**2)
    z0_t_m  = vtx_z - np.sinh(eta_t) * vtx_r_t
    cs_t = torch.tensor(float(q_sign_t) if q_sign_t != 0 else 1.0, dtype=torch.float32)
    xt, yt, zt, _ = _build_helix_path(
        torch.tensor(phi_t, dtype=torch.float32),
        torch.tensor(eta_t,  dtype=torch.float32),
        torch.tensor(max(abs(pt_t), 1e-6), dtype=torch.float32),
        cs_t,
        torch.tensor(d0_t_m, dtype=torch.float32),
        torch.tensor(z0_t_m, dtype=torch.float32),
        B_FIELD_T, HELIX_CLIP_M, n_steps=HELIX_N_STEPS,
    )
    rx, ry = _tr2d(_np(xt), _np(yt))
    ax.plot(rx, ry, color="tab:red", lw=1.5, ls="--", zorder=4, label="Truth")

    ax.scatter([0], [0], color="tab:green", marker="*", s=60,
               edgecolors="black", linewidths=0.5, zorder=6)
    ax.axhline(0, color="dimgray", lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color="dimgray", lw=0.5, ls=":", zorder=0)
    ax.set_aspect("equal")
    ax.legend(fontsize=5, framealpha=0.8, loc="upper left")
    ax.set_xlabel(r"$x'$ (along fit, transverse) [m]", fontsize=7)
    ax.set_ylabel(r"$y'$ ($d_0$ dir.) [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, lw=0.4)


def _draw_perigee_zr_panel(
    ax: plt.Axes,
    rec: TrackRecord,
    col: int,
) -> None:
    """z-r view in perigee frame: origin shifted to (z0_fit, |d0_fit|).

    At the perigee dr/ds = 0 by definition (minimum transverse radius),
    so the track naturally starts moving along Δz — no rotation needed.
    """
    inputs  = rec["inputs"]
    targets = rec["targets"]
    p_idx   = rec["p_idx"]

    pt_t   = float(_np(targets["particle_mom.r"][0])[p_idx])
    eta_t  = float(_np(targets["particle_mom.eta"][0])[p_idx])
    phi_t  = float(_np(targets["particle_mom.phi"][0])[p_idx])
    qopt_t = float(_np(targets["particle_mom.qopt"][0])[p_idx])
    vtx_x  = float(_np(targets["particle_vtx.x"][0])[p_idx]) * 1e-3
    vtx_y  = float(_np(targets["particle_vtx.y"][0])[p_idx]) * 1e-3
    vtx_z  = float(_np(targets["particle_vtx.z"][0])[p_idx]) * 1e-3
    q_sign_t = np.sign(qopt_t)

    p_vtxd   = targets["particle_vtxd_valid"][0][p_idx]
    p_trkr   = targets["particle_trkr_valid"][0][p_idx]
    vtxd_x   = _np(inputs["vtxd_pos.x"][0]); vtxd_y = _np(inputs["vtxd_pos.y"][0]); vtxd_z = _np(inputs["vtxd_pos.z"][0])
    trkr_x   = _np(inputs["trkr_pos.x"][0]); trkr_y = _np(inputs["trkr_pos.y"][0]); trkr_z = _np(inputs["trkr_pos.z"][0])
    vtxd_sel = _np(p_vtxd).astype(bool)
    trkr_sel = _np(p_trkr).astype(bool)
    hx = np.concatenate([vtxd_x[vtxd_sel], trkr_x[trkr_sel]])
    hy = np.concatenate([vtxd_y[vtxd_sel], trkr_y[trkr_sel]])
    hz = np.concatenate([vtxd_z[vtxd_sel], trkr_z[trkr_sel]])
    n_vtxd = int(p_vtxd.sum().item())

    hfit = _helix_fit_on_hits(hx, hy, hz, n_vtxd)
    if hfit is not None:
        ref_phi  = hfit["phi"]
        ref_eta  = hfit["eta"]
        ref_pt   = max(abs(hfit["pt"]), 1e-6)
        ref_d0_m = hfit["d0_mm"] * 1e-3
        ref_z0_m = hfit["z0_mm"] * 1e-3
        ref_cs   = float(hfit["charge_sign"])
        frame_label = "Helix fit"
    else:
        R_t = abs(pt_t) / (0.3 * B_FIELD_T)
        xc_t = vtx_x + q_sign_t * R_t * np.sin(phi_t)
        yc_t = vtx_y - q_sign_t * R_t * np.cos(phi_t)
        c_t  = np.sqrt(xc_t**2 + yc_t**2)
        ref_d0_m = -q_sign_t * (c_t**2 - R_t**2) / (c_t + R_t) if (c_t + R_t) > 1e-12 else 0.0
        vtx_r_t  = np.sqrt(vtx_x**2 + vtx_y**2)
        ref_z0_m = vtx_z - np.sinh(eta_t) * vtx_r_t
        ref_phi, ref_eta = phi_t, eta_t
        ref_pt   = max(abs(pt_t), 1e-6)
        ref_cs   = float(q_sign_t)
        frame_label = "Truth (fit failed)"

    peri_r = abs(ref_d0_m)
    peri_z = ref_z0_m

    eta_sign = 1.0 if ref_eta >= 0.0 else -1.0

    def _tr_zr(x, y, z):
        r = np.sqrt(np.asarray(x, float)**2 + np.asarray(y, float)**2)
        return (np.asarray(z, float) - peri_z) * eta_sign, r - peri_r

    # Particle hits
    if vtxd_sel.any():
        dz, dr = _tr_zr(vtxd_x[vtxd_sel], vtxd_y[vtxd_sel], vtxd_z[vtxd_sel])
        ax.scatter(dz, dr, marker="x", color="black", s=18, linewidths=0.8, zorder=5)
    if trkr_sel.any():
        dz, dr = _tr_zr(trkr_x[trkr_sel], trkr_y[trkr_sel], trkr_z[trkr_sel])
        ax.scatter(dz, dr, marker="+", color="black", s=28, linewidths=0.8, zorder=5)

    # Helix fit path
    cs_ref = torch.tensor(ref_cs, dtype=torch.float32)
    xref, yref, zref, _ = _build_helix_path(
        torch.tensor(ref_phi, dtype=torch.float32),
        torch.tensor(ref_eta,  dtype=torch.float32),
        torch.tensor(ref_pt,   dtype=torch.float32),
        cs_ref,
        torch.tensor(ref_d0_m, dtype=torch.float32),
        torch.tensor(ref_z0_m, dtype=torch.float32),
        B_FIELD_T, HELIX_CLIP_M, n_steps=HELIX_N_STEPS,
    )
    dz, dr = _tr_zr(_np(xref), _np(yref), _np(zref))
    ax.plot(dz, dr, color="tab:green", lw=1.5, ls=":", zorder=3, label=frame_label)

    # Truth helix path
    R_t  = abs(pt_t) / (0.3 * B_FIELD_T)
    xc_t = vtx_x + q_sign_t * R_t * np.sin(phi_t)
    yc_t = vtx_y - q_sign_t * R_t * np.cos(phi_t)
    c_t  = np.sqrt(xc_t**2 + yc_t**2)
    d0_t_m  = -q_sign_t * (c_t**2 - R_t**2) / (c_t + R_t) if (c_t + R_t) > 1e-12 else 0.0
    vtx_r_t = np.sqrt(vtx_x**2 + vtx_y**2)
    z0_t_m  = vtx_z - np.sinh(eta_t) * vtx_r_t
    cs_t = torch.tensor(float(q_sign_t) if q_sign_t != 0 else 1.0, dtype=torch.float32)
    xt, yt, zt, _ = _build_helix_path(
        torch.tensor(phi_t, dtype=torch.float32),
        torch.tensor(eta_t,  dtype=torch.float32),
        torch.tensor(max(abs(pt_t), 1e-6), dtype=torch.float32),
        cs_t,
        torch.tensor(d0_t_m, dtype=torch.float32),
        torch.tensor(z0_t_m, dtype=torch.float32),
        B_FIELD_T, HELIX_CLIP_M, n_steps=HELIX_N_STEPS,
    )
    dz, dr = _tr_zr(_np(xt), _np(yt), _np(zt))
    ax.plot(dz, dr, color="tab:red", lw=1.5, ls="--", zorder=4, label="Truth")

    ax.scatter([0], [0], color="tab:green", marker="*", s=60,
               edgecolors="black", linewidths=0.5, zorder=6)
    ax.axhline(0, color="dimgray", lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color="dimgray", lw=0.5, ls=":", zorder=0)
    ax.legend(fontsize=5, framealpha=0.8, loc="upper left")
    ax.set_xlabel(r"$\Delta z \cdot \mathrm{sign}(\pi/2 - \theta)$ [m]", fontsize=7)
    ax.set_ylabel(r"$\Delta r$ [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, lw=0.4)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = _cfg()
    tracks = _scan_tracks(cfg, n_tracks=N_TRACKS, n_events=N_EVENTS_SCAN)

    if not tracks:
        print("No candidate tracks found — check data path and config.")
        return

    out_dir = Path(__file__).resolve().parents[1] / "plots" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(tracks):
        pt = rec["pt"]

        fig, axes = plt.subplots(
            2, 2,
            figsize=(9, 8),
            gridspec_kw={"height_ratios": [1.2, 1]},
        )
        _draw_panel(axes[0, 0], axes[0, 1], rec)
        _draw_perigee_panel(axes[1, 0], rec, 0)
        _draw_perigee_zr_panel(axes[1, 1], rec, 0)

        fig.suptitle(
            rf"CLD track {i+1:02d}  ($p_T^\mathrm{{true}} = {pt:.3f}$ GeV)"
            r" | truth (red --), Pandora (blue —), helix fit (green $\cdots$)",
            fontsize=8,
        )
        fig.tight_layout()

        out_path = out_dir / f"track_{i+1:02d}_pt{pt:.3f}GeV.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i+1:02d}/{N_TRACKS}] Saved → {out_path}")

    print(f"\nAll {len(tracks)} plots saved in {out_dir}")


if __name__ == "__main__":
    main()
