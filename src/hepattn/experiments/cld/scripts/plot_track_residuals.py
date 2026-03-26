"""CLD track residual plots: Pandora vs helix fit vs truth particle.

For each charged truth particle the script:
  1. Runs a naive helix fit on the truth silicon hits (vtxd + trkr).
  2. Matches the best Pandora charged particle by silicon-hit IoU.
  3. Compares pT, q/pT, eta, phi, d0, z0 residuals for both baselines.

The B field for the CLD solenoid is set via B_FIELD_T (default 2 T).
Check src/hepattn/experiments/cld/task.py if you need to verify this value.

Usage (from repo root)::

    python src/hepattn/experiments/cld/scripts/plot_track_residuals.py

Output is written to src/hepattn/experiments/cld/plots/tracks/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataset
from hepattn.experiments.cld.plotting import (
    RESIDUAL_YLABEL,
    RESOLUTION_DIVIDER,
    RESOLUTION_YLABEL,
    RESOLUTION_VS_TRUTH_YLIM,
    TRUTH_BINS,
    make_arcsinh_fig,
    make_residual_fig,
    make_residual_fullrange_fig,
    make_residual_vs_truth_fig,
)
from hepattn.utils.helix import _fit_helices_flat, helix_params_to_track_params

plt.rcParams["figure.dpi"] = 300
plt.rcParams["text.usetex"] = True

# ── constants ──────────────────────────────────────────────────────────────
B_FIELD_T = 2.0          # CLD solenoid field [T] — matches task.py
IOI_MATCH_THRESH = 0.5   # minimum hit-IoU to accept a Pandora↔truth match
N_EVENTS = 1000           # events to process


# ── helpers ────────────────────────────────────────────────────────────────

def _cfg() -> dict:
    p = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"
    return yaml.safe_load(p.read_text())["data"]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _helix_fit_on_hits(
    hx: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    n_vtxd: int,
) -> dict | None:
    """Helix fit on raw silicon-hit positions.

    vtxd hits (first *n_vtxd* entries) are used for the z(r) fit; trkr hits
    are down-weighted in z because their z resolution is poorer.

    Returns a dict with keys pt, phi, eta, d0_mm, z0_mm, charge_sign,
    or None if the fit fails or there are fewer than 3 hits.
    """
    n = hx.shape[0]
    if n < 3:
        return None

    # Sort by transverse radius so unwrap sees a monotone sequence
    hr = np.sqrt(hx ** 2 + hy ** 2)
    order = np.argsort(hr)
    hx, hy, hz = hx[order], hy[order], hz[order]

    x = torch.tensor(hx, dtype=torch.float32).unsqueeze(0)  # (1, K)
    y = torch.tensor(hy, dtype=torch.float32).unsqueeze(0)
    z = torch.tensor(hz, dtype=torch.float32).unsqueeze(0)
    w = torch.ones_like(x)

    # Build z-fit weights: vtxd hits → 1.0, trkr hits → 0.0.
    # Always pass vtxd-only weights so that fits with < 3 vtxd hits are marked
    # failed (via sum_zw < 3 in _fit_helices_flat) rather than falling back to
    # trkr hits whose cm-scale z-resolution produces catastrophic z0 outliers.
    is_vtxd = np.zeros(n, dtype=bool)
    is_vtxd[:n_vtxd] = True
    is_vtxd_sorted = is_vtxd[order]
    zfit_w = torch.tensor(is_vtxd_sorted.astype(np.float32)).unsqueeze(0)

    R, phi0, eta, d0, z0, ok, cs = _fit_helices_flat(x, y, z, w, zfit_w=zfit_w)

    if not ok[0].item():
        return None

    pt, phi, eta_v, d0_mm, z0_mm = helix_params_to_track_params(
        R[0], phi0[0], eta[0], d0[0], z0[0], B_FIELD_T
    )
    return {
        "pt":          float(pt),
        "phi":         float(phi),
        "eta":         float(eta_v),
        "d0_mm":       float(d0_mm),
        "z0_mm":       float(z0_mm),
        "charge_sign": float(cs[0]),
    }


def _match_pandora(
    p_vtxd: torch.Tensor,   # (N_vtxd,) bool — truth particle's vtxd hits
    p_trkr: torch.Tensor,   # (N_trkr,) bool — truth particle's trkr hits
    pan_vtxd: torch.Tensor, # (N_pan, N_vtxd) bool
    pan_trkr: torch.Tensor, # (N_pan, N_trkr) bool
    pan_valid: torch.Tensor,   # (N_pan,) bool
    pan_charged: torch.Tensor, # (N_pan,) bool
) -> tuple[float, int]:
    """IoU-based matching of one truth particle to the best charged Pandora object.

    Returns (best_iou, pandora_index) or (0.0, -1) if no match exceeds threshold.
    """
    p_sihit = torch.cat([p_vtxd, p_trkr]).float()              # (N_sihit,)
    pan_sihit = torch.cat([pan_vtxd, pan_trkr], dim=-1).float() # (N_pan, N_sihit)

    intersection = (pan_sihit * p_sihit.unsqueeze(0)).sum(-1)
    union = (pan_sihit + p_sihit.unsqueeze(0) - pan_sihit * p_sihit.unsqueeze(0)).sum(-1)
    iou = intersection / union.clamp_min(1e-6)

    iou[~(pan_valid & pan_charged)] = 0.0

    best_iou, best_idx = iou.max(dim=0)
    if best_iou.item() < IOI_MATCH_THRESH:
        return 0.0, -1
    return float(best_iou.item()), int(best_idx.item())


# ── main collection loop ────────────────────────────────────────────────────

def collect_residuals(cfg: dict, n_events: int = N_EVENTS) -> dict[str, dict[str, np.ndarray]]:
    """Iterate over the dataset and collect track-parameter arrays.

    Returns a dict with keys ``"truth"``, ``"pandora"``, ``"helix"``, each
    mapping field name → 1-D NumPy array of values.  The Pandora and helix
    arrays are aligned to *each other* (only entries where *both* exist are
    kept), but they are subsets of the truth array.
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
        # Do not force-pad to 384; avoids wasting memory in an offline script
        force_pad_sizes=None,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    FIELDS = ["pt", "qopt", "eta", "phi", "d0", "z0"]

    # Separate accumulators for the two baselines so sizes can differ
    truth_all  = {f: [] for f in FIELDS}
    pan_vals   = {f: [] for f in FIELDS}  # only matched entries
    helix_vals = {f: [] for f in FIELDS}  # only fitted entries
    pan_helix_vals = {f: [] for f in FIELDS}  # helix fit on Pandora hits (matched only)
    # Aligned pairs for the residual vs truth plots
    pan_truth_aligned   = {f: [] for f in FIELDS}
    helix_truth_aligned = {f: [] for f in FIELDS}
    pan_helix_truth_aligned = {f: [] for f in FIELDS}

    n_total = n_no_match = n_fit_fail = n_pan_fit_fail = 0

    for inputs, targets in tqdm(loader, desc="Events", unit="ev"):
        # ── truth particle fields ──────────────────────────────────────────
        part_valid   = targets["particle_valid"][0].bool()     # (N_par,)
        part_charged = targets["particle_is_charged"][0].bool()

        pt_t   = _np(targets["particle_mom.r"][0])    # pT [GeV]
        eta_t  = _np(targets["particle_mom.eta"][0])
        phi_t  = _np(targets["particle_mom.phi"][0])
        qopt_t = _np(targets["particle_mom.qopt"][0]) # q/pT [GeV⁻¹]
        d0_t_mm  = _np(targets["particle_perigee.d0"][0])  # [mm]
        z0_t_mm  = _np(targets["particle_perigee.z0"][0])  # [mm]

        # ── pandora fields ─────────────────────────────────────────────────
        pan_valid   = targets["pandora_valid"][0].bool()
        pan_charged = targets["pandora_is_charged"][0].bool()
        pt_pan     = _np(targets["pandora_mom.r"][0])
        eta_pan    = _np(targets["pandora_mom.eta"][0])
        phi_pan    = _np(targets["pandora_mom.phi"][0])
        charge_pan = _np(targets["pandora_charge"][0])
        qopt_pan   = np.where(np.abs(pt_pan) > 1e-6, charge_pan / pt_pan, 0.0)
        d0_pan_mm  = _np(targets["pandora_perigee.d0"][0])  # [mm]
        z0_pan_mm  = _np(targets["pandora_perigee.z0"][0])  # [mm]

        # ── hit masks and positions ─────────────────────────────────────────
        p_vtxd_all  = targets["particle_vtxd_valid"][0]   # (N_par, N_vtxd) bool
        p_trkr_all  = targets["particle_trkr_valid"][0]   # (N_par, N_trkr) bool
        pan_vtxd_all = targets["pandora_vtxd_valid"][0]   # (N_pan, N_vtxd) bool
        pan_trkr_all = targets["pandora_trkr_valid"][0]   # (N_pan, N_trkr) bool

        vtxd_x = _np(inputs["vtxd_pos.x"][0])  # (N_vtxd,) [m]
        vtxd_y = _np(inputs["vtxd_pos.y"][0])
        vtxd_z = _np(inputs["vtxd_pos.z"][0])
        trkr_x = _np(inputs["trkr_pos.x"][0])  # (N_trkr,) [m]
        trkr_y = _np(inputs["trkr_pos.y"][0])
        trkr_z = _np(inputs["trkr_pos.z"][0])

        N_par = int(part_valid.shape[0])

        for p_idx in range(N_par):
            if not (part_valid[p_idx] and part_charged[p_idx]):
                continue

            n_total += 1

            p_vtxd = p_vtxd_all[p_idx]  # (N_vtxd,) bool
            p_trkr = p_trkr_all[p_idx]  # (N_trkr,) bool
            n_vtxd = int(p_vtxd.sum().item())
            n_trkr = int(p_trkr.sum().item())

            if n_vtxd + n_trkr < 3:
                continue

            # ── helix fit ──────────────────────────────────────────────────
            vtxd_sel = _np(p_vtxd).astype(bool)
            trkr_sel = _np(p_trkr).astype(bool)
            hx = np.concatenate([vtxd_x[vtxd_sel], trkr_x[trkr_sel]])
            hy = np.concatenate([vtxd_y[vtxd_sel], trkr_y[trkr_sel]])
            hz = np.concatenate([vtxd_z[vtxd_sel], trkr_z[trkr_sel]])
            hfit = _helix_fit_on_hits(hx, hy, hz, n_vtxd)

            # ── Pandora matching ───────────────────────────────────────────
            iou, pan_idx = _match_pandora(
                p_vtxd, p_trkr,
                pan_vtxd_all, pan_trkr_all,
                pan_valid, pan_charged,
            )

            if pan_idx < 0:
                n_no_match += 1

            if hfit is None:
                n_fit_fail += 1

            # ── helix fit on Pandora hits ───────────────────────────────────
            pan_hfit = None
            if pan_idx >= 0:
                pan_vtxd_sel = _np(pan_vtxd_all[pan_idx]).astype(bool)
                pan_trkr_sel = _np(pan_trkr_all[pan_idx]).astype(bool)
                n_vtxd_pan = int(pan_vtxd_sel.sum())
                phx = np.concatenate([vtxd_x[pan_vtxd_sel], trkr_x[pan_trkr_sel]])
                phy = np.concatenate([vtxd_y[pan_vtxd_sel], trkr_y[pan_trkr_sel]])
                phz = np.concatenate([vtxd_z[pan_vtxd_sel], trkr_z[pan_trkr_sel]])
                pan_hfit = _helix_fit_on_hits(phx, phy, phz, n_vtxd_pan)
                if pan_hfit is None:
                    n_pan_fit_fail += 1

            # Accumulate truth arrays (all charged valid particles)
            pi = p_idx
            truth_all["pt"].append(pt_t[pi])
            truth_all["qopt"].append(qopt_t[pi])
            truth_all["eta"].append(eta_t[pi])
            truth_all["phi"].append(phi_t[pi])
            truth_all["d0"].append(d0_t_mm[pi])
            truth_all["z0"].append(z0_t_mm[pi])

            # Pandora residuals (only where matched)
            if pan_idx >= 0:
                pan_vals["pt"].append(pt_pan[pan_idx] - pt_t[pi])
                pan_vals["qopt"].append(qopt_pan[pan_idx] - qopt_t[pi])
                pan_vals["eta"].append(eta_pan[pan_idx] - eta_t[pi])
                pan_vals["phi"].append(float(
                    np.arctan2(np.sin(phi_pan[pan_idx] - phi_t[pi]),
                               np.cos(phi_pan[pan_idx] - phi_t[pi]))
                ))
                for f in ["pt", "qopt", "eta", "phi"]:
                    pan_truth_aligned[f].append(truth_all[f][-1])
                pan_vals["d0"].append(d0_pan_mm[pan_idx] - d0_t_mm[pi])
                pan_vals["z0"].append(z0_pan_mm[pan_idx] - z0_t_mm[pi])
                for f in ["d0", "z0"]:
                    pan_truth_aligned[f].append(truth_all[f][-1])

            # Helix-fit residuals (only where fit succeeded)
            if hfit is not None:
                hq = hfit["charge_sign"] / max(abs(hfit["pt"]), 1e-6)
                helix_vals["pt"].append(hfit["pt"] - pt_t[pi])
                helix_vals["qopt"].append(hq - qopt_t[pi])
                helix_vals["eta"].append(hfit["eta"] - eta_t[pi])
                helix_vals["phi"].append(float(
                    np.arctan2(np.sin(hfit["phi"] - phi_t[pi]),
                               np.cos(hfit["phi"] - phi_t[pi]))
                ))
                for f in ["pt", "qopt", "eta", "phi"]:
                    helix_truth_aligned[f].append(truth_all[f][-1])
                helix_vals["d0"].append(hfit["d0_mm"] - d0_t_mm[pi])
                helix_vals["z0"].append(hfit["z0_mm"] - z0_t_mm[pi])
                for f in ["d0", "z0"]:
                    helix_truth_aligned[f].append(truth_all[f][-1])

            # Pandora-hit helix fit residuals (only where Pandora matched and fit succeeded)
            if pan_hfit is not None:
                phq = pan_hfit["charge_sign"] / max(abs(pan_hfit["pt"]), 1e-6)
                pan_helix_vals["pt"].append(pan_hfit["pt"] - pt_t[pi])
                pan_helix_vals["qopt"].append(phq - qopt_t[pi])
                pan_helix_vals["eta"].append(pan_hfit["eta"] - eta_t[pi])
                pan_helix_vals["phi"].append(float(
                    np.arctan2(np.sin(pan_hfit["phi"] - phi_t[pi]),
                               np.cos(pan_hfit["phi"] - phi_t[pi]))
                ))
                for f in ["pt", "qopt", "eta", "phi"]:
                    pan_helix_truth_aligned[f].append(truth_all[f][-1])
                pan_helix_vals["d0"].append(pan_hfit["d0_mm"] - d0_t_mm[pi])
                pan_helix_vals["z0"].append(pan_hfit["z0_mm"] - z0_t_mm[pi])
                for f in ["d0", "z0"]:
                    pan_helix_truth_aligned[f].append(truth_all[f][-1])

    n_matched = n_total - n_no_match
    print(f"\nProcessed {n_total} charged truth particles across {n_events} events.")
    print(f"  Pandora match rate      : {100*n_matched/max(n_total,1):.1f}%")
    print(f"  Helix fit success       : {100*(n_total-n_fit_fail)/max(n_total,1):.1f}%")
    print(f"  Pandora-hit helix fail  : {100*n_pan_fit_fail/max(n_matched,1):.1f}% of matched")

    def to_arr(d):
        return {f: np.array(v, dtype=np.float32) for f, v in d.items()}

    return {
        "truth":                   to_arr(truth_all),
        "pandora_residual":        to_arr(pan_vals),
        "helix_residual":          to_arr(helix_vals),
        "pan_helix_residual":      to_arr(pan_helix_vals),
        "pandora_truth":           to_arr(pan_truth_aligned),
        "helix_truth":             to_arr(helix_truth_aligned),
        "pan_helix_truth":         to_arr(pan_helix_truth_aligned),
    }


# ── plotting ────────────────────────────────────────────────────────────────

def build_series(data: dict, n_fields: int) -> dict[str, dict]:
    """Convert collected arrays into the series format expected by plotting.py."""
    FIELDS = ["pt", "qopt", "eta", "phi", "d0", "z0"]

    def _residuals(key):
        d = data[key]
        return [d.get(f) for f in FIELDS]

    series = {}
    if data["pandora_residual"]["pt"].size > 0:
        series["Pandora"] = {
            "color": "cornflowerblue", "ls": "-",
            "data":  _residuals("pandora_residual"),
            "truth": [data["pandora_truth"].get(f) for f in FIELDS],
        }
    if data["helix_residual"]["pt"].size > 0:
        series["Helix fit (truth hits)"] = {
            "color": "mediumseagreen", "ls": ":",
            "data":  _residuals("helix_residual"),
            "truth": [data["helix_truth"].get(f) for f in FIELDS],
        }
    if data["pan_helix_residual"]["pt"].size > 0:
        series["Helix fit (Pandora hits)"] = {
            "color": "teal", "ls": "--",
            "data":  _residuals("pan_helix_residual"),
            "truth": [data["pan_helix_truth"].get(f) for f in FIELDS],
        }
    return series


def make_resolution_series(
    series: dict[str, dict],
    data: dict,
) -> dict[str, dict]:
    """Derive a resolution series (relative residual) from the existing residual series."""
    FIELDS = ["pt", "qopt", "eta", "phi", "d0", "z0"]
    truth_keys = {
        "Pandora":                 "pandora_truth",
        "Helix fit (truth hits)":  "helix_truth",
        "Helix fit (Pandora hits)": "pan_helix_truth",
    }
    res_series: dict[str, dict] = {}
    for label, props in series.items():
        truth_key = truth_keys.get(label)
        if truth_key is None:
            continue
        truth_arrays = [data[truth_key].get(f) for f in FIELDS]
        res_data = []
        for i, f in enumerate(FIELDS):
            raw = props["data"][i]
            div = RESOLUTION_DIVIDER.get(f)
            if div is None or raw is None or raw.size == 0:
                res_data.append(raw)
            else:
                t = truth_arrays[i]
                if t is not None and t.size == raw.size:
                    res_data.append(raw / div(t))
                else:
                    res_data.append(raw)
        res_series[label] = {
            "color": props["color"], "ls": props["ls"],
            "data":  res_data,
            "truth": truth_arrays,
        }
    return res_series


def main() -> None:
    cfg = _cfg()
    data = collect_residuals(cfg, n_events=N_EVENTS)
    series = build_series(data, n_fields=6)

    if not series:
        print("No data collected — check data path and config.")
        return

    res_series = make_resolution_series(series, data)

    out_dir = Path(__file__).resolve().parents[1] / "plots" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (make_residual_fig(series,
                           suptitle="CLD track residuals — Pandora vs helix fit vs truth"),
         "cld_track_residuals.png"),
        (make_residual_fullrange_fig(series,
                                     suptitle="CLD track residuals (full range)"),
         "cld_track_residuals_fullrange.png"),
        (make_arcsinh_fig(series,
                          suptitle=r"CLD track residuals — $\mathrm{arcsinh}(\Delta/\mathrm{MAD})$"),
         "cld_track_residuals_arcsinh.png"),
        (make_residual_vs_truth_fig(series, RESIDUAL_YLABEL,
                                    suptitle="CLD track bias vs truth",
                                    min_bin_count=20),
         "cld_track_bias_vs_truth.png"),
        (make_residual_vs_truth_fig(res_series, RESOLUTION_YLABEL,
                                    suptitle="CLD track resolution vs truth",
                                    ylim_map=RESOLUTION_VS_TRUTH_YLIM,
                                    min_bin_count=20),
         "cld_track_resolution_vs_truth.png"),
    ]

    for fig, fname in plots:
        path = out_dir / fname
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
