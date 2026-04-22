"""Baseline plots for v7_ablation propreg model on ttbar_fixed.

Produces per-selection (tight/loose) residual plots comparing:
 * Charged tracking params (pred/seed/sitrack vs truth perigee)
 * Charged energy (track-derived with PID-conditional mass vs Pandora vs truth)
 * Charge sign accuracy (seed vs pred vs sitrack)
 * Neutral energy/eta/phi (model pred/calo centroid vs Pandora vs truth)

Key points
----------
- v7_ablation predates hybrid seeding, so we recompute the hybrid helix seed
  offline from boolean flow_{vtxd,trkr}_valid predicted masks.
- Truth for charged = perigee transported from creation vertex + momentum.
- Truth for neutral = particle_{energy, mom.eta, mom.phi} at creation vertex.
- Matching uses IoU-cost Hungarian, mirroring eval_tracking_all_vs_pandora.yaml.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from hepattn.models.matcher import Matcher
from hepattn.utils.eval_utils import apply_matching, calc_cost, calculate_selections
from hepattn.utils.helix import helix_perigee_from_vertex, helix_seed_3pt

mpl.use("Agg")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 9

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_FIELD = 2.0
# Helix seed needs ≥ 3 hits for a 3-point circle fit. v7 was TRAINED with MIN_SIHIT=5
# so the model only predicts pT (flow_pred_qpt != 0) for tracks with pred_nhits ≥ 5,
# but at analysis time we can still fit a seed down to 3 hits (pure geometry) and use
# it as a fall-back when the model abstains. Route 3 (pred_nhits < 3) still can't fit
# and drops to calo-head / calo-sum E.
MIN_SIHIT = 3
XMIN, XMAX, NBINS = 0.1, 100.0, 30
EPS = 1e-10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PID class indices (CLDTask.class_name_to_idx)
PID_CHARGED_HADRON = 2
PID_PHOTON = 3
PID_ELECTRON = 4
PID_MUON = 5
PID_CHARGED = {PID_CHARGED_HADRON, PID_ELECTRON, PID_MUON}
PID_NEUTRAL = {1, PID_PHOTON}  # neutral_hadron, photon

# Masses [GeV]
M_PION = 0.13957
M_ELECTRON = 5.109989e-4
M_MUON = 0.10566

# Colors
C_PRED = "#E63946"  # red — model
C_SEED = "#F4A261"  # orange — recomputed helix seed
C_SITRACK = "#457B9D"  # blue — SiTrack
C_PANDORA = "#2A9D8F"  # teal — Pandora

HITS = ["vtxd", "trkr", "ecal", "hcal", "muon"]
HITS_NO_MUON = ["vtxd", "trkr", "ecal", "hcal"]

# Match metrics mirroring eval_tracking_all_vs_pandora.yaml#eval.match_metrics.default
FLOW_MATCH_METRICS = {
    "sihit": {"weight": 1.0, "metric": "iou", "field": "valid"},
    "ecal": {"weight": 1.0, "metric": "iou", "field": "valid"},
    "hcal": {"weight": 1.0, "metric": "iou", "field": "valid"},
    "muon": {"weight": 1.0, "metric": "iou", "field": "valid"},
}
PANDORA_MATCH_METRICS = FLOW_MATCH_METRICS
SITRACK_MATCH_METRICS = {
    "vtxd": {"weight": 1.0, "metric": "dice", "field": "valid"},
    "trkr": {"weight": 1.0, "metric": "dice", "field": "valid"},
}

# Selections from eval_tracking_all_vs_pandora.yaml
SELECTIONS = {
    "charged_reconstructable_tight": [
        "is_primary",
        "mom.r >= 0.1",
        "mom.abs_eta <= 2.44",
        "vtx.r <= 50",
        "num_sihit >= 4",
        "isolation >= 0.02",
    ],
    "charged_reconstructable_loose": [
        "is_charged_hadron",
        "mom.r >= 0.01",
        "mom.abs_eta <= 4",
        "num_sihit >= 3",
        "calib_energy_ecal >= 0.01",
        "calib_energy_hcal >= 0.01",
    ],
    "neutral_reconstructable": [
        "is_neutral_hadron",
        "is_primary",
        "mom.r >= 0.1",
        "mom.abs_eta <= 2.44",
        "vtx.r <= 50",
        "isolation >= 0.02",
        "calib_energy_ecal >= 0.1",
        "calib_energy_hcal >= 0.1",
    ],
    "neutral_reconstructable_loose": [
        "is_neutral_hadron",
        "mom.r >= 0.01",
        "mom.abs_eta <= 4",
        "calib_energy_ecal >= 0.1",
        "calib_energy_hcal >= 0.1",
    ],
}

# Truth-particle fields we need
TRUTH_FIELDS = [
    "particle_valid",
    "particle_charge",
    "particle_energy",
    "particle_mom.r",
    "particle_mom.eta",
    "particle_mom.abs_eta",
    "particle_mom.phi",
    "particle_mom.qopt",
    "particle_mom.x",
    "particle_mom.y",
    "particle_mom.z",
    "particle_vtx.x",
    "particle_vtx.y",
    "particle_vtx.z",
    "particle_vtx.r",
    "particle_isolation",
    "particle_num_sihit",
    "particle_is_charged",
    "particle_is_neutral",
    "particle_is_charged_hadron",
    "particle_is_neutral_hadron",
    "particle_is_electron",
    "particle_is_muon",
    "particle_is_photon",
    "particle_is_primary",
    "particle_calib_energy_ecal",
    "particle_calib_energy_hcal",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dev(arr) -> torch.Tensor:
    return torch.from_numpy(arr[:]).to(DEVICE)


def _log_bins(xmin: float = XMIN, xmax: float = XMAX, nbins: int = NBINS):
    edges = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    xerr = 0.5 * widths
    return edges, centers, widths, xerr


def per_bin_stats(x: np.ndarray, r: np.ndarray, edges: np.ndarray, min_count: int = 20):
    nb = len(edges) - 1
    idx = np.digitize(x, edges) - 1
    valid = (idx >= 0) & (idx < nb) & np.isfinite(r) & np.isfinite(x)
    mean = np.full(nb, np.nan)
    median = np.full(nb, np.nan)
    q1 = np.full(nb, np.nan)
    q3 = np.full(nb, np.nan)
    cnt = np.zeros(nb, dtype=int)
    for i in range(nb):
        m = valid & (idx == i)
        cnt[i] = m.sum()
        if cnt[i] >= min_count:
            vals = r[m]
            mean[i] = np.mean(vals)
            median[i] = np.median(vals)
            q1[i] = np.quantile(vals, 0.25)
            q3[i] = np.quantile(vals, 0.75)
    iqr = q3 - q1
    keep = cnt >= min_count
    return mean, median, q1, q3, iqr, cnt, keep


def wrap_phi(d: np.ndarray) -> np.ndarray:
    """Wrap phi residual to [-pi, pi]."""
    return np.arctan2(np.sin(d), np.cos(d))


def plot_3panel(series: list[dict], xlabel: str, ylabel: str, title: str, out_path: Path,
                xmin: float = XMIN, xmax: float = XMAX, nbins: int = NBINS,
                show_iqr_box: bool = True) -> None:
    """3-panel: median (with optional IQR box), mean, IQR, vs truth pT."""
    edges, centers, widths, xerr = _log_bins(xmin, xmax, nbins)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for s in series:
        vals = s.get("vals")
        if vals is None or len(vals) == 0:
            continue
        mn, med, q1, _q3, iqr, _cnt, keep = per_bin_stats(s["x"], vals, edges)
        w = widths[keep] * 0.95
        c = s["color"]
        lbl = f"{s['label']} (N={s['n']:,})"

        # Median panel with optional IQR shading
        if show_iqr_box:
            axes[0].bar(centers[keep], iqr[keep], bottom=q1[keep], width=w, align="center",
                        alpha=0.12, linewidth=0, color=c)
        axes[0].errorbar(centers[keep], med[keep], xerr=xerr[keep], fmt="none", capsize=0,
                         elinewidth=2, color=c, label=lbl)

        axes[1].errorbar(centers[keep], mn[keep], xerr=xerr[keep], fmt="none", capsize=0,
                         elinewidth=2, color=c, label=lbl)
        axes[2].errorbar(centers[keep], iqr[keep], xerr=xerr[keep], fmt="none", capsize=0,
                         elinewidth=2, color=c, label=lbl)

    for ax, t in zip(axes, ("Median", "Mean", "IQR"), strict=False):
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_xlim(xmin, xmax)
        ax.set_title(t)
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel(ylabel)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path.relative_to(out_path.parents[2])}")


def plot_hist(series: list[dict], xlabel: str, title: str, out_path: Path, bins: int = 100) -> None:
    """Overlay histogram of residuals (x-limits from pooled IQR x 3)."""
    pooled = np.concatenate([s["vals"][np.isfinite(s["vals"])] for s in series
                              if s.get("vals") is not None and len(s["vals"]) > 0])
    if len(pooled) == 0:
        return
    q25, q75 = np.percentile(pooled, [25, 75])
    iqr = q75 - q25
    xlim = (q25 - 3 * iqr, q75 + 3 * iqr)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for s in series:
        vals = s.get("vals")
        if vals is None or len(vals) == 0:
            continue
        vfin = vals[np.isfinite(vals)]
        ax.hist(vfin, bins=bins, range=xlim, alpha=0.5, color=s["color"],
                label=f"{s['label']} (N={s['n']:,})", density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_charge_sign_accuracy(pt: np.ndarray, truth_sign: np.ndarray, series: list[dict],
                               out_path: Path) -> None:
    """Accuracy vs truth pT for sign(helix_seed), sign(model pred), sign(sitrack)."""
    edges, centers, _widths, xerr = _log_bins()
    nb = len(edges) - 1

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for s in series:
        sign = s["sign"]
        valid_mask = s.get("valid", np.ones_like(sign, dtype=bool)) & np.isfinite(sign) & np.isfinite(pt)
        if valid_mask.sum() == 0:
            continue
        x = pt[valid_mask]
        correct = (np.sign(sign[valid_mask]) == truth_sign[valid_mask]).astype(float)
        idx = np.digitize(x, edges) - 1
        acc = np.full(nb, np.nan)
        cnt = np.zeros(nb, dtype=int)
        for i in range(nb):
            m = idx == i
            cnt[i] = m.sum()
            if cnt[i] >= 20:
                acc[i] = correct[m].mean()
        keep = cnt >= 20
        ax.errorbar(centers[keep], acc[keep], xerr=xerr[keep], fmt="none", capsize=0, elinewidth=2,
                    color=s["color"], label=f"{s['label']} (N={valid_mask.sum():,})")

    ax.set_xscale("log")
    ax.set_xlabel("Truth $p_T$ [GeV]")
    ax.set_ylabel("Charge-sign accuracy")
    ax.set_title(out_path.stem.replace("_", " "))
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_path.relative_to(out_path.parents[2])}")


def write_metrics(series: list[dict], edges: np.ndarray, out_path: Path) -> None:
    nb = len(edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    with Path(out_path).open("w") as f:
        header = f"{'Bin':>14s}"
        for s in series:
            header += f" | {s['label']:>12s} median    mean      IQR       N"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i in range(nb):
            row = f"{centers[i]:>10.3f} GeV"
            for s in series:
                mn, med, q1, q3, iqr, cnt, _ = per_bin_stats(s["x"], s["vals"], edges)
                if cnt[i] >= 20:
                    row += f" | {med[i]:>+14.6f} {mn[i]:>+10.6f} {iqr[i]:>10.6f} {cnt[i]:>5d}"
                else:
                    row += f" | {'':>14s} {'':>10s} {'':>10s} {cnt[i]:>5d}"
            f.write(row + "\n")
        f.write("-" * len(header) + "\n")
        row = f"{'OVERALL':>14s}"
        for s in series:
            vals = s["vals"][np.isfinite(s["vals"])]
            if len(vals) > 0:
                med = float(np.median(vals))
                mn = float(np.mean(vals))
                q1, q3 = np.percentile(vals, [25, 75])
                row += f" | {med:>+14.6f} {mn:>+10.6f} {q3 - q1:>10.6f} {len(vals):>5d}"
            else:
                row += f" | {'':>14s} {'':>10s} {'':>10s} {0:>5d}"
        f.write(row + "\n")


SITRACK_STATE_FIELDS = ("omega", "D0", "Z0", "phi", "tanLambda", "location")


def _event_filename_to_event_id(path: Path) -> int:
    """Same inverse of data.py's filename-to-event_id mapping."""
    parts = path.stem.replace("_condor", "").split("_")
    job_id = parts[-3]
    proc_id = parts[-2]
    ev_id = parts[-1]
    return int(job_id + proc_id.zfill(4) + ev_id.zfill(4))


def build_sitrack_cache(npz_dir: Path, sample_ids: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Scan npz_dir once, pre-load sitrack.state.* for every sample_id we need.

    Returns {event_id: {field: (N_sitrack, 4) np.ndarray, ...}} for the subset of
    NPZ files whose event_id is in sample_ids.
    """
    want = {int(s) for s in sample_ids}
    print(f"Pre-scanning {npz_dir} for {len(want):,} NPZ files...")
    t0 = time.time()
    all_paths = list(npz_dir.rglob("*reco*.npz"))
    print(f"  found {len(all_paths):,} NPZ total in {time.time() - t0:.1f}s")

    path_by_id: dict[int, Path] = {}
    for p in all_paths:
        try:
            eid = _event_filename_to_event_id(p)
        except (ValueError, IndexError):
            continue
        if eid in want:
            path_by_id[eid] = p
    missing = want - set(path_by_id)
    if missing:
        print(f"  WARNING: {len(missing):,} sample_ids not found in NPZ dir (using 0 examples)")

    print(f"Loading sitrack state from {len(path_by_id):,} NPZ files...")
    t0 = time.time()
    cache: dict[int, dict[str, np.ndarray]] = {}
    bad = 0
    for i, (eid, path) in enumerate(path_by_id.items()):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(path_by_id):,} ({(time.time() - t0):.1f}s elapsed)", flush=True)
        try:
            arch = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            bad += 1
            continue
        entry: dict[str, np.ndarray] = {}
        ok = True
        for fi in SITRACK_STATE_FIELDS:
            k = f"sitrack.state.{fi}"
            if k not in arch.files:
                ok = False
                break
            a = np.asarray(arch[k])
            if a.ndim != 2:
                ok = False
                break
            entry[fi] = a.astype(np.float32)
        if ok:
            cache[eid] = entry
        else:
            bad += 1
    print(f"  cached {len(cache):,} events, {bad} bad, in {time.time() - t0:.1f}s")
    return cache


def _sitrack_state_from_cache(sample_id, cache: dict, pad_n: int) -> dict[str, torch.Tensor]:
    """Turn a cached {field: (N, 4) array} into (1, pad_n, 4) tensors on DEVICE."""
    entry = cache.get(int(sample_id))
    if not entry:
        return {}
    out: dict[str, torch.Tensor] = {}
    for fi in SITRACK_STATE_FIELDS:
        a = entry[fi]
        n_native, n_pt = a.shape
        padded = np.zeros((pad_n, n_pt), dtype=np.float32)
        padded[:n_native] = a
        out[f"sitrack_state.{fi}"] = torch.from_numpy(padded).unsqueeze(0).to(DEVICE)
    return out


def _extract_sitrack_5params_from_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute the 5 perigee params at AtIP (location==1) from a sitrack_state.* dict.

    state["sitrack_state.location"] / state["sitrack_state.omega"] etc. are (1, N, 4).
    """
    st_loc = state["sitrack_state.location"].detach().cpu().numpy().squeeze(0)  # (N, 4)
    ip_col = 0
    for c in range(st_loc.shape[1]):
        if np.any(st_loc[:, c] == 1):
            ip_col = c
            break

    result = {}
    st_omega = state["sitrack_state.omega"][:, :, ip_col]  # (1, N) in 1/mm
    result["inv_pt"] = st_omega.abs() * 1000.0 / (0.3 * B_FIELD)
    # EDM4HEP sitrack.state.omega convention: charge = +sign(omega).
    # Empirically validated on ttbar_fixed NPZ: 94.7% agreement with truth charge
    # (vs 5.3% for -sign(omega)) — see /share/gpu1/syw24/tmp/sitrack_debug/sitrack_charge_sign_debug.py.
    # NOTE: this differs from our internal helix_fit.py seed convention, which uses -sign(omega).
    result["qpt"] = st_omega * 1000.0 / (0.3 * B_FIELD)
    result["charge_sign"] = torch.sign(st_omega)

    st_tanl = state["sitrack_state.tanLambda"][:, :, ip_col]
    result["eta"] = torch.asinh(st_tanl)
    # helix_fit.py was aligned with EDM4HEP on 2026-04-19 (see d0_sign_convention_mismatch.md),
    # so SiTrack D0 can be used directly — no sign flip needed.
    result["d0"] = state["sitrack_state.D0"][:, :, ip_col]
    result["z0"] = state["sitrack_state.Z0"][:, :, ip_col]
    result["phi0"] = state["sitrack_state.phi"][:, :, ip_col]
    return result


def _flow_pred_charged_energy(pred_qpt: np.ndarray, seed_eta: np.ndarray,
                               pid_class: np.ndarray) -> np.ndarray:
    """PID-conditional track-derived energy for predicted charged particles.

    E = sqrt((pT * cosh(eta))^2 + m^2), with m determined by the predicted PID.
    """
    pt = 1.0 / np.clip(np.abs(pred_qpt), EPS, None)
    p = pt * np.cosh(seed_eta)
    mass = np.full_like(pt, M_PION, dtype=np.float64)
    mass[pid_class == PID_ELECTRON] = M_ELECTRON
    mass[pid_class == PID_MUON] = M_MUON
    return np.sqrt(p**2 + mass**2)


# ---------------------------------------------------------------------------
# Per-event data loading / matching
# ---------------------------------------------------------------------------


def load_event(f: h5py.File, sample_id: str, matcher: Matcher, sitrack_cache: dict | None = None) -> dict | None:
    """Load one event, run flow + pandora + sitrack matching, return per-particle arrays.

    Returns None if the event has no flow or targets.
    """
    try:
        targets = f[f"{sample_id}/targets"]
        inputs = f[f"{sample_id}/inputs"]
        preds_reco = f[f"{sample_id}/preds/final/reco"]
        preds_propreg = f[f"{sample_id}/preds/final/flow_property_regression"]
    except KeyError:
        return None

    data: dict[str, torch.Tensor] = {}
    for field in TRUTH_FIELDS:
        if field in targets:
            data[field] = _to_dev(targets[field])

    # --- Truth hit masks (particle -> hit) ---
    for hit in HITS:
        tk = f"particle_{hit}_valid"
        if tk in targets:
            data[tk] = _to_dev(targets[tk])
            # Global "hit exists" mask for IoU denominator
            data[f"{hit}_valid"] = data[tk].any(dim=-2)
    if "particle_vtxd_valid" in data and "particle_trkr_valid" in data:
        data["particle_sihit_valid"] = torch.cat([data["particle_vtxd_valid"], data["particle_trkr_valid"]], dim=-1)
    if "vtxd_valid" in data and "trkr_valid" in data:
        data["sihit_valid"] = torch.cat([data["vtxd_valid"], data["trkr_valid"]], dim=-1)

    # --- Flow predictions ---
    data["flow_valid"] = _to_dev(preds_reco["flow_valid"]).bool()
    if "flow_class_idx" in preds_reco:
        data["flow_class_idx"] = _to_dev(preds_reco["flow_class_idx"]).long()
    else:
        # Fallback: derive from flow_logit
        flow_logit = _to_dev(preds_reco["flow_logit"])
        data["flow_class_idx"] = flow_logit.argmax(-1).long()

    for hit in HITS:
        fk = f"flow_{hit}_valid"
        tk = f"particle_{hit}_valid"
        if fk in preds_reco and tk in data:
            data[fk] = _to_dev(preds_reco[fk])
            # Trim pred mask to match truth hit count, respect flow_valid
            data[fk] = data[fk][:, :, : data[tk].shape[-1]] & data["flow_valid"].unsqueeze(-1)
    if "flow_vtxd_valid" in data and "flow_trkr_valid" in data:
        data["flow_sihit_valid"] = torch.cat([data["flow_vtxd_valid"], data["flow_trkr_valid"]], dim=-1)

    # Flow propreg outputs
    data["flow_pred_qpt"] = _to_dev(preds_propreg["flow_pred_qpt"])
    if "flow_pred_energy" in preds_propreg:
        data["flow_pred_energy"] = _to_dev(preds_propreg["flow_pred_energy"])
    elif "flow_log_E" in preds_propreg:
        data["flow_pred_energy"] = torch.exp(_to_dev(preds_propreg["flow_log_E"]))
    else:
        data["flow_pred_energy"] = torch.zeros_like(data["flow_pred_qpt"])
    if "flow_calo_centroid_eta" in preds_propreg:
        data["flow_calo_centroid_eta"] = _to_dev(preds_propreg["flow_calo_centroid_eta"])
        data["flow_calo_centroid_phi"] = _to_dev(preds_propreg["flow_calo_centroid_phi"])
    elif "flow_calo_eta" in preds_propreg:
        data["flow_calo_centroid_eta"] = _to_dev(preds_propreg["flow_calo_eta"])
        data["flow_calo_centroid_phi"] = _to_dev(preds_propreg["flow_calo_phi"])
    else:
        data["flow_calo_centroid_eta"] = torch.zeros_like(data["flow_pred_qpt"])
        data["flow_calo_centroid_phi"] = torch.zeros_like(data["flow_pred_qpt"])

    # --- Pandora ---
    if "pandora_valid" in targets:
        data["pandora_valid"] = _to_dev(targets["pandora_valid"]).bool()
        for key in ("pandora_energy", "pandora_mom.eta", "pandora_mom.phi", "pandora_mom.r", "pandora_charge"):
            if key in targets:
                data[key] = _to_dev(targets[key])
        for hit in HITS:
            tk = f"pandora_{hit}_valid"
            if tk in targets:
                data[tk] = _to_dev(targets[tk])
                truth_key = f"particle_{hit}_valid"
                if truth_key in data:
                    data[tk] = data[tk][:, :, : data[truth_key].shape[-1]]
                data[tk] = data[tk] & data["pandora_valid"].unsqueeze(-1)
        if "pandora_vtxd_valid" in data and "pandora_trkr_valid" in data:
            data["pandora_sihit_valid"] = torch.cat([data["pandora_vtxd_valid"], data["pandora_trkr_valid"]], dim=-1)

    # --- SiTrack ---
    data["sitrack_valid"] = _to_dev(targets["sitrack_valid"]).bool()
    for hit in ("vtxd", "trkr"):
        tk = f"sitrack_{hit}_valid"
        if tk in targets:
            data[tk] = _to_dev(targets[tk])
            truth_key = f"particle_{hit}_valid"
            if truth_key in data:
                data[tk] = data[tk][:, :, : data[truth_key].shape[-1]]
            data[tk] = data[tk] & data["sitrack_valid"].unsqueeze(-1)
    if "sitrack_vtxd_valid" in data and "sitrack_trkr_valid" in data:
        data["sitrack_sihit_valid"] = torch.cat([data["sitrack_vtxd_valid"], data["sitrack_trkr_valid"]], dim=-1)

    # sitrack state fields: prefer H5 if present, else pull from pre-scanned NPZ cache
    state: dict[str, torch.Tensor] = {}
    pad_n = data["sitrack_valid"].shape[-1]  # 512
    if all(f"sitrack_state.{fi}" in targets for fi in SITRACK_STATE_FIELDS):
        for fi in SITRACK_STATE_FIELDS:
            state[f"sitrack_state.{fi}"] = _to_dev(targets[f"sitrack_state.{fi}"])
    elif sitrack_cache is not None:
        state = _sitrack_state_from_cache(sample_id, sitrack_cache, pad_n)
        if not state or not all(f"sitrack_state.{fi}" in state for fi in SITRACK_STATE_FIELDS):
            return None
    else:
        return None

    st_params = _extract_sitrack_5params_from_dict(state)
    for k, v in st_params.items():
        data[f"sitrack_{k}"] = v

    # --- Matching (permutes prediction fields in-place to align with truth) ---
    with torch.inference_mode():
        flow_m = {k: v for k, v in FLOW_MATCH_METRICS.items() if f"flow_{k}_valid" in data and f"particle_{k}_valid" in data}
        if flow_m:
            flow_cost = calc_cost(data, "particle", "flow", flow_m)
            if flow_cost is not None:
                apply_matching(data, "particle", "flow", flow_cost, matcher)
        pan_m = {k: v for k, v in PANDORA_MATCH_METRICS.items() if f"pandora_{k}_valid" in data and f"particle_{k}_valid" in data}
        if pan_m and "pandora_valid" in data:
            pan_cost = calc_cost(data, "particle", "pandora", pan_m)
            if pan_cost is not None:
                apply_matching(data, "particle", "pandora", pan_cost, matcher)
        st_m = {k: v for k, v in SITRACK_MATCH_METRICS.items() if f"sitrack_{k}_valid" in data}
        if st_m:
            st_cost = calc_cost(data, "particle", "sitrack", st_m)
            if st_cost is not None:
                apply_matching(data, "particle", "sitrack", st_cost, matcher)

    # --- Selections ---
    calculate_selections(data, "particle", SELECTIONS)

    # --- Truth perigee from creation vertex + momentum ---
    vtx_x = data["particle_vtx.x"][0].cpu().numpy()
    vtx_y = data["particle_vtx.y"][0].cpu().numpy()
    vtx_z = data["particle_vtx.z"][0].cpu().numpy()
    mom_x = data["particle_mom.x"][0].cpu().numpy()
    mom_y = data["particle_mom.y"][0].cpu().numpy()
    mom_z = data["particle_mom.z"][0].cpu().numpy()
    charge_ = data["particle_charge"][0].cpu().numpy()
    truth_perigee = helix_perigee_from_vertex(vtx_x, vtx_y, vtx_z, mom_x, mom_y, mom_z, charge_)
    truth_inv_pt = np.abs(truth_perigee["omega"]) * 1000.0 / (0.3 * B_FIELD)
    truth_eta_perigee = np.arcsinh(truth_perigee["tan_lambda"])

    # --- Hybrid helix seed (recomputed) from flow-predicted masks ---
    sihit_x_parts, sihit_y_parts, sihit_z_parts, sihit_t_parts = [], [], [], []
    flow_mask_parts = []
    for hit in ("vtxd", "trkr"):
        pos_x_key = f"{hit}_pos.x"
        if pos_x_key not in inputs:
            continue
        sihit_x_parts.append(_to_dev(inputs[pos_x_key]))
        sihit_y_parts.append(_to_dev(inputs[f"{hit}_pos.y"]))
        sihit_z_parts.append(_to_dev(inputs[f"{hit}_pos.z"]))
        if f"{hit}_time" in inputs:
            sihit_t_parts.append(_to_dev(inputs[f"{hit}_time"]))
        flow_mask_parts.append(data[f"flow_{hit}_valid"])

    if not sihit_x_parts or not flow_mask_parts:
        return None

    sihit_x = torch.cat(sihit_x_parts, dim=-1)
    sihit_y = torch.cat(sihit_y_parts, dim=-1)
    sihit_z = torch.cat(sihit_z_parts, dim=-1)
    sihit_t = torch.cat(sihit_t_parts, dim=-1) if sihit_t_parts else None
    flow_sihit_mask = torch.cat(flow_mask_parts, dim=-1)  # (1, N_query, N_sihit)

    n_vtxd = flow_mask_parts[0].shape[-1]
    vtxd_only = flow_sihit_mask.clone()
    vtxd_only[..., n_vtxd:] = False
    enough_vtxd = vtxd_only.sum(dim=-1, keepdim=True) >= 3
    z_valid_mask = torch.where(enough_vtxd, vtxd_only, flow_sihit_mask)

    n_hits_per_query = flow_sihit_mask[0].sum(dim=-1)
    has_enough = n_hits_per_query >= MIN_SIHIT

    n_q = flow_sihit_mask.shape[1]
    m_sihit = sihit_x.shape[-1]
    time_exp = sihit_t[0].unsqueeze(0).expand(n_q, m_sihit) if sihit_t is not None else None
    # Respect an explicit CLI override if set (see __main__ block); default is the auto-pick.
    _override = globals().get("_SORT_MODE_OVERRIDE")
    if _override == "rt":
        # Force pre-fix legacy mode (no time usage).
        sort_mode = "rt"
        time_exp = None
    elif _override == "time":
        sort_mode = "time"
    else:
        sort_mode = "hybrid" if time_exp is not None else "rt"
    with torch.inference_mode():
        helix = helix_seed_3pt(
            sihit_x[0].unsqueeze(0).expand(n_q, m_sihit),
            sihit_y[0].unsqueeze(0).expand(n_q, m_sihit),
            sihit_z[0].unsqueeze(0).expand(n_q, m_sihit),
            flow_sihit_mask[0].bool(),
            min_hits=MIN_SIHIT,
            z_valid=z_valid_mask[0].bool(),
            z_phi_sort=False,
            time=time_exp,
            sort_mode=sort_mode,
        )

    # Seed perigee (convert m → mm and 1/m → 1/mm to match SiTrack units)
    seed_d0 = helix["d0"].cpu().numpy() * 1000.0
    seed_z0 = helix["z0"].cpu().numpy() * 1000.0
    seed_omega_mm = helix["omega"].cpu().numpy() / 1000.0  # 1/mm
    seed_phi0 = helix["phi0"].cpu().numpy()
    seed_tanlam = helix["tan_lambda"].cpu().numpy()
    seed_inv_pt = np.abs(seed_omega_mm) * 1000.0 / (0.3 * B_FIELD)
    seed_eta = np.arcsinh(seed_tanlam)
    # Charge convention: positive charge -> omega < 0 (B_z > 0, CW).
    seed_charge_sign = -np.sign(seed_omega_mm)
    seed_valid_mask = has_enough.cpu().numpy()

    # --- Collect per-particle arrays ---
    pv = data["particle_valid"][0].bool().cpu().numpy()
    out: dict[str, np.ndarray] = {}

    def _np(k):
        return data[k][0].cpu().numpy()

    out["particle_valid"] = pv
    out["truth_pt"] = _np("particle_mom.r")
    out["truth_qpt"] = _np("particle_mom.qopt")
    out["truth_energy"] = _np("particle_energy")
    out["truth_eta"] = _np("particle_mom.eta")
    out["truth_phi"] = _np("particle_mom.phi")
    out["truth_charge"] = _np("particle_charge")
    out["truth_inv_pt"] = truth_inv_pt
    out["truth_d0"] = truth_perigee["d0"]
    out["truth_z0"] = truth_perigee["z0"]
    out["truth_phi0"] = truth_perigee["phi0"]
    out["truth_eta_perigee"] = truth_eta_perigee
    out["is_charged"] = _np("particle_is_charged").astype(bool)
    out["is_neutral"] = _np("particle_is_neutral").astype(bool)

    # Selections
    for sel in SELECTIONS:
        out[f"sel_{sel}"] = _np(f"particle_{sel}").astype(bool)

    # Flow (model)
    out["flow_valid"] = data["flow_valid"][0].cpu().numpy().astype(bool)
    out["flow_pred_qpt"] = _np("flow_pred_qpt")
    out["flow_pred_energy"] = _np("flow_pred_energy")
    out["flow_class_idx"] = data["flow_class_idx"][0].cpu().numpy().astype(int)
    out["flow_calo_centroid_eta"] = _np("flow_calo_centroid_eta")
    out["flow_calo_centroid_phi"] = _np("flow_calo_centroid_phi")

    # --- Raw calo-hit energy sum per matched particle (neutral-energy baseline) ---
    # Sum {ecal,hcal}_energy over the slots the model claims for this particle.
    # Purely calo-based — no ML energy head, no Pandora. Model-mask-dependent.
    calo_E_sum = None
    if "ecal_energy" in inputs and "flow_ecal_valid" in data \
            and "hcal_energy" in inputs and "flow_hcal_valid" in data:
        ecal_E = _to_dev(inputs["ecal_energy"])[0]  # (N_ecal_hits,)
        hcal_E = _to_dev(inputs["hcal_energy"])[0]  # (N_hcal_hits,)
        flow_ecal = data["flow_ecal_valid"][0].bool()  # (N_particles, N_ecal_hits) post-match
        flow_hcal = data["flow_hcal_valid"][0].bool()
        # Trim to matching hit dimension (particle_ecal_valid shape set this above)
        calo_E_sum = (flow_ecal.float() * ecal_E[None, : flow_ecal.shape[-1]]).sum(-1) \
                   + (flow_hcal.float() * hcal_E[None, : flow_hcal.shape[-1]]).sum(-1)

    # --- Energy-mean investigation fields (post-matching, per truth particle) ---
    if "particle_sihit_valid" in data and "flow_sihit_valid" in data:
        truth_m = data["particle_sihit_valid"][0].bool()  # [N_particles, N_sihit]
        flow_m = data["flow_sihit_valid"][0].bool()       # [N_particles, N_sihit] (post-match aligned)
        n_truth = truth_m.sum(dim=-1).float()
        n_flow = flow_m.sum(dim=-1).float()
        n_correct = (truth_m & flow_m).sum(dim=-1).float()
        n_fake = (flow_m & ~truth_m).sum(dim=-1).float()
        n_missing = (truth_m & ~flow_m).sum(dim=-1).float()
        union = (truth_m | flow_m).sum(dim=-1).float().clamp(min=1.0)
        match_iou = n_correct / union
        out["hit_n_truth"] = n_truth.cpu().numpy().astype(np.int32)
        out["hit_n_flow"] = n_flow.cpu().numpy().astype(np.int32)
        out["hit_n_correct"] = n_correct.cpu().numpy().astype(np.int32)
        out["hit_n_fake"] = n_fake.cpu().numpy().astype(np.int32)
        out["hit_n_missing"] = n_missing.cpu().numpy().astype(np.int32)
        out["hit_match_iou"] = match_iou.cpu().numpy().astype(np.float32)
    else:
        out["hit_n_truth"] = np.zeros_like(out["truth_pt"], dtype=np.int32)
        out["hit_n_flow"] = np.zeros_like(out["truth_pt"], dtype=np.int32)
        out["hit_n_correct"] = np.zeros_like(out["truth_pt"], dtype=np.int32)
        out["hit_n_fake"] = np.zeros_like(out["truth_pt"], dtype=np.int32)
        out["hit_n_missing"] = np.zeros_like(out["truth_pt"], dtype=np.int32)
        out["hit_match_iou"] = np.zeros_like(out["truth_pt"], dtype=np.float32)
    # pred-pT saturation flag (the 150 GeV clamp in the hybrid formula binds)
    native_inv_pt = np.abs(out["flow_pred_qpt"])
    out["pred_pt_saturated"] = (native_inv_pt < (1.0 / 150.0)).astype(bool)

    # Calo E sum per particle (third neutral-energy baseline alongside Pred calo head + Pandora)
    if calo_E_sum is not None:
        out["flow_calo_E_sum"] = calo_E_sum.cpu().numpy().astype(np.float32)
    else:
        out["flow_calo_E_sum"] = np.zeros_like(out["truth_pt"], dtype=np.float32)

    # Helix seed (from flow masks)
    out["seed_valid"] = seed_valid_mask.astype(bool)
    out["seed_inv_pt"] = seed_inv_pt
    out["seed_d0"] = seed_d0
    out["seed_z0"] = seed_z0
    out["seed_phi0"] = seed_phi0
    out["seed_eta"] = seed_eta
    out["seed_charge_sign"] = seed_charge_sign

    # SiTrack
    out["sitrack_valid"] = data["sitrack_valid"][0].cpu().numpy().astype(bool)
    out["sitrack_inv_pt"] = _np("sitrack_inv_pt")
    out["sitrack_qpt"] = _np("sitrack_qpt")
    out["sitrack_d0"] = _np("sitrack_d0")
    out["sitrack_z0"] = _np("sitrack_z0")
    out["sitrack_phi0"] = _np("sitrack_phi0")
    out["sitrack_eta"] = _np("sitrack_eta")
    out["sitrack_charge_sign"] = _np("sitrack_charge_sign")

    # Pandora
    if "pandora_valid" in data:
        out["pandora_valid"] = data["pandora_valid"][0].cpu().numpy().astype(bool)
        out["pandora_energy"] = _np("pandora_energy") if "pandora_energy" in data else np.full_like(out["truth_pt"], np.nan)
        out["pandora_eta"] = _np("pandora_mom.eta") if "pandora_mom.eta" in data else np.full_like(out["truth_pt"], np.nan)
        out["pandora_phi"] = _np("pandora_mom.phi") if "pandora_mom.phi" in data else np.full_like(out["truth_pt"], np.nan)
    else:
        out["pandora_valid"] = np.zeros_like(pv)
        out["pandora_energy"] = np.full_like(out["truth_pt"], np.nan)
        out["pandora_eta"] = np.full_like(out["truth_pt"], np.nan)
        out["pandora_phi"] = np.full_like(out["truth_pt"], np.nan)

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def load_all(h5_path: str, matcher: Matcher, n_events: int, npz_dir: Path | None = None) -> dict[str, np.ndarray]:
    print(f"=== Loading from {Path(h5_path).name} ===")
    # Fetch the list of sample_ids first so we can pre-scan only the NPZs we need.
    with h5py.File(h5_path, "r") as f:
        sids = sorted(f.keys())[:n_events]
    print(f"  {len(sids)} events")

    sitrack_cache: dict | None = None
    if npz_dir is not None:
        sitrack_cache = build_sitrack_cache(npz_dir, [int(s) for s in sids])

    agg: dict[str, list[np.ndarray]] = {}
    n_kept = 0
    with h5py.File(h5_path, "r") as f:
        for idx, sid in enumerate(sids):
            if (idx + 1) % 200 == 0:
                print(f"    event {idx + 1}/{len(sids)}", flush=True)
            out = load_event(f, sid, matcher, sitrack_cache=sitrack_cache)
            if out is None:
                continue
            n_kept += 1
            for k, v in out.items():
                agg.setdefault(k, []).append(v)
    print(f"  kept {n_kept} events, concatenating")
    return {k: np.concatenate(v) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Plot drivers
# ---------------------------------------------------------------------------


def _rel(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return (pred - truth) / np.clip(np.abs(truth), EPS, None)


def plot_charged_selection(d: dict[str, np.ndarray], sel_key: str, out_dir: Path) -> None:
    """All charged plots for one selection."""
    pv = d["particle_valid"]
    sel = d[f"sel_{sel_key}"] & pv & d["is_charged"]
    matched_flow = sel & d["flow_valid"]
    matched_pandora = sel & d["pandora_valid"]
    matched_sitrack = sel & d["sitrack_valid"]
    matched_seed = sel & d["seed_valid"]
    # Combined: only particles where all three reco chains produce something for fair tracking comparison
    tracking_all = sel & d["flow_valid"] & d["sitrack_valid"] & d["seed_valid"]

    print(f"  charged/{sel_key}: sel={sel.sum():,} flow={matched_flow.sum():,} "
          f"sitrack={matched_sitrack.sum():,} seed={matched_seed.sum():,} pandora={matched_pandora.sum():,}")
    if tracking_all.sum() < 50:
        print(f"  charged/{sel_key}: too few (tracking_all={tracking_all.sum()}), skipping tracking plots")

    edges, *_ = _log_bins()

    # 1) inv_pT relative residual (pred vs SiTrack vs seed)
    mm = tracking_all
    if mm.sum() > 50:
        x = d["truth_pt"][mm]
        tru = d["truth_inv_pt"][mm]
        series = [
            {"x": x, "vals": _rel(np.abs(d["flow_pred_qpt"][mm]), tru), "color": C_PRED, "label": "Pred", "n": mm.sum()},
            {"x": x, "vals": _rel(d["sitrack_inv_pt"][mm], tru), "color": C_SITRACK, "label": "SiTrack", "n": mm.sum()},
            {"x": x, "vals": _rel(d["seed_inv_pt"][mm], tru), "color": C_SEED, "label": "Seed (hybrid)", "n": mm.sum()},
        ]
        plot_3panel(series, "Truth $p_T$ [GeV]", "$(1/p_T)^{pred}-(1/p_T)^{true}$ / $(1/p_T)^{true}$",
                    f"1/pT rel. residual — charged_{sel_key}", out_dir / "inv_pt_rel.png")
        plot_hist(series, "1/pT relative residual", f"1/pT rel. residual — charged_{sel_key}", out_dir / "inv_pt_rel_hist.png")
        write_metrics(series, edges, out_dir / "inv_pt_rel_metrics.txt")

    # 2) q/pT relative residual
    if mm.sum() > 50:
        x = d["truth_pt"][mm]
        tqpt = d["truth_qpt"][mm]
        series = [
            {"x": x, "vals": _rel(d["flow_pred_qpt"][mm], tqpt), "color": C_PRED, "label": "Pred", "n": mm.sum()},
            {"x": x, "vals": _rel(d["sitrack_qpt"][mm], tqpt), "color": C_SITRACK, "label": "SiTrack", "n": mm.sum()},
        ]
        plot_3panel(series, "Truth $p_T$ [GeV]", "$(q/p_T)$ rel. residual",
                    f"q/pT rel. residual — charged_{sel_key}", out_dir / "qpt_rel.png")
        plot_hist(series, "q/pT relative residual", f"q/pT rel. residual — charged_{sel_key}", out_dir / "qpt_rel_hist.png")
        write_metrics(series, edges, out_dir / "qpt_rel_metrics.txt")

    # 3) d0/z0/phi0/eta — seed vs SiTrack vs truth-perigee
    if mm.sum() > 50:
        x = d["truth_pt"][mm]
        for param, ylabel, truth_col in [
            ("d0", "$d_0$ residual [mm]", "truth_d0"),
            ("z0", "$z_0$ residual [mm]", "truth_z0"),
            ("phi0", "$\\phi_0$ residual [rad]", "truth_phi0"),
            ("eta", "$\\eta$ residual", "truth_eta_perigee"),
        ]:
            tru = d[truth_col][mm]
            seed_res = d[f"seed_{param}"][mm] - tru
            sitrack_res = d[f"sitrack_{param}"][mm] - tru
            if param == "phi0":
                seed_res = wrap_phi(seed_res)
                sitrack_res = wrap_phi(sitrack_res)
            series = [
                {"x": x, "vals": seed_res, "color": C_SEED, "label": "Seed (hybrid)", "n": mm.sum()},
                {"x": x, "vals": sitrack_res, "color": C_SITRACK, "label": "SiTrack", "n": mm.sum()},
            ]
            plot_3panel(series, "Truth $p_T$ [GeV]", ylabel,
                        f"{param} residual — charged_{sel_key}", out_dir / f"{param}_residual.png")
            plot_hist(series, ylabel, f"{param} residual — charged_{sel_key}", out_dir / f"{param}_residual_hist.png")
            write_metrics(series, edges, out_dir / f"{param}_residual_metrics.txt")

    # 4) Charged energy: track-derived (PID-conditional) vs Pandora vs truth.
    # The v7_ablation model abstains (flow_pred_qpt == 0) for seeds with <5 hits; those
    # entries are excluded from the track-derived series because 1/|qpt| is undefined.
    # The IQR box overlay on the Median panel is disabled — Pandora's wider IQR was
    # drowning out the narrow track-derived error bars (also noted in
    # session_wrap_20260422_routed_E_complete.md).
    charged_pred = matched_flow & matched_seed & np.isin(d["flow_class_idx"], list(PID_CHARGED))
    pred_nonzero = charged_pred & (np.abs(d["flow_pred_qpt"]) > 0)
    if pred_nonzero.sum() > 50:
        x = d["truth_pt"][pred_nonzero]
        tru_E = d["truth_energy"][pred_nonzero]
        pred_E = _flow_pred_charged_energy(
            d["flow_pred_qpt"][pred_nonzero],
            d["seed_eta"][pred_nonzero],
            d["flow_class_idx"][pred_nonzero],
        )
        pred_rel = (pred_E - tru_E) / np.clip(np.abs(tru_E), EPS, None)
        n_abstain = int(charged_pred.sum() - pred_nonzero.sum())
        print(f"    charged_{sel_key}: {n_abstain:,} track-E abstainers (flow_pred_qpt==0) excluded from E plot")
        series = [{"x": x, "vals": pred_rel, "color": C_PRED,
                   "label": "Track-derived (PID cond.)", "n": pred_nonzero.sum()}]
        pandora_mask = charged_pred & d["pandora_valid"]
        if pandora_mask.sum() > 50:
            xp = d["truth_pt"][pandora_mask]
            pan_rel = (d["pandora_energy"][pandora_mask] - d["truth_energy"][pandora_mask]) / \
                       np.clip(np.abs(d["truth_energy"][pandora_mask]), EPS, None)
            series.append({"x": xp, "vals": pan_rel, "color": C_PANDORA,
                           "label": "Pandora", "n": pandora_mask.sum()})
        plot_3panel(series, "Truth $p_T$ [GeV]", "$(E^{pred}-E^{true})/E^{true}$",
                    f"Charged energy rel. residual — charged_{sel_key}", out_dir / "energy_rel.png",
                    show_iqr_box=False)
        plot_hist(series, "Energy relative residual", f"Charged energy rel. residual — charged_{sel_key}",
                  out_dir / "energy_rel_hist.png")
        write_metrics(series, edges, out_dir / "energy_rel_metrics.txt")

    # 5) Charge-sign accuracy — helix seed vs model pred vs SiTrack
    if sel.sum() > 50:
        series = [
            {"sign": d["seed_charge_sign"], "valid": matched_seed, "color": C_SEED, "label": "Seed"},
            {"sign": np.sign(d["flow_pred_qpt"]), "valid": matched_flow, "color": C_PRED, "label": "Model pred"},
            {"sign": d["sitrack_charge_sign"], "valid": matched_sitrack, "color": C_SITRACK, "label": "SiTrack"},
        ]
        # Constrain to selected particles and finite truth
        for s in series:
            s["valid"] = s["valid"] & sel
            s["sign"] = s["sign"].astype(float)
        plot_charge_sign_accuracy(d["truth_pt"], np.sign(d["truth_charge"]),
                                  series, out_dir / f"charge_sign_accuracy_{sel_key}.png")


def plot_neutral_selection(d: dict[str, np.ndarray], sel_key: str, out_dir: Path) -> None:
    pv = d["particle_valid"]
    sel = d[f"sel_{sel_key}"] & pv & d["is_neutral"]
    matched_flow = sel & d["flow_valid"]
    matched_pandora = sel & d["pandora_valid"]
    print(f"  neutral/{sel_key}: sel={sel.sum():,} flow={matched_flow.sum():,} pandora={matched_pandora.sum():,}")
    if matched_flow.sum() < 50:
        print(f"  neutral/{sel_key}: too few, skipping")
        return

    edges, *_ = _log_bins()
    x_f = d["truth_pt"][matched_flow]
    tru_E = d["truth_energy"][matched_flow]
    pred_E = d["flow_pred_energy"][matched_flow]

    # 6) Neutral energy
    series = [{"x": x_f, "vals": (pred_E - tru_E) / np.clip(np.abs(tru_E), EPS, None),
               "color": C_PRED, "label": "Model pred", "n": matched_flow.sum()}]
    if matched_pandora.sum() > 50:
        xp = d["truth_pt"][matched_pandora]
        pan_rel = (d["pandora_energy"][matched_pandora] - d["truth_energy"][matched_pandora]) / \
                   np.clip(np.abs(d["truth_energy"][matched_pandora]), EPS, None)
        series.append({"x": xp, "vals": pan_rel, "color": C_PANDORA, "label": "Pandora",
                       "n": matched_pandora.sum()})
    plot_3panel(series, "Truth $p_T$ [GeV]", "$(E^{pred}-E^{true})/E^{true}$",
                f"Neutral energy rel. residual — {sel_key}", out_dir / "energy_rel.png")
    plot_hist(series, "Energy relative residual", f"Neutral energy rel. residual — {sel_key}",
              out_dir / "energy_rel_hist.png")
    write_metrics(series, edges, out_dir / "energy_rel_metrics.txt")

    # 7) Neutral eta — calo centroid vs Pandora vs truth
    series = [{"x": x_f, "vals": d["flow_calo_centroid_eta"][matched_flow] - d["truth_eta"][matched_flow],
               "color": C_PRED, "label": "Calo centroid", "n": matched_flow.sum()}]
    if matched_pandora.sum() > 50:
        series.append({"x": d["truth_pt"][matched_pandora],
                        "vals": d["pandora_eta"][matched_pandora] - d["truth_eta"][matched_pandora],
                        "color": C_PANDORA, "label": "Pandora", "n": matched_pandora.sum()})
    plot_3panel(series, "Truth $p_T$ [GeV]", "$\\eta$ residual",
                f"Neutral $\\eta$ residual — {sel_key}", out_dir / "eta_residual.png")
    plot_hist(series, "$\\eta$ residual", f"Neutral $\\eta$ residual — {sel_key}", out_dir / "eta_residual_hist.png")
    write_metrics(series, edges, out_dir / "eta_residual_metrics.txt")

    # 8) Neutral phi — wrapped
    series = [{"x": x_f, "vals": wrap_phi(d["flow_calo_centroid_phi"][matched_flow] - d["truth_phi"][matched_flow]),
               "color": C_PRED, "label": "Calo centroid", "n": matched_flow.sum()}]
    if matched_pandora.sum() > 50:
        series.append({"x": d["truth_pt"][matched_pandora],
                        "vals": wrap_phi(d["pandora_phi"][matched_pandora] - d["truth_phi"][matched_pandora]),
                        "color": C_PANDORA, "label": "Pandora", "n": matched_pandora.sum()})
    plot_3panel(series, "Truth $p_T$ [GeV]", "$\\phi$ residual [rad]",
                f"Neutral $\\phi$ residual — {sel_key}", out_dir / "phi_residual.png")
    plot_hist(series, "$\\phi$ residual [rad]", f"Neutral $\\phi$ residual — {sel_key}", out_dir / "phi_residual_hist.png")
    write_metrics(series, edges, out_dir / "phi_residual_metrics.txt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="v7_ablation baseline plots vs SiTrack + Pandora")
    parser.add_argument(
        "--h5",
        type=str,
        default="/share/gpu1/syw24/hepattn/logs/CLD_kmax_frozen_property_regression_v7_ablation_20260326-T155114/ckpts/"
                "epoch=000-val_loss=15.70569_ttbar_fixed_eval.h5",
        help="Path to v7_ablation eval H5 (re-run on ttbar_fixed with pandora energy).",
    )
    parser.add_argument("--n_events", type=int, default=2000)
    parser.add_argument("--out_dir", type=str,
                        default="/share/gpu1/syw24/hepattn/src/hepattn/experiments/cld/plots/baseline_v7_ablation")
    parser.add_argument(
        "--npz_dir",
        type=str,
        default="/share/rcif2/maxhart/data/cld/prepped/ttbar_fixed",
        help="NPZ root dir to load sitrack.state.* fields when H5 lacks them.",
    )
    parser.add_argument(
        "--cache_pkl",
        type=str,
        default=None,
        help="If set, pickle the aggregated dict to this path after loading (for notebook reuse).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plotting (useful together with --cache_pkl to just rebuild the cache).",
    )
    parser.add_argument(
        "--sort_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "rt", "time"],
        help="helix_seed_3pt hit-ordering strategy: hybrid (default, 2026-04-15 fix), "
             "rt (pre-fix legacy), or time (time-only).",
    )
    args = parser.parse_args()

    # Propagate sort_mode to load_event via a module-level override so we don't have to plumb
    # it through load_all -> load_event signature chain.
    global _SORT_MODE_OVERRIDE
    _SORT_MODE_OVERRIDE = args.sort_mode

    t0 = time.time()
    matcher = Matcher(adaptive_solver=False)
    npz_dir = Path(args.npz_dir) if args.npz_dir else None
    d = load_all(args.h5, matcher, args.n_events, npz_dir=npz_dir)
    print(f"Loaded in {time.time() - t0:.0f}s — {len(d['particle_valid']):,} particles total")

    if args.cache_pkl:
        import pickle
        cache_path = Path(args.cache_pkl)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = cache_path.stat().st_size / 1e6
        print(f"Pickled aggregated dict to {cache_path} ({size_mb:.1f} MB)")

    # Sanity: truth charge sign consistency
    tc = d["truth_charge"][d["is_charged"]]
    if len(tc) > 0:
        print(f"  truth charge sign summary: pos={np.mean(np.sign(tc) > 0):.3f}, "
              f"neg={np.mean(np.sign(tc) < 0):.3f}, zero={np.mean(np.sign(tc) == 0):.3f}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.no_plots:
        print("--no_plots set; skipping plotting.")
    else:
        # Charged plots (tight + loose)
        plot_charged_selection(d, "charged_reconstructable_tight", out_root / "charged_tight")
        plot_charged_selection(d, "charged_reconstructable_loose", out_root / "charged_loose")

        # Neutral plots (tight + loose)
        plot_neutral_selection(d, "neutral_reconstructable", out_root / "neutral_tight")
        plot_neutral_selection(d, "neutral_reconstructable_loose", out_root / "neutral_loose")

        print(f"\nAll plots under {out_root}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
