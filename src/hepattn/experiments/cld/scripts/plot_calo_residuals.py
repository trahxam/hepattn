"""CLD calorimeter residual plots: Pandora vs energy-weighted calo fit vs truth.

For each truth particle with ECAL / HCAL deposits the script:
  1. Computes an energy-weighted mean η/φ from the truth calorimeter hits.
  2. Matches the best Pandora object by ECAL+HCAL hit IoU.
  3. Compares η and φ residuals for both baselines.

Produces four separate output figures:
  calo_ecal_charged.png   — ECAL, charged particles
  calo_ecal_neutral.png   — ECAL, neutral particles
  calo_hcal_charged.png   — HCAL, charged particles
  calo_hcal_neutral.png   — HCAL, neutral particles

Each figure shows bias (median ± IQR) and spread (IQR) vs truth η and φ.

Usage (from repo root)::

    python src/hepattn/experiments/cld/scripts/plot_calo_residuals.py
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

plt.rcParams["figure.dpi"] = 300
plt.rcParams["text.usetex"] = True

# ── constants ──────────────────────────────────────────────────────────────
IOI_MATCH_THRESH = 0.5
N_EVENTS         = 100
FIELDS           = ["eta", "phi"]

TRUTH_BINS = {
    "eta": (np.linspace(-3.0, 3.0, 24),     r"Truth $\eta$",        "linear"),
    "phi": (np.linspace(-np.pi, np.pi, 24), r"Truth $\phi$ [rad]",  "linear"),
}
RESIDUAL_YLABEL = {
    "eta": r"$\eta^\mathrm{pred} - \eta^\mathrm{true}$",
    "phi": r"$\phi^\mathrm{pred} - \phi^\mathrm{true}$ [rad]",
}

# ── energy residual constants ───────────────────────────────────────────────
ENERGY_TRUTH_BINS_E    = np.logspace(-1, 2, 24)              # GeV, log scale
ENERGY_TRUTH_BINS_ETA  = np.linspace(-3.0, 3.0, 24)
ENERGY_RESIDUAL_YLABEL = r"$(E^\mathrm{calo} - E^\mathrm{truth}) / E^\mathrm{truth}$"


# ── helpers ────────────────────────────────────────────────────────────────

def _cfg() -> dict:
    p = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"
    return yaml.safe_load(p.read_text())["data"]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _dphi(a: float, b: float) -> float:
    """Signed angular difference a − b wrapped to (−π, π]."""
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _calo_fit_np(
    eta: np.ndarray, phi: np.ndarray, energy: np.ndarray
) -> tuple[float, float] | None:
    """Energy-weighted mean η and φ (numpy). Returns (eta, phi) or None."""
    total = float(energy.sum())
    if total < 1e-12:
        return None
    eta_m = float((energy * eta).sum() / total)
    sin_m = float((energy * np.sin(phi)).sum() / total)
    cos_m = float((energy * np.cos(phi)).sum() / total)
    return eta_m, float(np.arctan2(sin_m, cos_m))


def _match_pandora(
    p_ecal: torch.Tensor,
    p_hcal: torch.Tensor,
    pan_ecal: torch.Tensor,
    pan_hcal: torch.Tensor,
    pan_valid: torch.Tensor,
    pan_is_same_type: torch.Tensor,
) -> int:
    """Match truth particle to best Pandora object via ECAL+HCAL hit IoU."""
    p_calo   = torch.cat([p_ecal, p_hcal]).float()
    pan_calo = torch.cat([pan_ecal, pan_hcal], dim=-1).float()
    inter = (pan_calo * p_calo.unsqueeze(0)).sum(-1)
    union = (pan_calo + p_calo.unsqueeze(0) - pan_calo * p_calo.unsqueeze(0)).sum(-1)
    iou = inter / union.clamp_min(1e-6)
    iou[~(pan_valid & pan_is_same_type)] = 0.0
    best_iou, best_idx = iou.max(dim=0)
    return int(best_idx.item()) if best_iou.item() >= IOI_MATCH_THRESH else -1


# ── data collection ────────────────────────────────────────────────────────

def collect_residuals(cfg: dict, n_events: int = N_EVENTS) -> tuple[dict, dict]:
    """Iterate the dataset and accumulate calo residuals.

    Returns:
      data  — nested dict for η/φ residuals:
                data[calo][charge_type]["fit_res"|"pan_res"|"fit_truth"|"pan_truth"][field]
      edata — nested dict for energy residuals:
                edata[charge_type]["truth_res"|"pan_res"|"truth_truth_e"|"truth_truth_eta"|
                                   "pan_truth_e"|"pan_truth_eta"]  (lists of floats)
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
        include_classes=None,  # include all classes; neutrals filtered by calo hit count
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
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=16)

    def _empty():
        return {
            calo: {
                ct: {
                    key: {f: [] for f in FIELDS}
                    for key in ("fit_res", "pan_res", "fit_truth", "pan_truth")
                }
                for ct in ("charged", "neutral")
            }
            for calo in ("ecal", "hcal", "calo")
        }

    data = _empty()

    edata: dict = {
        ct: {key: [] for key in (
            "truth_res",          "pan_res",          "pan_reported_res",   "truth_full_res",
            "truth_truth_e",      "truth_truth_eta",
            "pan_truth_e",        "pan_truth_eta",
            "pan_rep_truth_e",    "pan_rep_truth_eta",
            "truth_full_truth_e", "truth_full_truth_eta",
        )}
        for ct in ("charged", "neutral")
    }

    n_total = n_no_match = n_ecal_fit_fail = n_hcal_fit_fail = 0

    for inputs, targets in tqdm(loader, desc="Events", unit="ev"):
        part_valid   = targets["particle_valid"][0].bool()
        part_charged = targets["particle_is_charged"][0].bool()

        eta_t = _np(targets["particle_mom.eta"][0])
        phi_t = _np(targets["particle_mom.phi"][0])

        # Pandora arrays
        pan_valid   = targets["pandora_valid"][0].bool()
        pan_charged = targets["pandora_is_charged"][0].bool()
        eta_pan = _np(targets["pandora_mom.eta"][0])
        phi_pan = _np(targets["pandora_mom.phi"][0])

        # Calo hit positions and energies (shared across all particles in the event)
        ecal_eta_all = _np(inputs["ecal_pos.eta"][0])
        ecal_phi_all = _np(inputs["ecal_pos.phi"][0])
        hcal_eta_all = _np(inputs["hcal_pos.eta"][0])
        hcal_phi_all = _np(inputs["hcal_pos.phi"][0])
        ecal_e_all   = _np(inputs["ecal_energy"][0])   # calibrated hit energies
        hcal_e_all   = _np(inputs["hcal_energy"][0])

        for p_idx in range(int(part_valid.shape[0])):
            if not part_valid[p_idx]:
                continue

            is_charged  = bool(part_charged[p_idx].item())
            charge_type = "charged" if is_charged else "neutral"

            truth_eta    = float(eta_t[p_idx])
            truth_phi    = float(phi_t[p_idx])
            truth_e      = float(_np(targets["particle_energy"][0])[p_idx])
            truth_calo_e = float(_np(targets["particle_calib_energy_calo"][0])[p_idx])

            # ── ECAL hit energies and positions for this particle ──────────
            ecal_mask   = _np(targets["particle_ecal_valid"][0][p_idx]).astype(bool)
            ecal_energy = _np(targets["particle_ecal_energy"][0][p_idx])
            n_ecal = int(ecal_mask.sum())

            # ── HCAL hit energies and positions for this particle ──────────
            hcal_mask   = _np(targets["particle_hcal_valid"][0][p_idx]).astype(bool)
            hcal_energy = _np(targets["particle_hcal_energy"][0][p_idx])
            n_hcal = int(hcal_mask.sum())

            if n_ecal + n_hcal == 0:
                continue

            n_total += 1

            # ── Pandora match ──────────────────────────────────────────────
            pan_is_type = pan_charged if is_charged else ~pan_charged
            pan_idx = _match_pandora(
                targets["particle_ecal_valid"][0][p_idx],
                targets["particle_hcal_valid"][0][p_idx],
                targets["pandora_ecal_valid"][0],
                targets["pandora_hcal_valid"][0],
                pan_valid, pan_is_type,
            )
            if pan_idx < 0:
                n_no_match += 1

            # ── energy residuals ───────────────────────────────────────────
            if truth_e > 1e-6:
                ed = edata[charge_type]
                ed["truth_res"].append((truth_calo_e - truth_e) / truth_e)
                ed["truth_truth_e"].append(truth_e)
                ed["truth_truth_eta"].append(truth_eta)

                if pan_idx >= 0:
                    pan_ecal_m = _np(targets["pandora_ecal_valid"][0][pan_idx]).astype(bool)
                    pan_hcal_m = _np(targets["pandora_hcal_valid"][0][pan_idx]).astype(bool)
                    pan_calo_e = float(ecal_e_all[pan_ecal_m].sum()) + float(hcal_e_all[pan_hcal_m].sum())
                    ed["pan_res"].append((pan_calo_e - truth_e) / truth_e)
                    ed["pan_truth_e"].append(truth_e)
                    ed["pan_truth_eta"].append(truth_eta)

                    pan_reported_e = float(_np(targets["pandora_energy"][0])[pan_idx])
                    ed["pan_reported_res"].append((pan_reported_e - truth_e) / truth_e)
                    ed["pan_rep_truth_e"].append(truth_e)
                    ed["pan_rep_truth_eta"].append(truth_eta)

                truth_full_e = float(_np(targets["particle_calib_full_energy_calo"][0])[p_idx])
                ed["truth_full_res"].append((truth_full_e - truth_e) / truth_e)
                ed["truth_full_truth_e"].append(truth_e)
                ed["truth_full_truth_eta"].append(truth_eta)

            # ── ECAL residuals ─────────────────────────────────────────────
            if n_ecal > 0:
                fit = _calo_fit_np(
                    ecal_eta_all[ecal_mask],
                    ecal_phi_all[ecal_mask],
                    ecal_energy[ecal_mask],
                )
                if fit is not None:
                    d = data["ecal"][charge_type]
                    d["fit_res"]["eta"].append(fit[0] - truth_eta)
                    d["fit_res"]["phi"].append(_dphi(fit[1], truth_phi))
                    d["fit_truth"]["eta"].append(truth_eta)
                    d["fit_truth"]["phi"].append(truth_phi)
                else:
                    n_ecal_fit_fail += 1

                if pan_idx >= 0:
                    d = data["ecal"][charge_type]
                    d["pan_res"]["eta"].append(float(eta_pan[pan_idx]) - truth_eta)
                    d["pan_res"]["phi"].append(_dphi(float(phi_pan[pan_idx]), truth_phi))
                    d["pan_truth"]["eta"].append(truth_eta)
                    d["pan_truth"]["phi"].append(truth_phi)

            # ── HCAL residuals ─────────────────────────────────────────────
            if n_hcal > 0:
                fit = _calo_fit_np(
                    hcal_eta_all[hcal_mask],
                    hcal_phi_all[hcal_mask],
                    hcal_energy[hcal_mask],
                )
                if fit is not None:
                    d = data["hcal"][charge_type]
                    d["fit_res"]["eta"].append(fit[0] - truth_eta)
                    d["fit_res"]["phi"].append(_dphi(fit[1], truth_phi))
                    d["fit_truth"]["eta"].append(truth_eta)
                    d["fit_truth"]["phi"].append(truth_phi)
                else:
                    n_hcal_fit_fail += 1

                if pan_idx >= 0:
                    d = data["hcal"][charge_type]
                    d["pan_res"]["eta"].append(float(eta_pan[pan_idx]) - truth_eta)
                    d["pan_res"]["phi"].append(_dphi(float(phi_pan[pan_idx]), truth_phi))
                    d["pan_truth"]["eta"].append(truth_eta)
                    d["pan_truth"]["phi"].append(truth_phi)

            # ── Combined ECAL+HCAL fit ─────────────────────────────────────
            if n_ecal + n_hcal > 0:
                comb_eta = np.concatenate([ecal_eta_all[ecal_mask], hcal_eta_all[hcal_mask]])
                comb_phi = np.concatenate([ecal_phi_all[ecal_mask], hcal_phi_all[hcal_mask]])
                comb_e   = np.concatenate([ecal_energy[ecal_mask],  hcal_energy[hcal_mask]])
                comb_fit = _calo_fit_np(comb_eta, comb_phi, comb_e)
                if comb_fit is not None:
                    d = data["calo"][charge_type]
                    d["fit_res"]["eta"].append(comb_fit[0] - truth_eta)
                    d["fit_res"]["phi"].append(_dphi(comb_fit[1], truth_phi))
                    d["fit_truth"]["eta"].append(truth_eta)
                    d["fit_truth"]["phi"].append(truth_phi)
                if pan_idx >= 0:
                    d = data["calo"][charge_type]
                    d["pan_res"]["eta"].append(float(eta_pan[pan_idx]) - truth_eta)
                    d["pan_res"]["phi"].append(_dphi(float(phi_pan[pan_idx]), truth_phi))
                    d["pan_truth"]["eta"].append(truth_eta)
                    d["pan_truth"]["phi"].append(truth_phi)

    print(f"\nProcessed {n_total} calo particles across {n_events} events.")
    print(f"  Pandora match rate : {100*(n_total-n_no_match)/max(n_total,1):.1f}%")
    print(f"  ECAL fit failures  : {n_ecal_fit_fail}")
    print(f"  HCAL fit failures  : {n_hcal_fit_fail}")

    # Convert lists to arrays
    for calo in ("ecal", "hcal", "calo"):
        for ct in ("charged", "neutral"):
            for key in ("fit_res", "pan_res", "fit_truth", "pan_truth"):
                for f in FIELDS:
                    data[calo][ct][key][f] = np.array(
                        data[calo][ct][key][f], dtype=np.float32
                    )
    for ct in ("charged", "neutral"):
        for key in edata[ct]:
            edata[ct][key] = np.array(edata[ct][key], dtype=np.float32)
    return data, edata


# ── plotting ────────────────────────────────────────────────────────────────

def _plot_vs_truth(
    ax: plt.Axes,
    series: dict,
    field_idx: int,
    bins: np.ndarray,
    xlabel: str,
    xscale: str,
    ylabel: str,
    stat: str,
    min_bin_count: int = 20,
) -> None:
    """Plot bias (median) or spread (IQR) of residual vs binned truth quantity."""
    n_bins = len(bins) - 1
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    half_bin    = 0.5 * (bins[1:] - bins[:-1])
    any_plotted = False

    for label, props in series.items():
        truth_arr = props["truth"][field_idx]
        data_arr  = props["data"][field_idx]
        if truth_arr is None or data_arr is None or data_arr.size < min_bin_count:
            continue

        bin_idx = np.clip(np.digitize(truth_arr, bins) - 1, 0, n_bins - 1)
        bs, ys, y_lo, y_hi = [], [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < min_bin_count:
                continue
            vals = data_arr[mask]
            if stat == "bias":
                ys.append(float(np.median(vals)))
                y_lo.append(float(np.percentile(vals, 25)))
                y_hi.append(float(np.percentile(vals, 75)))
            else:  # spread = IQR
                iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                ys.append(iqr)
                y_lo.append(iqr)
                y_hi.append(iqr)
            bs.append(b)

        if not bs:
            continue
        idx   = np.array(bs)
        xc    = bin_centres[idx]
        ys_a  = np.array(ys)
        ylo_a = np.array(y_lo)
        yhi_a = np.array(y_hi)
        ax.errorbar(
            xc, ys_a,
            xerr=half_bin[idx],
            yerr=[ys_a - ylo_a, yhi_a - ys_a] if stat == "bias" else None,
            fmt="o", color=props["color"], label=label,
            capsize=3, elinewidth=1.0, markersize=3, linestyle="none",
        )
        any_plotted = True

    if not any_plotted:
        ax.set_visible(False)
        return

    ax.axhline(0, color="grey", lw=0.5, ls=":", zorder=0)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(fontsize=6, framealpha=0.8)


def make_calo_fig(
    series: dict,
    title: str,
    min_bin_count: int = 20,
) -> plt.Figure:
    """1 row × 2 cols figure: bias (median ± IQR) vs truth η and φ."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for col, field in enumerate(FIELDS):
        bins, xlabel, xscale = TRUTH_BINS[field]
        _plot_vs_truth(axes[col], series, col, bins, xlabel, xscale,
                       f"Bias (median $\\pm$ IQR)\n{RESIDUAL_YLABEL[field]}",
                       "bias", min_bin_count)
        axes[col].set_title(f"Calo {field}", fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def make_energy_fig(series: dict, title: str, min_bin_count: int = 20) -> plt.Figure:
    """1 row × 2 cols figure: bias (median ± IQR) vs truth E and truth η.

    ``data[0]`` and ``data[1]`` are the same energy residual array; ``truth[0]``
    is truth energy (log-scale x) and ``truth[1]`` is truth η (linear x).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    col_specs = [
        (0, ENERGY_TRUTH_BINS_E,   r"Truth $E$ [GeV]", "log"),
        (1, ENERGY_TRUTH_BINS_ETA, r"Truth $\eta$",    "linear"),
    ]
    ylabel = f"Bias (median $\\pm$ IQR)\n{ENERGY_RESIDUAL_YLABEL}"

    for col, (field_idx, bins, xlabel, xscale) in enumerate(col_specs):
        _plot_vs_truth(axes[col], series, field_idx, bins,
                       xlabel, xscale, ylabel, "bias", min_bin_count)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = _cfg()
    data, edata = collect_residuals(cfg, n_events=N_EVENTS)

    out_dir = Path(__file__).resolve().parents[1] / "plots" / "calo"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── η/φ residual figures ───────────────────────────────────────────────
    for calo in ("ecal", "hcal"):
        calo_color = "mediumseagreen" if calo == "ecal" else "teal"
        calo_label = "ECAL fit" if calo == "ecal" else "HCAL fit"

        for charge_type in ("charged", "neutral"):
            d      = data[calo][charge_type]
            d_comb = data["calo"][charge_type]

            series: dict = {}
            if d["pan_res"]["eta"].size > 0:
                series["Pandora"] = {
                    "color": "cornflowerblue", "ls": "-",
                    "data":  [d["pan_res"][f]   for f in FIELDS],
                    "truth": [d["pan_truth"][f] for f in FIELDS],
                }
            if d["fit_res"]["eta"].size > 0:
                series[calo_label] = {
                    "color": calo_color, "ls": ":",
                    "data":  [d["fit_res"][f]   for f in FIELDS],
                    "truth": [d["fit_truth"][f] for f in FIELDS],
                }
            if d_comb["fit_res"]["eta"].size > 0:
                series["ECAL+HCAL fit"] = {
                    "color": "darkorange", "ls": "--",
                    "data":  [d_comb["fit_res"][f]   for f in FIELDS],
                    "truth": [d_comb["fit_truth"][f] for f in FIELDS],
                }

            if not series:
                print(f"No data for {calo}/{charge_type} — skipping.")
                continue

            calo_str   = calo.upper()
            charge_str = charge_type.capitalize()
            title = f"CLD {calo_str} {charge_str} — Pandora vs calo fit vs truth"
            fig   = make_calo_fig(series, title=title)

            fname = f"calo_{calo}_{charge_type}.png"
            path  = out_dir / fname
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved → {path}")

    # ── energy residual figures ────────────────────────────────────────────
    for charge_type in ("charged", "neutral"):
        ed = edata[charge_type]

        series_e: dict = {}
        if ed["truth_res"].size > 0:
            series_e["Truth calo hits"] = {
                "color": "mediumseagreen",
                "data":  [ed["truth_res"],       ed["truth_res"]],
                "truth": [ed["truth_truth_e"],   ed["truth_truth_eta"]],
            }
        if ed["pan_res"].size > 0:
            series_e["Pandora (calo sum)"] = {
                "color": "cornflowerblue",
                "data":  [ed["pan_res"],       ed["pan_res"]],
                "truth": [ed["pan_truth_e"],   ed["pan_truth_eta"]],
            }
        if ed["pan_reported_res"].size > 0:
            series_e["Pandora (reported)"] = {
                "color": "darkorange",
                "data":  [ed["pan_reported_res"],    ed["pan_reported_res"]],
                "truth": [ed["pan_rep_truth_e"],     ed["pan_rep_truth_eta"]],
            }
        if ed["truth_full_res"].size > 0:
            series_e["Truth (full hit)"] = {
                "color": "tab:purple",
                "data":  [ed["truth_full_res"],       ed["truth_full_res"]],
                "truth": [ed["truth_full_truth_e"],   ed["truth_full_truth_eta"]],
            }

        if not series_e:
            print(f"No energy data for {charge_type} — skipping.")
            continue

        charge_str = charge_type.capitalize()
        title_e = f"CLD Calo Energy {charge_str} — Pandora vs truth calo hits vs truth"
        fig_e   = make_energy_fig(series_e, title=title_e)

        fname_e = f"calo_energy_{charge_type}.png"
        path_e  = out_dir / fname_e
        fig_e.savefig(path_e, bbox_inches="tight")
        plt.close(fig_e)
        print(f"Saved → {path_e}")


if __name__ == "__main__":
    main()
