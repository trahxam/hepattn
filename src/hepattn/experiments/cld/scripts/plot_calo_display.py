"""CLD calorimeter cluster event display: truth / Pandora / calo fit.

Layout: 2 rows × N_PARTICLES columns
  Row 0: η-φ scatter (ECAL hits as circles, HCAL hits as squares, sized by energy)
  Row 1: parameter table — ECAL and HCAL sections separately

Usage (from repo root)::

    python src/hepattn/experiments/cld/scripts/plot_calo_display.py
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

plt.rcParams["figure.dpi"] = 300
plt.rcParams["text.usetex"] = True

# ── constants ──────────────────────────────────────────────────────────────
N_PARTICLES      = 6
N_EVENTS_SCAN    = 20
IOI_MATCH_THRESH = 0.5
MIN_CALO_HITS    = 2   # minimum ECAL + HCAL hits to include a particle


# ── helpers ────────────────────────────────────────────────────────────────

def _cfg() -> dict:
    p = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"
    return yaml.safe_load(p.read_text())["data"]


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _calo_fit(
    eta: np.ndarray, phi: np.ndarray, energy: np.ndarray
) -> tuple[float, float] | None:
    """Energy-weighted mean η and φ. Returns (eta, phi) or None if no energy."""
    total = float(energy.sum())
    if total < 1e-12:
        return None
    eta_m = float((energy * eta).sum() / total)
    sin_m = float((energy * np.sin(phi)).sum() / total)
    cos_m = float((energy * np.cos(phi)).sum() / total)
    return eta_m, float(np.arctan2(sin_m, cos_m))


def _unwrap_phi(phi: np.ndarray | float, phi_center: float) -> np.ndarray | float:
    """Shift phi values to the interval [phi_center - pi, phi_center + pi]."""
    arr = np.asarray(phi, dtype=float)
    delta = arr - phi_center
    return phi_center + (delta + np.pi) % (2 * np.pi) - np.pi


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


# ── particle selection ──────────────────────────────────────────────────────

ParticleRecord = dict


def _scan_particles(cfg: dict, n_particles: int, n_events: int) -> list[ParticleRecord]:
    """Scan events for particles with calorimeter deposits."""
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

    candidates: list[ParticleRecord] = []

    for inputs, targets in tqdm(loader, desc="Scanning", unit="ev", total=n_events):
        part_valid = targets["particle_valid"][0].bool()
        ecal_valid = targets["particle_ecal_valid"][0]  # (N_par, N_ecal)
        hcal_valid = targets["particle_hcal_valid"][0]  # (N_par, N_hcal)

        for p_idx in range(int(part_valid.shape[0])):
            if not part_valid[p_idx]:
                continue
            n_ecal = int(ecal_valid[p_idx].sum())
            n_hcal = int(hcal_valid[p_idx].sum())
            if n_ecal + n_hcal < MIN_CALO_HITS:
                continue
            candidates.append({
                "inputs": inputs, "targets": targets,
                "p_idx": p_idx, "n_ecal": n_ecal, "n_hcal": n_hcal,
            })
        if len(candidates) >= n_particles * 20:
            break

    if not candidates:
        return []

    # Spread picks across the sorted list to cover a range of hit multiplicities
    candidates.sort(key=lambda r: r["n_ecal"] + r["n_hcal"])
    if len(candidates) <= n_particles:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, n_particles, dtype=int)
    return [candidates[i] for i in picks]


# ── per-column panel drawing ─────────────────────────────────────────────────

def _draw_panel(
    ax_scatter: plt.Axes,
    ax_table: plt.Axes,
    rec: ParticleRecord,
    col: int,
) -> None:
    inputs  = rec["inputs"]
    targets = rec["targets"]
    p_idx   = rec["p_idx"]

    # ── truth direction ────────────────────────────────────────────────────
    eta_t      = float(_np(targets["particle_mom.eta"][0])[p_idx])
    phi_t      = float(_np(targets["particle_mom.phi"][0])[p_idx])
    is_charged = bool(_np(targets["particle_is_charged"][0])[p_idx])

    # ── ECAL hits ──────────────────────────────────────────────────────────
    ecal_mask   = _np(targets["particle_ecal_valid"][0][p_idx]).astype(bool)
    ecal_energy = _np(targets["particle_ecal_energy"][0][p_idx])
    ecal_eta    = _np(inputs["ecal_pos.eta"][0])
    ecal_phi    = _np(inputs["ecal_pos.phi"][0])

    ecal_energy_sel = ecal_energy[ecal_mask]
    ecal_eta_sel    = ecal_eta[ecal_mask]
    ecal_phi_sel    = ecal_phi[ecal_mask]

    # ── HCAL hits ──────────────────────────────────────────────────────────
    hcal_mask   = _np(targets["particle_hcal_valid"][0][p_idx]).astype(bool)
    hcal_energy = _np(targets["particle_hcal_energy"][0][p_idx])
    hcal_eta    = _np(inputs["hcal_pos.eta"][0])
    hcal_phi    = _np(inputs["hcal_pos.phi"][0])

    hcal_energy_sel = hcal_energy[hcal_mask]
    hcal_eta_sel    = hcal_eta[hcal_mask]
    hcal_phi_sel    = hcal_phi[hcal_mask]

    # ── phi display center (energy-weighted circular mean of all hits) ─────
    _all_phi_disp = np.concatenate([ecal_phi_sel, hcal_phi_sel]) if (len(ecal_phi_sel) + len(hcal_phi_sel)) > 0 else np.array([phi_t])
    _all_e_disp   = np.concatenate([ecal_energy_sel, hcal_energy_sel]) if (len(ecal_phi_sel) + len(hcal_phi_sel)) > 0 else np.array([1.0])
    _tot_e_disp   = float(_all_e_disp.sum())
    if _tot_e_disp > 1e-12:
        phi_center = float(np.arctan2(
            (_all_e_disp * np.sin(_all_phi_disp)).sum() / _tot_e_disp,
            (_all_e_disp * np.cos(_all_phi_disp)).sum() / _tot_e_disp,
        ))
    else:
        phi_center = phi_t
    ecal_phi_plot = _unwrap_phi(ecal_phi_sel, phi_center)
    hcal_phi_plot = _unwrap_phi(hcal_phi_sel, phi_center)
    phi_t_plot    = float(_unwrap_phi(phi_t, phi_center))

    # ── energy scale for marker sizes ─────────────────────────────────────
    all_e = np.concatenate([ecal_energy_sel, hcal_energy_sel]) if (len(ecal_energy_sel) + len(hcal_energy_sel)) > 0 else np.array([1.0])
    e_max = float(all_e.max()) if all_e.max() > 1e-12 else 1.0
    s_scale = 200.0

    # ── draw calorimeter hits ──────────────────────────────────────────────
    if len(ecal_eta_sel) > 0:
        ax_scatter.scatter(
            ecal_phi_plot, ecal_eta_sel,
            s=ecal_energy_sel / e_max * s_scale + 5,
            c="tab:orange", marker="o", alpha=0.7, zorder=3, linewidths=0,
        )
    if len(hcal_eta_sel) > 0:
        ax_scatter.scatter(
            hcal_phi_plot, hcal_eta_sel,
            s=hcal_energy_sel / e_max * s_scale + 5,
            c="tab:red", marker="s", alpha=0.6, zorder=3, linewidths=0,
        )

    # ── line fits ─────────────────────────────────────────────────────────
    ecal_fit = _calo_fit(ecal_eta_sel, ecal_phi_sel, ecal_energy_sel)
    hcal_fit = _calo_fit(hcal_eta_sel, hcal_phi_sel, hcal_energy_sel)

    # ── truth marker ──────────────────────────────────────────────────────
    ax_scatter.plot(phi_t_plot, eta_t, marker="*", color="tab:red", ms=12,
                    zorder=6, ls="none", markeredgewidth=0.5, markeredgecolor="white")

    # ── Pandora match ─────────────────────────────────────────────────────
    pan_valid    = targets["pandora_valid"][0].bool()
    pan_charged  = targets["pandora_is_charged"][0].bool()
    pan_is_type  = pan_charged if is_charged else ~pan_charged
    pan_idx = _match_pandora(
        targets["particle_ecal_valid"][0][p_idx],
        targets["particle_hcal_valid"][0][p_idx],
        targets["pandora_ecal_valid"][0],
        targets["pandora_hcal_valid"][0],
        pan_valid, pan_is_type,
    )

    eta_pan = phi_pan = phi_pan_plot = None
    if pan_idx >= 0:
        eta_pan = float(_np(targets["pandora_mom.eta"][0])[pan_idx])
        phi_pan = float(_np(targets["pandora_mom.phi"][0])[pan_idx])
        phi_pan_plot = float(_unwrap_phi(phi_pan, phi_center))
        ax_scatter.plot(phi_pan_plot, eta_pan, marker="D", color="cornflowerblue",
                        ms=9, zorder=6, ls="none",
                        markeredgewidth=0.5, markeredgecolor="white")

    # ── calo fit markers ──────────────────────────────────────────────────
    if ecal_fit is not None:
        ax_scatter.plot(float(_unwrap_phi(ecal_fit[1], phi_center)), ecal_fit[0],
                        marker="P", color="mediumseagreen",
                        ms=9, zorder=6, ls="none",
                        markeredgewidth=0.5, markeredgecolor="white")
    if hcal_fit is not None:
        ax_scatter.plot(float(_unwrap_phi(hcal_fit[1], phi_center)), hcal_fit[0],
                        marker="P", color="mediumvioletred",
                        ms=9, zorder=6, ls="none",
                        markeredgewidth=0.5, markeredgecolor="white")

    # ── scatter cosmetics ──────────────────────────────────────────────────
    charge_str = "charged" if is_charged else "neutral"
    ax_scatter.set_title(
        rf"P{col+1} ({charge_str})  $N_\mathrm{{ECAL}}={rec['n_ecal']}$  $N_\mathrm{{HCAL}}={rec['n_hcal']}$",
        fontsize=6.5,
    )
    ax_scatter.set_xlabel(r"$\phi$ [rad]", fontsize=7)
    ax_scatter.set_ylabel(r"$\eta$", fontsize=7)
    ax_scatter.tick_params(labelsize=6)
    ax_scatter.grid(True, alpha=0.2, lw=0.4)

    # ── parameter table ───────────────────────────────────────────────────
    def _f(v: float | None, fmt: str = "+.3f") -> str:
        return format(v, fmt) if v is not None else "---"

    pan_eta_str = _f(eta_pan) if pan_idx >= 0 else "---"
    pan_phi_str = _f(phi_pan) if pan_idx >= 0 else "---"

    rows = [
        # label, truth, pandora, ecal_fit
        ("ECAL", "", "", ""),   # section header
        (r"$\eta$",        _f(eta_t), pan_eta_str, _f(ecal_fit[0]) if ecal_fit else "---"),
        (r"$\phi$ [rad]",  _f(phi_t), pan_phi_str, _f(ecal_fit[1]) if ecal_fit else "---"),
        (r"$N_\mathrm{hits}$", str(rec["n_ecal"]), "---", "---"),
        ("HCAL", "", "", ""),   # section header
        (r"$\eta$",        _f(eta_t), pan_eta_str, _f(hcal_fit[0]) if hcal_fit else "---"),
        (r"$\phi$ [rad]",  _f(phi_t), pan_phi_str, _f(hcal_fit[1]) if hcal_fit else "---"),
        (r"$N_\mathrm{hits}$", str(rec["n_hcal"]), "---", "---"),
    ]
    col_labels = ["Param", "Truth", "Pandora", "Calo fit"]
    col_colors = ["#444",  "tab:red", "cornflowerblue", "mediumseagreen"]

    ax_table.axis("off")
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)

    n_r    = len(rows) + 1
    rh     = 1.0 / n_r
    pw     = 0.35
    n_c    = len(col_labels)
    vw     = (1.0 - pw) / (n_c - 1)
    col_xs = [0.0] + [pw + vw * i for i in range(n_c - 1)] + [1.0]

    def cx(j: int) -> float:
        return (col_xs[j] + col_xs[j + 1]) / 2

    fs = 4.5
    kw = dict(transform=ax_table.transAxes, clip_on=False)
    for yy, lw in [(1.0, 0.7), (1.0 - rh, 0.35), (0.0, 0.7)]:
        ax_table.plot([0, 1], [yy, yy], color="black", lw=lw, **kw)
    for j, (lab, clr) in enumerate(zip(col_labels, col_colors)):
        ax_table.text(cx(j), 1.0 - rh * 0.5, lab,
                      ha="center", va="center", fontsize=fs, fontweight="bold",
                      color=clr, **kw)

    for i, row in enumerate(rows):
        yc = 1.0 - rh * (i + 1.5)
        if row[1] == "" and row[2] == "" and row[3] == "":
            # Section divider
            ax_table.plot([0, 1], [yc + rh * 0.35, yc + rh * 0.35],
                          color="#ccc", lw=0.4, **kw)
            ax_table.text(col_xs[0] + 0.01, yc, row[0],
                          ha="left", va="center", fontsize=fs,
                          fontweight="bold", color="#666", **kw)
        else:
            ax_table.text(col_xs[0] + 0.01, yc, row[0],
                          ha="left", va="center", fontsize=fs, color="#444", **kw)
            for j in range(1, n_c):
                ax_table.text(col_xs[j + 1] - 0.01, yc, row[j],
                              ha="right", va="center", fontsize=fs,
                              color=col_colors[j], **kw)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg       = _cfg()
    particles = _scan_particles(cfg, n_particles=N_PARTICLES, n_events=N_EVENTS_SCAN)

    if not particles:
        print("No candidate particles found — check data path and config.")
        return

    n_cols   = len(particles)
    col_w    = 3.0
    h_ratios = [3, 2.2]
    fig_h    = col_w * sum(h_ratios) / h_ratios[0]

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(col_w * n_cols, fig_h),
        gridspec_kw={"height_ratios": h_ratios},
    )
    if n_cols == 1:
        axes = axes[:, None]

    for col, rec in enumerate(particles):
        _draw_panel(axes[0, col], axes[1, col], rec, col)

    legend_handles = [
        Line2D([0], [0], marker="*",  color="tab:red",       ms=8,  ls="none", label="Truth direction"),
        Line2D([0], [0], marker="D",  color="cornflowerblue", ms=6,  ls="none", label="Pandora"),
        Line2D([0], [0], marker="P",  color="mediumseagreen", ms=6,  ls="none", label="ECAL fit"),
        Line2D([0], [0], marker="P",  color="mediumvioletred", ms=6,  ls="none", label="HCAL fit"),
        Line2D([0], [0], marker="o",  color="tab:orange",     ms=6,  ls="none", label="ECAL hit"),
        Line2D([0], [0], marker="s",  color="tab:red",        ms=6,  ls="none", label="HCAL hit"),
    ]
    axes[0, 0].legend(handles=legend_handles, fontsize=5.0, framealpha=0.8,
                      loc="upper left", handlelength=1.5)

    fig.suptitle("CLD calorimeter display --- truth, Pandora, calo fit", fontsize=8)
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[1] / "plots" / "calo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cld_calo_display.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
