"""CLD track-fitting residual plots.

Shared between ``scripts/plot_track_residuals.py`` (offline) and
``TrackResidualPlotCallback`` (per-validation-epoch Comet uploads).

Adapted from the colliderml equivalent: ACTS → Pandora, pt removed.
Units: d0/z0 inputs to plotting functions are in mm (callback handles m→mm).
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 150

# (track_field, particle_field, transform, xlabel, fixed_bins, angular)
# track_field: label + key in _PANDORA_TARGET_KEY
# particle_field: key in _TRUTH_TARGET_KEY
# All inputs to transform are in display units (mm for d0/z0).
RESIDUALS: list[tuple] = [
    (
        "theta", "theta",
        lambda t, p: t - p,
        r"$\theta^\mathrm{pred} - \theta^\mathrm{true}$ [rad]",
        np.linspace(-0.05, 0.05, 101),
        False,
    ),
    (
        "phi", "phi_perigee",
        lambda t, p: np.arctan2(np.sin(t - p), np.cos(t - p)),
        r"$\phi^\mathrm{pred} - \phi^\mathrm{true}$ [rad]",
        np.linspace(-0.05, 0.05, 101),
        True,
    ),
    (
        "qopt", "qopt",
        lambda t, p: t - p,
        r"$(q/p_T)^\mathrm{pred} - (q/p_T)^\mathrm{true}$ [GeV$^{-1}$]",
        np.linspace(-2.0, 2.0, 101),
        False,
    ),
    (
        "d0", "d0_perigee_m",
        lambda t, p: t - p,
        r"$d_0^\mathrm{pred} - d_0^\mathrm{true}$ [mm]",
        np.linspace(-1.0, 1.0, 101),
        False,
    ),
    (
        "z0", "z0_perigee_m",
        lambda t, p: t - p,
        r"$z_0^\mathrm{pred} - z_0^\mathrm{true}$ [mm]",
        np.linspace(-5.0, 5.0, 101),
        False,
    ),
]

TRUTH_BINS: dict[str, tuple] = {
    "theta":       (np.linspace(0.036, 3.106, 24),        r"Truth $\theta$ [rad]",           "linear"),
    "phi_perigee": (np.linspace(-np.pi, np.pi, 24),       r"Truth $\phi$ perigee [rad]",     "linear"),
    "qopt":        (np.linspace(-1.0, 1.0, 24),           r"Truth $q/p_T$ [GeV$^{-1}$]",    "linear"),
    "d0_perigee_m":(np.geomspace(0.001, 10.0, 24),        r"Truth $|d_0|$ perigee [mm]",     "log"),
    "z0_perigee_m":(np.linspace(-100.0, 100.0, 24),       r"Truth $z_0$ perigee [mm]",       "linear"),
}

RESIDUAL_YLABEL: dict[str, str] = {
    "theta": r"Median$[\theta^\mathrm{pred} - \theta^\mathrm{true}]$ [rad]",
    "phi":   r"Median$[\phi^\mathrm{pred} - \phi^\mathrm{true}]$ [rad]",
    "qopt":  r"Median$[(q/p_T)^\mathrm{pred} - (q/p_T)^\mathrm{true}]$ [GeV$^{-1}$]",
    "d0":    r"Median$[d_0^\mathrm{pred} - d_0^\mathrm{true}]$ [mm]",
    "z0":    r"Median$[z_0^\mathrm{pred} - z_0^\mathrm{true}]$ [mm]",
}

RESOLUTION_YLABEL: dict[str, str] = {
    "theta": r"$\sigma[\theta^\mathrm{pred} - \theta^\mathrm{true}]$ [rad]",
    "phi":   r"$\sigma[\phi^\mathrm{pred} - \phi^\mathrm{true}]$ [rad]",
    "qopt":  r"$\sigma[(q/p_T)^\mathrm{pred} - (q/p_T)^\mathrm{true}]$ [GeV$^{-1}$]",
    "d0":    r"$\sigma[(d_0^\mathrm{pred} - d_0^\mathrm{true})\,/\,|d_0^\mathrm{true}|]$",
    "z0":    r"$\sigma[(z_0^\mathrm{pred} - z_0^\mathrm{true})\,/\,|z_0^\mathrm{true}|]$",
}

ARCSINH_XLABEL: dict[str, str] = {
    "theta": r"$\mathrm{arcsinh}[(\theta^\mathrm{pred}-\theta^\mathrm{true})/\mathrm{MAD}]$",
    "phi":   r"$\mathrm{arcsinh}[(\phi^\mathrm{pred}-\phi^\mathrm{true})/\mathrm{MAD}]$",
    "qopt":  r"$\mathrm{arcsinh}[((q/p_T)^\mathrm{pred}-(q/p_T)^\mathrm{true})/\mathrm{MAD}]$",
    "d0":    r"$\mathrm{arcsinh}[(d_0^\mathrm{pred}-d_0^\mathrm{true})/\mathrm{MAD}]$",
    "z0":    r"$\mathrm{arcsinh}[(z_0^\mathrm{pred}-z_0^\mathrm{true})/\mathrm{MAD}]$",
}

# Resolution normalisation: None → no normalisation (show absolute residual).
RESOLUTION_DIVIDER: dict[str, object] = {
    "theta": None,
    "phi":   None,
    "qopt": None,
    "d0":   lambda truth: np.clip(np.abs(truth), 1e-4, None),
    "z0":   lambda truth: np.clip(np.abs(truth), 0.1, None),
}


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def compute_mad(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))))


def compute_fwhm(counts: np.ndarray, bins: np.ndarray) -> float:
    if counts.max() == 0:
        return float("nan")
    half_max = counts.max() / 2.0
    above = counts >= half_max
    if not above.any():
        return float("nan")
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    left_idx  = int(np.argmax(above))
    right_idx = int(len(above) - 1 - np.argmax(above[::-1]))
    return float(bin_centres[right_idx] - bin_centres[left_idx])


def _stats_text(arr: np.ndarray) -> str:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return "N = 0"
    return (
        f"$\\mu$={np.mean(v):.4f}\n"
        f"$\\sigma$={np.std(v):.4f}\n"
        f"MAD={compute_mad(v):.4f}\n"
        f"IQR={float(np.percentile(v, 75) - np.percentile(v, 25)):.4f}\n"
        f"N={v.size}"
    )


def _figure_grid(n_fields: int) -> tuple:
    ncols = (n_fields + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7))
    return fig, axes.flatten()


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def make_residual_fig(
    series: dict[str, dict],
    suptitle: str = "Track Residuals",
) -> plt.Figure:
    """Fixed-range residual histogram, one panel per field."""
    n = len(RESIDUALS)
    fig, axes = _figure_grid(n)
    for i, (field, _pf, _tr, xlabel, bins, _ang) in enumerate(RESIDUALS):
        ax = axes[i]
        y_pos = 0.97
        plotted = False
        for label, props in series.items():
            arr = props["data"][i]
            if arr is None or arr.size < 2:
                continue
            counts, _ = np.histogram(arr, bins=bins)
            density = counts / (counts.sum() * (bins[1] - bins[0])) if counts.sum() > 0 else counts.astype(float)
            ax.step(bins[:-1], density, where="post",
                    color=props["color"], ls=props.get("ls", "-"), linewidth=1.2, label=label)
            ax.text(0.97, y_pos, f"[{label}]\n{_stats_text(arr)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color=props["color"])
            y_pos -= 0.30
            plotted = True
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {field}", fontsize=8)
        if plotted:
            ax.legend(fontsize=6, framealpha=0.8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def make_residual_fullrange_fig(
    series: dict[str, dict],
    suptitle: str = "Track Residuals (full range)",
    n_bins: int = 100,
) -> plt.Figure:
    """Auto-scaled residual histogram."""
    n = len(RESIDUALS)
    fig, axes = _figure_grid(n)
    for i, (field, _pf, _tr, xlabel, _fb, _ang) in enumerate(RESIDUALS):
        ax = axes[i]
        all_data = [props["data"][i] for props in series.values()
                    if props["data"][i] is not None and props["data"][i].size > 1]
        if not all_data:
            ax.set_visible(False)
            continue
        combined = np.concatenate(all_data)
        combined = combined[np.isfinite(combined)]
        if combined.size < 2:
            ax.set_visible(False)
            continue
        lo = float(np.nanpercentile(combined, 0.5))
        hi = float(np.nanpercentile(combined, 99.5))
        if lo >= hi:
            ax.set_visible(False)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        y_pos = 0.97
        for label, props in series.items():
            arr = props["data"][i]
            if arr is None or arr.size < 2:
                continue
            counts, _ = np.histogram(arr, bins=bins)
            w = bins[1] - bins[0]
            density = counts / (counts.sum() * w) if counts.sum() > 0 else counts.astype(float)
            ax.step(bins[:-1], density, where="post",
                    color=props["color"], ls=props.get("ls", "-"), linewidth=1.2, label=label)
            ax.text(0.97, y_pos, f"[{label}]\n{_stats_text(arr)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6, color=props["color"])
            y_pos -= 0.30
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {field} (full range)", fontsize=8)
        ax.legend(fontsize=6, framealpha=0.8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def make_arcsinh_fig(
    series: dict[str, dict],
    suptitle: str = "Track Residuals — arcsinh scaled",
    n_bins: int = 100,
    ema_scales: dict[str, float] | None = None,
) -> plt.Figure:
    """arcsinh(residual / MAD) density plot."""
    n = len(RESIDUALS)
    fig, axes = _figure_grid(n)
    for i, (field, _pf, _tr, _xl, _fb, _ang) in enumerate(RESIDUALS):
        ax = axes[i]
        scale = None
        if ema_scales and field in ema_scales and ema_scales[field] > 0:
            scale = ema_scales[field]
        else:
            first = next(
                (props["data"][i] for props in series.values()
                 if props["data"][i] is not None and props["data"][i].size > 1), None)
            if first is not None:
                m = compute_mad(first)
                if m > 0:
                    scale = m
        if scale is None:
            ax.set_visible(False)
            continue
        all_t = [np.arcsinh(props["data"][i] / scale)
                 for props in series.values()
                 if props["data"][i] is not None and props["data"][i].size > 1]
        if not all_t:
            ax.set_visible(False)
            continue
        combined = np.concatenate(all_t)
        lo = float(np.nanpercentile(combined, 0.5))
        hi = float(np.nanpercentile(combined, 99.5))
        bins = np.linspace(lo, hi, n_bins + 1)
        w = bins[1] - bins[0]
        y_pos = 0.97
        for label, props in series.items():
            arr = props["data"][i]
            if arr is None or arr.size < 2:
                continue
            t = np.arcsinh(arr / scale)
            counts, _ = np.histogram(t, bins=bins)
            density = counts / (counts.sum() * w) if counts.sum() > 0 else counts.astype(float)
            ax.step(bins[:-1], density, where="post",
                    color=props["color"], ls=props.get("ls", "-"), linewidth=1.2, label=label)
            mad_v = compute_mad(arr)
            iqr_v = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            ax.text(0.97, y_pos,
                    f"[{label}]\nMAD={mad_v:.4f}\nIQR={iqr_v:.4f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=props["color"])
            y_pos -= 0.22
        xlabel = ARCSINH_XLABEL.get(field, r"$\mathrm{arcsinh}(\Delta/\mathrm{MAD})$")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {field} arcsinh (scale={scale:.4g})", fontsize=8)
        ax.legend(fontsize=6, framealpha=0.8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def make_residual_vs_truth_fig(
    series: dict[str, dict],
    ylabel_map: dict[str, str],
    suptitle: str = "Track Residuals vs Truth",
    min_bin_count: int = 5,
) -> plt.Figure:
    """Median residual ± IQR vs binned truth quantity."""
    n = len(RESIDUALS)
    fig, axes = _figure_grid(n)
    for i, (_tf, particle_field, *_) in enumerate(RESIDUALS):
        track_field = RESIDUALS[i][0]
        ax = axes[i]
        if particle_field not in TRUTH_BINS:
            ax.set_visible(False)
            continue
        truth_bin_edges, x_label, x_scale = TRUTH_BINS[particle_field]
        n_b = len(truth_bin_edges) - 1
        bin_centres = (
            np.sqrt(truth_bin_edges[:-1] * truth_bin_edges[1:])
            if x_scale == "log"
            else 0.5 * (truth_bin_edges[:-1] + truth_bin_edges[1:])
        )
        any_plotted = False
        all_ys: list[float] = []
        for label, props in series.items():
            truth_arr = (props.get("truth") or [None] * n)[i]
            data_arr  = (props.get("data")  or [None] * n)[i]
            if truth_arr is None or data_arr is None or truth_arr.size < min_bin_count:
                continue
            bin_vals = np.abs(truth_arr) if particle_field in ("d0_perigee_m",) else truth_arr
            bin_idx  = np.clip(np.digitize(bin_vals, truth_bin_edges) - 1, 0, n_b - 1)
            bs, meds, lqs, uqs = [], [], [], []
            for b in range(n_b):
                mask_b = bin_idx == b
                if mask_b.sum() < min_bin_count:
                    continue
                vals = data_arr[mask_b]
                bs.append(b)
                lqs.append(float(np.percentile(vals, 25)))
                meds.append(float(np.percentile(vals, 50)))
                uqs.append(float(np.percentile(vals, 75)))
            if not bs:
                continue
            idx    = np.array(bs)
            xc     = bin_centres[idx]
            meds_a = np.array(meds)
            lqs_a  = np.array(lqs)
            uqs_a  = np.array(uqs)
            ax.errorbar(xc, meds_a,
                        yerr=[meds_a - lqs_a, uqs_a - meds_a],
                        fmt="o", color=props["color"], label=label,
                        capsize=3, elinewidth=1.0, markersize=3, linestyle="none")
            any_plotted = True
            all_ys.extend(meds_a.tolist() + lqs_a.tolist() + uqs_a.tolist())
        if not any_plotted:
            ax.set_visible(False)
            continue
        all_y = np.array(all_ys)
        if all_y.size > 3:
            p2, p98 = np.percentile(all_y, [2, 98])
            margin  = max((p98 - p2) * 0.15, 1e-9)
            ax.set_ylim(p2 - margin, p98 + margin)
        ax.axhline(0, color="grey", lw=0.5, ls=":", zorder=0)
        if x_scale == "log":
            ax.set_xscale("log")
        ax.set_xlim(truth_bin_edges[0], truth_bin_edges[-1])
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(ylabel_map.get(track_field, ""), fontsize=8)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {track_field}", fontsize=8)
        ax.legend(fontsize=6, framealpha=0.8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
