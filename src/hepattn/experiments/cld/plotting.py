"""Common residual-plotting utilities for the CLD tracking experiments.

Shared between ``scripts/plot_track_residuals.py`` (offline analysis) and any
future online callback that uploads residual plots per validation epoch.

Series format
-------------
All plotting functions accept a *series* dict::

    series = {
        "Pandora":   {"color": "tab:blue",   "ls": "-",  "data": [arr_field0, arr_field1, ...]},
        "Helix fit": {"color": "tab:green",  "ls": ":",  "data": [arr_field0, ...]},
        "Model":     {"color": "tab:orange", "ls": "--", "data": [arr_field0, ...]},
    }

where ``data[i]`` is a 1-D NumPy array of residual values for ``RESIDUALS[i]``, or
``None`` if the series does not cover that field.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------------------------
# Field metadata
# ---------------------------------------------------------------------------

# 6-element tuple per field:
# (track_field, particle_field, transform, xlabel, bins, angular)
RESIDUALS: list[tuple] = [
    (
        "pt",   "pt",
        lambda t, p: t - p,
        r"$p_T^\mathrm{pred} - p_T^\mathrm{true}$ [GeV]",
        np.linspace(-5.0, 5.0, 101),
        False,
    ),
    (
        "qopt", "qopt",
        lambda t, p: t - p,
        r"$q/p_T^\mathrm{pred} - q/p_T^\mathrm{true}$ [GeV$^{-1}$]",
        np.linspace(-2.0, 2.0, 101),
        False,
    ),
    (
        "eta",  "eta",
        lambda t, p: t - p,
        r"$\eta^\mathrm{pred} - \eta^\mathrm{true}$",
        np.linspace(-0.05, 0.05, 101),
        False,
    ),
    (
        "phi",  "phi",
        lambda t, p: np.arctan2(np.sin(t - p), np.cos(t - p)),
        r"$\phi^\mathrm{pred} - \phi^\mathrm{true}$ [rad]",
        np.linspace(-0.05, 0.05, 101),
        True,
    ),
    (
        "d0",   "d0",
        lambda t, p: t - p,
        r"$d_0^\mathrm{pred} - d_0^\mathrm{true}$ [mm]",
        np.linspace(-1.0, 1.0, 101),
        False,
    ),
    (
        "z0",   "z0",
        lambda t, p: t - p,
        r"$z_0^\mathrm{pred} - z_0^\mathrm{true}$ [mm]",
        np.linspace(-5.0, 5.0, 101),
        False,
    ),
]

# ---------------------------------------------------------------------------
# Truth-binned quantile plot metadata
# ---------------------------------------------------------------------------

TRUTH_BINS: dict[str, tuple] = {
    "pt":   (np.geomspace(0.1, 100.0, 24),   r"Truth $p_T$ [GeV]",              "log"),
    "qopt": (np.linspace(-10.0, 10.0, 24),   r"Truth $q/p_T$ [GeV$^{-1}$]",    "linear"),
    "eta":  (np.linspace(-3.0, 3.0, 24),     r"Truth $\eta$",                   "linear"),
    "phi":  (np.linspace(-np.pi, np.pi, 24), r"Truth $\phi$ [rad]",             "linear"),
    "d0":   (np.geomspace(1e-3, 1e2, 24),     r"Truth $|d_0|$ [mm]",             "log"),
    "z0":   (np.linspace(-30.0, 30.0, 24), r"Truth $z_0$ [mm]",               "linear"),
}

# Optional fixed y-limits for residual-vs-truth plots (bias, in data units).
# None → use automatic p2/p98 scaling.
RESIDUAL_VS_TRUTH_YLIM: dict[str, tuple | None] = {
    "pt":   None,
    "qopt": None,
    "eta":  None,
    "phi":  None,
    "d0":   (-1.0, 1.0),
    "z0":   (-2.0, 2.0),
}

# Y-limits for resolution-vs-truth plots (after RESOLUTION_DIVIDER normalisation).
RESOLUTION_VS_TRUTH_YLIM: dict[str, tuple | None] = {
    "pt":   None,
    "qopt": None,
    "eta":  None,
    "phi":  None,
    "d0":   (-1.0, 1.0),
    "z0":   None,
}

RESOLUTION_VS_TRUTH_YSCALE: dict[str, str] = {}

RESIDUAL_YLABEL: dict[str, str] = {
    "pt":   r"Median$[p_T^\mathrm{pred} - p_T^\mathrm{true}]$ [GeV]",
    "qopt": r"Median$[q/p_T^\mathrm{pred} - q/p_T^\mathrm{true}]$ [GeV$^{-1}$]",
    "eta":  r"Median$[\eta^\mathrm{pred} - \eta^\mathrm{true}]$",
    "phi":  r"Median$[\phi^\mathrm{pred} - \phi^\mathrm{true}]$ [rad]",
    "d0":   r"Median$[d_0^\mathrm{pred} - d_0^\mathrm{true}]$ [mm]",
    "z0":   r"Median$[z_0^\mathrm{pred} - z_0^\mathrm{true}]$ [mm]",
}

RESOLUTION_YLABEL: dict[str, str] = {
    "pt":   r"$\sigma[(p_T^\mathrm{pred} - p_T^\mathrm{true})\,/\,p_T^\mathrm{true}]$",
    "qopt": r"$\sigma[q/p_T^\mathrm{pred} - q/p_T^\mathrm{true}]$ [GeV$^{-1}$]",
    "eta":  r"$\sigma[\eta^\mathrm{pred} - \eta^\mathrm{true}]$",
    "phi":  r"$\sigma[\phi^\mathrm{pred} - \phi^\mathrm{true}]$ [rad]",
    "d0":   r"$\sigma[(d_0^\mathrm{pred} - d_0^\mathrm{true})\,/\,|d_0^\mathrm{true}|]$",
    "z0":   r"$\sigma[z_0^\mathrm{pred} - z_0^\mathrm{true}]$ [mm]",
}

# Per-field resolution normalisation: abs_residual / RESOLUTION_DIVIDER[field](truth).
# None → resolution = abs_residual (no normalisation).
RESOLUTION_DIVIDER: dict[str, object] = {
    "pt":   lambda truth: np.clip(truth, 1e-6, None),
    "qopt": None,
    "eta":  None,
    "phi":  None,
    "d0":   lambda truth: np.clip(np.abs(truth), 1e-4, None),
    "z0":   None,
}

# X-axis labels for arcsinh(residual / EMA-MAD) plots
ARCSINH_XLABEL: dict[str, str] = {
    "pt":   r"$\mathrm{arcsinh}[(p_T^\mathrm{pred}-p_T^\mathrm{true})/\mathrm{MAD}]$",
    "qopt": r"$\mathrm{arcsinh}[((q/p_T)^\mathrm{pred}-(q/p_T)^\mathrm{true})/\mathrm{MAD}]$",
    "eta":  r"$\mathrm{arcsinh}[(\eta^\mathrm{pred}-\eta^\mathrm{true})/\mathrm{MAD}]$",
    "phi":  r"$\mathrm{arcsinh}[(\phi^\mathrm{pred}-\phi^\mathrm{true})/\mathrm{MAD}]$",
    "d0":   r"$\mathrm{arcsinh}[(d_0^\mathrm{pred}-d_0^\mathrm{true})/\mathrm{MAD}]$",
    "z0":   r"$\mathrm{arcsinh}[(z_0^\mathrm{pred}-z_0^\mathrm{true})/\mathrm{MAD}]$",
}


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def compute_mad(values: np.ndarray) -> float:
    """Median absolute deviation."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))))


def compute_fwhm(counts: np.ndarray, bins: np.ndarray) -> float:
    """Empirical FWHM from a histogram.  Returns NaN if undetermined."""
    if counts.max() == 0:
        return float("nan")
    half_max = counts.max() / 2.0
    above    = counts >= half_max
    if not above.any():
        return float("nan")
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    left_idx    = int(np.argmax(above))
    right_idx   = int(len(above) - 1 - np.argmax(above[::-1]))
    return float(bin_centres[right_idx] - bin_centres[left_idx])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stats_text(arr: np.ndarray, extra: str = "") -> str:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return "N = 0"
    mean = float(np.mean(v))
    std  = float(np.std(v))
    mad  = compute_mad(v)
    iqr  = float(np.percentile(v, 75) - np.percentile(v, 25))
    return f"{extra}$\\mu$={mean:.4f}\n$\\sigma$={std:.4f}\nMAD={mad:.4f}\nIQR={iqr:.4f}\nN={v.size}"


def _n_cols(n_fields: int) -> int:
    return (n_fields + 1) // 2


def _figure_grid(n_fields: int) -> tuple:
    ncols = _n_cols(n_fields)
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
    n_fields = len(RESIDUALS)
    fig, axes = _figure_grid(n_fields)

    for i, (field, _pfield, _transform, xlabel, bins, _angular) in enumerate(RESIDUALS):
        ax = axes[i]
        plotted = False
        y_pos = 0.97

        for label, props in series.items():
            arr = props["data"][i]
            if arr is None or arr.size == 0:
                continue
            counts = np.histogram(np.clip(arr, bins[0], bins[-1]), bins=bins)[0].astype(float)
            ax.step(bins[:-1], counts, where="post",
                    color=props["color"], ls=props.get("ls", "-"),
                    linewidth=1.2, label=label)
            ax.text(0.97, y_pos, _stats_text(arr, f"[{label}]\n"),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=props["color"])
            y_pos -= 0.33
            plotted = True

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Tracks", fontsize=8)
        ax.set_yscale("log")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {field} residual", fontsize=8)
        if plotted:
            ax.legend(fontsize=6, framealpha=0.8)

    for j in range(n_fields, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def make_residual_fullrange_fig(
    series: dict[str, dict],
    suptitle: str = "Track Residuals (full range)",
) -> plt.Figure:
    """Auto-scaled residual histogram (0.1–99.9th percentile range)."""
    n_fields = len(RESIDUALS)
    fig, axes = _figure_grid(n_fields)

    for i, (field, _pfield, _transform, xlabel, _bins, _angular) in enumerate(RESIDUALS):
        ax = axes[i]

        all_vals = np.concatenate(
            [props["data"][i] for props in series.values()
             if props["data"][i] is not None and props["data"][i].size > 0]
        ) if any(props["data"][i] is not None and props["data"][i].size > 0
                 for props in series.values()) else np.array([])

        if all_vals.size == 0:
            ax.set_visible(False)
            continue

        r_min  = float(np.nanpercentile(all_vals, 0.1))
        r_max  = float(np.nanpercentile(all_vals, 99.9))
        bins   = np.linspace(r_min, r_max, 101)
        plotted = False
        y_pos   = 0.97

        for label, props in series.items():
            arr = props["data"][i]
            if arr is None or arr.size == 0:
                continue
            counts = np.histogram(arr, bins=bins)[0].astype(float)
            ax.step(bins[:-1], counts, where="post",
                    color=props["color"], ls=props.get("ls", "-"),
                    linewidth=1.2, label=label)
            ax.text(0.97, y_pos, _stats_text(arr, f"[{label}]\n"),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=props["color"])
            y_pos -= 0.33
            plotted = True

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Tracks", fontsize=8)
        ax.set_yscale("log")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {field} residual (full range)", fontsize=8)
        if plotted:
            ax.legend(fontsize=6, framealpha=0.8)

    for j in range(n_fields, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def make_arcsinh_fig(
    series: dict[str, dict],
    suptitle: str = "Track Residuals — arcsinh(residual / MAD)",
    n_bins: int = 80,
    ema_scales: dict[str, float] | None = None,
) -> plt.Figure:
    """arcsinh(residual / scale) density histogram.

    Each series is normalised by its own MAD unless ``ema_scales`` is provided,
    in which case all series share that scale for direct comparison.
    """
    n_fields = len(RESIDUALS)
    fig, axes = _figure_grid(n_fields)

    for i, (field, _pfield, _transform, _xlabel, _bins, _angular) in enumerate(RESIDUALS):
        ax = axes[i]

        shared_scale = None
        if ema_scales and field in ema_scales and ema_scales[field] > 0:
            shared_scale = ema_scales[field]
        else:
            first_arr = next(
                (props["data"][i] for props in series.values()
                 if props["data"][i] is not None and props["data"][i].size > 1),
                None,
            )
            if first_arr is not None:
                mad = compute_mad(first_arr)
                if mad > 0:
                    shared_scale = mad

        if shared_scale is None:
            ax.set_visible(False)
            continue

        all_t = []
        for props in series.values():
            arr = props["data"][i]
            if arr is not None and arr.size > 1:
                all_t.append(np.arcsinh(arr / shared_scale))

        if not all_t:
            ax.set_visible(False)
            continue

        combined = np.concatenate(all_t)
        r_min = float(np.nanpercentile(combined, 0.5))
        r_max = float(np.nanpercentile(combined, 99.5))
        bins_t = np.linspace(r_min, r_max, n_bins + 1)
        width  = bins_t[1] - bins_t[0]
        plotted = False
        y_pos   = 0.97

        for label, props in series.items():
            arr = props["data"][i]
            if arr is None or arr.size < 2:
                continue
            t = np.arcsinh(arr / shared_scale)
            counts = np.histogram(t, bins=bins_t)[0].astype(float)
            density = counts / (counts.sum() * width) if counts.sum() > 0 else counts
            ax.step(bins_t[:-1], density, where="post",
                    color=props["color"], ls=props.get("ls", "-"),
                    linewidth=1.2, label=label)
            mad_v = compute_mad(arr)
            iqr_v = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            ax.text(0.97, y_pos,
                    f"[{label}]\nMAD={mad_v:.4f}\nIQR={iqr_v:.4f}\n$\\mu$={float(np.mean(t)):.3f}\n$\\sigma$={float(np.std(t)):.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=props["color"])
            y_pos -= 0.35
            plotted = True

        xlabel = ARCSINH_XLABEL.get(field, r"$\mathrm{arcsinh}(\Delta/\mathrm{MAD})$")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {field} arcsinh residual (scale={shared_scale:.4g})", fontsize=8)
        if plotted:
            ax.legend(fontsize=6, framealpha=0.8)

    for j in range(n_fields, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def make_residual_vs_truth_fig(
    series: dict[str, dict],
    ylabel_map: dict[str, str],
    suptitle: str = "Track Residuals vs Truth",
    min_bin_count: int = 5,
    ylim_map: dict[str, tuple | None] | None = None,
    yscale_map: dict[str, str] | None = None,
) -> plt.Figure:
    """Median residual ± IQR bars vs binned truth quantity, using TRUTH_BINS.

    Parameters
    ----------
    series:
        Dict mapping series label → props dict with keys:
            ``"color"``  — matplotlib colour string
            ``"truth"``  — list[np.ndarray | None], truth values per RESIDUALS[i]
            ``"data"``   — list[np.ndarray | None], residual/resolution values per RESIDUALS[i]
        ``truth[i]`` and ``data[i]`` must be aligned arrays of the same length.
        For d0 the x-bins are applied to ``|truth|``.
    ylabel_map:
        Maps track_field → y-axis label (use RESIDUAL_YLABEL or RESOLUTION_YLABEL).
    min_bin_count:
        Minimum entries in a truth bin to plot a point.
    """
    n_fields = len(RESIDUALS)
    fig, axes = _figure_grid(n_fields)

    for i, (_tf, particle_field, *_) in enumerate(RESIDUALS):
        track_field = RESIDUALS[i][0]
        ax = axes[i]

        if particle_field not in TRUTH_BINS:
            ax.set_visible(False)
            continue

        truth_bin_edges, x_label, x_scale = TRUTH_BINS[particle_field]
        n_bins = len(truth_bin_edges) - 1
        bin_centres = (
            np.sqrt(truth_bin_edges[:-1] * truth_bin_edges[1:])
            if x_scale == "log"
            else 0.5 * (truth_bin_edges[:-1] + truth_bin_edges[1:])
        )

        any_plotted = False
        all_ys: list[float] = []

        for label, props in series.items():
            truth_arr = (props.get("truth") or [None] * n_fields)[i]
            data_arr  = (props.get("data")  or [None] * n_fields)[i]
            if truth_arr is None or data_arr is None or truth_arr.size < min_bin_count:
                continue

            bin_vals = np.abs(truth_arr) if particle_field == "d0" else truth_arr
            bin_idx  = np.clip(np.digitize(bin_vals, truth_bin_edges) - 1, 0, n_bins - 1)

            bs, meds, lqs, uqs = [], [], [], []
            for b in range(n_bins):
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
            xe = (
                [xc - truth_bin_edges[idx], truth_bin_edges[idx + 1] - xc]
                if x_scale == "log"
                else 0.5 * (truth_bin_edges[1:] - truth_bin_edges[:-1])[idx]
            )
            ax.errorbar(
                xc, meds_a,
                xerr=xe, yerr=[meds_a - lqs_a, uqs_a - meds_a],
                fmt="o", color=props["color"], label=label,
                capsize=3, elinewidth=1.0, markersize=3, linestyle="none",
            )
            any_plotted = True
            all_ys.extend(meds_a.tolist() + lqs_a.tolist() + uqs_a.tolist())

        if not any_plotted:
            ax.set_visible(False)
            continue

        _ylim_map = ylim_map if ylim_map is not None else RESIDUAL_VS_TRUTH_YLIM
        fixed_ylim = _ylim_map.get(track_field)
        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
        else:
            all_y = np.array(all_ys)
            if all_y.size > 3:
                p2, p98 = np.percentile(all_y, [2, 98])
                margin  = max((p98 - p2) * 0.15, 1e-9)
                ax.set_ylim(p2 - margin, p98 + margin)

        y_scale = (yscale_map or {}).get(track_field, "linear")
        ax.set_yscale(y_scale)
        if y_scale != "log":
            ax.axhline(0, color="grey", lw=0.5, ls=":", zorder=0)
        if x_scale == "log":
            ax.set_xscale("log")
        ax.set_xlim(truth_bin_edges[0], truth_bin_edges[-1])
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(ylabel_map.get(track_field, ""), fontsize=8)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_title(f"Track {track_field}", fontsize=8)
        ax.legend(fontsize=6, framealpha=0.8)

    for j in range(n_fields, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Render *fig* to a PNG byte buffer and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
