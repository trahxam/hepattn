# ruff: noqa: ARG001, PTH123, BLE001, S110
import json
import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from atlasify import atlasify
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from tqdm import tqdm

from reconstruct_anything.utils.plotting import plot_hist_to_ax
from reconstruct_anything.utils.stats import bayesian_binomial_error

# ── Plot style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 300,
    "font.size": 16,
    "figure.constrained_layout.use": True,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 18,
})

LEGEND_FONTSIZE = 14
SUB_FONTSIZE = 16

# ── Configuration ───────────────────────────────────────────────���─────────────

EVAL_PATH = Path("logs/TIDE_32trk_F32_pix_only_20260408-T193712/ckpts/epoch=009-train_loss=0.63014_test_eval.h5")
DATA_DIR = Path("/share/rcifdata/maxhart/data/ambi/test/")
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots" / "eval"

MAX_ROIS = None
PIX_ONLY = True
NUM_QUERIES = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)

# Working points: name -> (threshold, display label)
WORKING_POINTS = {
    "dm": (0.50, "Double Majority"),
    "lhc": (0.75, "LHC"),
    "perf": (0.99, "Perfect"),
}

PRED_NAMES = ["sudo", "sisp", "reco", "pred"]
COLORS = {"sudo": "black", "sisp": "red", "reco": "blue", "pred": "purple"}
NAME_ALIASES = {
    "sudo": "Particles",
    "sisp": "SiSp Tracks",
    "reco": "Reco Tracks",
    "pred": "TIDE Tracks",
}

# Quantities binned over true particles (for efficiency plots)
TRUE_QTYS = [
    ("pt", r"Particle $p_\mathrm{T}$ [GeV]", "log", np.geomspace(2, 3.5e3, 32)),
    ("bhad_pt", r"Particle b-hadron $p_\mathrm{T}$ [GeV]", "log", np.geomspace(150, 5e3, 32)),
    ("eta", r"Particle $\eta$", "linear", np.linspace(-2.25, 2.25, 32)),
    ("phi", r"Particle $\phi$", "linear", np.linspace(-np.pi, np.pi, 32)),
    ("deta", r"Particle - RoI Axis $\Delta \eta$", "linear", np.linspace(-0.05, 0.05, 32)),
    ("dphi", r"Particle - RoI Axis $\Delta \phi$", "linear", np.linspace(-0.05, 0.05, 32)),
]

# Quantities binned over pred tracks (for purity/fake/dup plots)
PRED_QTYS = [
    ("phi", r"RoI $\phi$", "linear", np.linspace(-np.pi, np.pi, 32)),
    ("energy", r"RoI Energy [GeV]", "log", np.geomspace(150, 5e3, 32)),
]

# ROI-level quantities (scalar per ROI, binned for both eff and pur plots)
ROI_QTYS = [
    ("n_pix_hits", r"Number of Pixel Hits in RoI", "linear", np.arange(0, 106, 5), False),
    ("n_particles", r"Number of Particles in RoI", "linear", np.arange(0.5, 23.5, 1), True),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_csr(file, roi_id, key):
    """Load a CSR sparse matrix from an h5 group and return as dense bool array."""
    g = file[roi_id]
    data = g[f"{key}_data"][:]
    indices = g[f"{key}_indices"][:]
    indptr = g[f"{key}_indptr"][:]
    shape = tuple(g[f"{key}_shape"][:])
    return np.array(csr_matrix((data, indices, indptr), shape, dtype=bool).todense())


def build_roi_index(data_dir):
    """Map roi_id -> h5 file path using the pre-built manifest."""
    manifest_path = Path(data_dir) / ".manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    roi_to_file = {}
    for file_path, roi_ids in manifest.items():
        for roi_id in roi_ids:
            roi_to_file[roi_id] = file_path
    return roi_to_file


def pad_to_shape(arr, target_rows, target_cols):
    """Pad a 2D boolean array to (target_rows, target_cols), trimming if larger."""
    out = np.zeros((target_rows, target_cols), dtype=bool)
    r = min(arr.shape[0], target_rows)
    c = min(arr.shape[1], target_cols)
    out[:r, :c] = arr[:r, :c]
    return out


def compute_matching(true_pix, pred_pix, true_sct, pred_sct, true_valid, pred_valid):
    """Compute hit efficiency/purity and optimal true-pred assignment.

    Returns:
    -------
    hit_eff : (N_true, N_pred) fraction of true particle's hits found in pred track
    hit_pur : (N_true, N_pred) fraction of pred track's hits from true particle
    true_idx, pred_idx : optimal assignment indices from Hungarian algorithm
    """
    pix_tp = np.einsum("nc,mc->nm", true_pix.astype(float), pred_pix.astype(float))
    num_true_hits = true_pix.sum(-1).astype(float)
    num_pred_hits = pred_pix.sum(-1).astype(float)

    if true_sct.any() or pred_sct.any():
        sct_tp = np.einsum("nc,mc->nm", true_sct.astype(float), pred_sct.astype(float))
        tp = pix_tp + sct_tp
        num_true_hits = num_true_hits + true_sct.sum(-1).astype(float)
        num_pred_hits = num_pred_hits + pred_sct.sum(-1).astype(float)
    else:
        tp = pix_tp

    hit_eff = tp / np.maximum(num_true_hits[:, None], 1)
    hit_pur = tp / np.maximum(num_pred_hits[None, :], 1)

    union = num_true_hits[:, None] + num_pred_hits[None, :] - tp
    iou = tp / np.maximum(union, 1)
    iou[~true_valid, :] = 0.0
    iou[:, ~pred_valid] = 0.0

    true_idx, pred_idx = linear_sum_assignment(iou, maximize=True)
    return hit_eff, hit_pur, true_idx, pred_idx


def classify_matches(hit_eff, hit_pur, true_valid, pred_valid, true_idx, pred_idx, threshold):
    """Classify true particles and pred tracks given a working-point threshold.

    A pair is matched iff hit_eff >= threshold AND hit_pur >= threshold.
    """
    hit_eff_r = hit_eff[:, pred_idx]
    hit_pur_r = hit_pur[:, pred_idx]

    match_matrix = (hit_eff_r >= threshold) & (hit_pur_r >= threshold)

    paired_match = match_matrix[true_idx, np.arange(len(pred_idx))]
    any_true_match = match_matrix.any(axis=0)

    pred_valid_r = pred_valid[pred_idx]

    true_is_eff = paired_match[true_valid]
    pred_is_pur = paired_match[pred_valid_r]
    pred_is_dup = (~paired_match & any_true_match)[pred_valid_r]
    pred_is_fak = (~any_true_match)[pred_valid_r]

    return true_is_eff, pred_is_pur, pred_is_dup, pred_is_fak


# ── Accumulator ───────────────────────────────────────────────────────────────


def make_bins():
    """Create empty bin accumulators for all working points, pred names, and quantities."""

    def zero_bins(qtys):
        return {p: {qty[0]: np.zeros(len(qty[3]) - 1) for qty in qtys} for p in PRED_NAMES}

    bins = {}
    for wp in WORKING_POINTS:
        bins[wp] = {
            "true_all": zero_bins(TRUE_QTYS),
            "true_eff": zero_bins(TRUE_QTYS),
            "pred_all": zero_bins(PRED_QTYS),
            "pred_pur": zero_bins(PRED_QTYS),
            "pred_dup": zero_bins(PRED_QTYS),
            "pred_fak": zero_bins(PRED_QTYS),
            "roi_true_all": zero_bins(ROI_QTYS),
            "roi_true_eff": zero_bins(ROI_QTYS),
            "roi_pred_all": zero_bins(ROI_QTYS),
            "roi_pred_pur": zero_bins(ROI_QTYS),
            "roi_pred_dup": zero_bins(ROI_QTYS),
            "roi_pred_fak": zero_bins(ROI_QTYS),
        }
    num_pix = {p: np.zeros((16, 16)) for p in PRED_NAMES}
    return bins, num_pix


def merge_results(results):
    """Sum bins and num_pix from all worker results."""
    all_bins, num_pix = make_bins()
    for w_bins, w_pix in results:
        for wp in WORKING_POINTS:
            for metric in all_bins[wp]:
                for pred_name in PRED_NAMES:
                    for qty_name in all_bins[wp][metric][pred_name]:
                        all_bins[wp][metric][pred_name][qty_name] += w_bins[wp][metric][pred_name][qty_name]
        for pred_name in PRED_NAMES:
            num_pix[pred_name] += w_pix[pred_name]
    return all_bins, num_pix


def fast_bin(qty, values, bin_edges):
    """Fast binned sum using searchsorted + bincount (replaces binned_statistic)."""
    n_bins = len(bin_edges) - 1
    idx = np.searchsorted(bin_edges, qty, side="right") - 1
    mask = (idx >= 0) & (idx < n_bins)
    idx_m = idx[mask]
    counts = np.bincount(idx_m, minlength=n_bins)[:n_bins].astype(float)
    sums = np.bincount(idx_m, weights=values[mask].astype(float), minlength=n_bins)[:n_bins]
    return counts, sums


def accumulate_true_bins(bins, pred_name, true_is_eff, targets, true_valid):
    """Add one ROI's true-particle quantities to the efficiency bins."""
    for qty_name, _, _, bin_edges in TRUE_QTYS:
        qty = targets[f"sudo_{qty_name}"][0][true_valid]
        n_all, n_eff = fast_bin(qty, true_is_eff, bin_edges)
        bins["true_all"][pred_name][qty_name] += n_all
        bins["true_eff"][pred_name][qty_name] += n_eff


def accumulate_pred_bins(bins, pred_name, pred_is_pur, pred_is_dup, pred_is_fak, targets):
    """Add one ROI's pred-track quantities to the purity/dup/fake bins."""
    for qty_name, _, _, bin_edges in PRED_QTYS:
        qty = np.full_like(pred_is_pur, targets[f"roi_{qty_name}"][0], dtype=np.float32)
        if qty_name == "energy":
            qty /= 1000.0
        if len(qty) == 0:
            continue
        n_all, n_pur = fast_bin(qty, pred_is_pur, bin_edges)
        _, n_dup = fast_bin(qty, pred_is_dup, bin_edges)
        _, n_fak = fast_bin(qty, pred_is_fak, bin_edges)
        bins["pred_all"][pred_name][qty_name] += n_all
        bins["pred_pur"][pred_name][qty_name] += n_pur
        bins["pred_dup"][pred_name][qty_name] += n_dup
        bins["pred_fak"][pred_name][qty_name] += n_fak


def accumulate_roi_bins(bins, pred_name, true_is_eff, pred_is_pur, pred_is_dup, pred_is_fak, roi_qty_values):
    """Add one ROI's ROI-level quantities to eff and pur bins."""
    for qty_name, _, _, bin_edges, _ in ROI_QTYS:
        val = roi_qty_values[qty_name]

        n_true = len(true_is_eff)
        if n_true > 0:
            qty_true = np.full(n_true, val)
            n_all, n_eff = fast_bin(qty_true, true_is_eff, bin_edges)
            bins["roi_true_all"][pred_name][qty_name] += n_all
            bins["roi_true_eff"][pred_name][qty_name] += n_eff

        n_pred = len(pred_is_pur)
        if n_pred > 0:
            qty_pred = np.full(n_pred, val)
            n_all, n_pur = fast_bin(qty_pred, pred_is_pur, bin_edges)
            _, n_dup = fast_bin(qty_pred, pred_is_dup, bin_edges)
            _, n_fak = fast_bin(qty_pred, pred_is_fak, bin_edges)
            bins["roi_pred_all"][pred_name][qty_name] += n_all
            bins["roi_pred_pur"][pred_name][qty_name] += n_pur
            bins["roi_pred_dup"][pred_name][qty_name] += n_dup
            bins["roi_pred_fak"][pred_name][qty_name] += n_fak


# ── Plotting ──────────────────────────────────────────────────────────────────


def auto_ylim(ax, margin=0.05):
    """Set y-axis limits based on plotted data, with some margin at bottom."""
    all_vals = []
    for line in ax.get_lines():
        ydata = line.get_ydata()
        finite = ydata[np.isfinite(ydata)]
        if len(finite):
            all_vals.append(finite)

    if not all_vals:
        return

    data_min = min(v.min() for v in all_vals)
    bottom = max(0.0, np.floor((data_min - margin) / 0.05) * 0.05)
    ax.set_ylim(bottom=bottom)


def make_sub_label(threshold):
    """Build the ATLAS-style sub-label text for a given working point threshold."""
    return (
        r"$\sqrt{s} = 13\,\mathrm{TeV},\; Z'\!\rightarrow q\bar{q}$"
        "\n"
        rf"$\mathrm{{WP}} \geq {threshold}$"
    )


def plot_efficiency(wp_bins, wp_label, threshold, plot_dir):
    """Plot particle (pixel) efficiency vs true-particle quantities."""
    hit_type = "Pixel " if PIX_ONLY else ""
    sub_label = make_sub_label(threshold)
    for qty_name, qty_label, scale, bin_edges in TRUE_QTYS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for pred_name in PRED_NAMES:
            k = wp_bins["true_eff"][pred_name][qty_name]
            n = wp_bins["true_all"][pred_name][qty_name]
            label = NAME_ALIASES[pred_name] if pred_name != "sudo" else None
            with np.errstate(invalid="ignore"):
                plot_hist_to_ax(
                    ax,
                    k / n,
                    bin_edges,
                    value_errors=bayesian_binomial_error(k, n),
                    label=label,
                    color=COLORS[pred_name],
                )
        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel(f"Particle {hit_type}Efficiency")
        ax.legend(fontsize=LEGEND_FONTSIZE)
        auto_ylim(ax)
        atlasify("Simulation Internal", sub_label, sub_font_size=SUB_FONTSIZE)
        fig.savefig(plot_dir / f"{qty_name}_eff.pdf")
        plt.close(fig)


def plot_pred_metric(wp_bins, key, ylabel, wp_label, threshold, plot_dir, filename_suffix):
    """Plot a pred-track metric (purity / fake rate / dup rate) vs pred quantities."""
    hit_type = "Pixel " if PIX_ONLY else ""
    sub_label = make_sub_label(threshold)
    for qty_name, qty_label, scale, bin_edges in PRED_QTYS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for pred_name in PRED_NAMES:
            k = wp_bins[key][pred_name][qty_name]
            n = wp_bins["pred_all"][pred_name][qty_name]
            label = NAME_ALIASES[pred_name] if pred_name != "sudo" else None
            with np.errstate(invalid="ignore"):
                plot_hist_to_ax(
                    ax,
                    k / n,
                    bin_edges,
                    value_errors=bayesian_binomial_error(k, n),
                    label=label,
                    color=COLORS[pred_name],
                )
        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel(f"Track {hit_type}{ylabel}")
        ax.legend(fontsize=LEGEND_FONTSIZE)
        auto_ylim(ax)
        atlasify("Simulation Internal", sub_label, sub_font_size=SUB_FONTSIZE)
        fig.savefig(plot_dir / f"{qty_name}_{filename_suffix}.pdf")
        plt.close(fig)


def plot_roi_efficiency(wp_bins, wp_label, threshold, plot_dir):
    """Plot particle efficiency vs ROI-level quantities."""
    hit_type = "Pixel " if PIX_ONLY else ""
    sub_label = make_sub_label(threshold)
    for qty_name, qty_label, scale, bin_edges, integer_ticks in ROI_QTYS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for pred_name in PRED_NAMES:
            k = wp_bins["roi_true_eff"][pred_name][qty_name]
            n = wp_bins["roi_true_all"][pred_name][qty_name]
            label = NAME_ALIASES[pred_name] if pred_name != "sudo" else None
            with np.errstate(invalid="ignore"):
                plot_hist_to_ax(
                    ax,
                    k / n,
                    bin_edges,
                    value_errors=bayesian_binomial_error(k, n),
                    label=label,
                    color=COLORS[pred_name],
                )
        ax.set_xscale(scale)
        if integer_ticks:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel(f"Particle {hit_type}Efficiency")
        ax.legend(fontsize=LEGEND_FONTSIZE)
        auto_ylim(ax)
        atlasify("Simulation Internal", sub_label, sub_font_size=SUB_FONTSIZE)
        fig.savefig(plot_dir / f"roi_{qty_name}_eff.pdf")
        plt.close(fig)


def plot_roi_pred_metric(wp_bins, key, ylabel, wp_label, threshold, plot_dir, filename_suffix):
    """Plot a pred-track metric vs ROI-level quantities."""
    hit_type = "Pixel " if PIX_ONLY else ""
    sub_label = make_sub_label(threshold)
    for qty_name, qty_label, scale, bin_edges, integer_ticks in ROI_QTYS:
        fig, ax = plt.subplots(figsize=(8, 6))
        for pred_name in PRED_NAMES:
            k = wp_bins[f"roi_{key}"][pred_name][qty_name]
            n = wp_bins["roi_pred_all"][pred_name][qty_name]
            label = NAME_ALIASES[pred_name] if pred_name != "sudo" else None
            with np.errstate(invalid="ignore"):
                plot_hist_to_ax(
                    ax,
                    k / n,
                    bin_edges,
                    value_errors=bayesian_binomial_error(k, n),
                    label=label,
                    color=COLORS[pred_name],
                )
        ax.set_xscale(scale)
        if integer_ticks:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel(f"Track {hit_type}{ylabel}")
        ax.legend(fontsize=LEGEND_FONTSIZE)
        auto_ylim(ax)
        atlasify("Simulation Internal", sub_label, sub_font_size=SUB_FONTSIZE)
        fig.savefig(plot_dir / f"roi_{qty_name}_{filename_suffix}.pdf")
        plt.close(fig)


def plot_pixel_sharing(num_pix, plot_dir):
    """Plot pixel hit sharing matrix for sisp, reco, and pred."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.2), layout="constrained")
    for i, pred_name in enumerate(["sisp", "reco", "pred"]):
        hist = num_pix[pred_name]
        denom = np.maximum(hist.sum(-1, keepdims=True), 1)
        frac = (hist / denom).T

        im = axes[i].imshow(frac, vmin=0, vmax=1)
        nrows, ncols = frac.shape
        for r in range(nrows):
            for c in range(ncols):
                val = float(frac[r, c])
                if val > 0.01:
                    axes[i].text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=4, color="black")

        axes[i].set_xlabel("Number of Particles\non Pixel Hit", fontsize=10)
        axes[i].set_xticks(np.arange(ncols))
        axes[i].set_yticks(np.arange(nrows))
        axes[i].set_xticklabels(range(ncols), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        axes[i].set_yticklabels(range(nrows), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        axes[i].text(
            0.01,
            0.01,
            NAME_ALIASES[pred_name],
            transform=axes[i].transAxes,
            color="white",
            ha="left",
            va="bottom",
            fontsize=10,
        )

    axes[0].set_ylabel("Number of Tracks\non Pixel Hit", fontsize=10)
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Fraction", fontsize=10)
    fig.savefig(plot_dir / "pixel_sharing.pdf")
    plt.close(fig)


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_track_from_data(data_file, sample_id, track_name, targets, num_pix_hits, num_sct_hits):
    """Load sisp/reco hit assignments from the original data h5 file."""
    t_valid = targets[f"{track_name}_valid"][0]
    n_tracks = t_valid.shape[0]

    pix_asgn = load_csr(data_file, sample_id, f"{track_name}_pix_valid")
    pix = pad_to_shape(pix_asgn, NUM_QUERIES, num_pix_hits)

    if PIX_ONLY:
        sct = np.zeros((NUM_QUERIES, num_sct_hits), dtype=bool)
    else:
        sct_asgn = load_csr(data_file, sample_id, f"{track_name}_sct_valid")
        sct = pad_to_shape(sct_asgn, NUM_QUERIES, num_sct_hits)

    valid = np.zeros(NUM_QUERIES, dtype=bool)
    n = min(n_tracks, NUM_QUERIES)
    valid[:n] = t_valid[:n]
    pix &= valid[:, None]
    sct &= valid[:, None]

    return valid, pix, sct


def load_pred_tracks(sample_id, targets, file_preds, data_file, num_pix_hits, num_sct_hits):
    """Load hit assignments for all pred names (sudo, sisp, reco, pred)."""
    true_valid = targets["sudo_valid"][0]
    true_pix = targets["sudo_pix_valid"][0] & true_valid[..., None]
    true_sct = targets["sudo_sct_valid"][0] & true_valid[..., None]

    preds = {}
    preds["sudo"] = (true_valid, true_pix, true_sct)

    for track_name in ["sisp", "reco"]:
        if data_file is not None:
            try:
                preds[track_name] = _load_track_from_data(
                    data_file,
                    sample_id,
                    track_name,
                    targets,
                    num_pix_hits,
                    num_sct_hits,
                )
                continue
            except Exception:
                pass
        preds[track_name] = (
            np.zeros(NUM_QUERIES, dtype=bool),
            np.zeros((NUM_QUERIES, num_pix_hits), dtype=bool),
            np.zeros((NUM_QUERIES, num_sct_hits), dtype=bool),
        )

    pred_valid = file_preds["pred_valid"]["pred_valid_prob"][0] >= 0.25
    pred_pix = file_preds["pred_pix_assignment"]["pred_pix_valid_prob"][0] >= 0.5
    pred_pix = pred_pix & pred_valid[..., None]
    pred_sct = np.zeros((NUM_QUERIES, num_sct_hits), dtype=bool)
    preds["pred"] = (pred_valid, pred_pix, pred_sct)

    return preds


# ── Worker ────────────────────────────────────────────────────────────────────


def process_chunk(sample_ids_chunk):
    """Process a chunk of ROIs. Each worker opens its own h5 file handles."""
    bins, num_pix = make_bins()
    roi_to_file = build_roi_index(DATA_DIR)
    open_data_files = {}

    def get_data_file(roi_id):
        fp = roi_to_file.get(roi_id)
        if fp is None:
            return None
        if fp not in open_data_files:
            open_data_files[fp] = h5py.File(fp, "r")
        return open_data_files[fp]

    with h5py.File(EVAL_PATH, "r") as file:
        for sample_id in sample_ids_chunk:
            targets = file[sample_id]["targets"]
            file_preds = file[sample_id]["preds"]["final"]

            true_valid = targets["sudo_valid"][0]
            true_pix = targets["sudo_pix_valid"][0] & true_valid[..., None]
            true_sct = targets["sudo_sct_valid"][0] & true_valid[..., None]
            pix_valid = targets["pix_valid"][0]

            num_pix_hits = pix_valid.shape[0]
            num_sct_hits = true_sct.shape[-1]

            true_sct_eval = np.zeros_like(true_sct) if PIX_ONLY else true_sct

            data_file = get_data_file(sample_id)
            preds = load_pred_tracks(
                sample_id,
                targets,
                file_preds,
                data_file,
                num_pix_hits,
                num_sct_hits,
            )

            roi_qty_values = {
                "n_pix_hits": float(pix_valid.sum()),
                "n_particles": float(true_valid.sum()),
            }

            for pred_name in PRED_NAMES:
                pred_valid, pred_pix, pred_sct = preds[pred_name]
                pred_sct_eval = np.zeros_like(pred_sct) if PIX_ONLY else pred_sct

                hit_eff, hit_pur, true_idx, pred_idx = compute_matching(
                    true_pix,
                    pred_pix,
                    true_sct_eval,
                    pred_sct_eval,
                    true_valid,
                    pred_valid,
                )

                pix_true_num = true_pix.sum(-2)[pix_valid]
                pix_pred_num = pred_pix.sum(-2)[pix_valid]
                num_pix[pred_name] += np.histogram2d(
                    pix_true_num,
                    pix_pred_num,
                    bins=np.arange(17) - 0.5,
                )[0]

                for wp_name, (threshold, _) in WORKING_POINTS.items():
                    true_is_eff, pred_is_pur, pred_is_dup, pred_is_fak = classify_matches(
                        hit_eff,
                        hit_pur,
                        true_valid,
                        pred_valid,
                        true_idx,
                        pred_idx,
                        threshold,
                    )
                    wp = bins[wp_name]
                    accumulate_true_bins(wp, pred_name, true_is_eff, targets, true_valid)
                    accumulate_pred_bins(wp, pred_name, pred_is_pur, pred_is_dup, pred_is_fak, targets)
                    accumulate_roi_bins(
                        wp,
                        pred_name,
                        true_is_eff,
                        pred_is_pur,
                        pred_is_dup,
                        pred_is_fak,
                        roi_qty_values,
                    )

    for f in open_data_files.values():
        f.close()

    return bins, num_pix


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    with h5py.File(EVAL_PATH, "r") as f:
        sample_ids = list(f.keys())
    if MAX_ROIS is not None:
        sample_ids = sample_ids[:MAX_ROIS]

    n_rois = len(sample_ids)
    n_workers = min(NUM_WORKERS, n_rois)

    chunks = [sample_ids[i::n_workers] for i in range(n_workers)]

    print(f"Processing {n_rois} ROIs with {n_workers} workers...")

    if n_workers <= 1:
        results = [process_chunk(sample_ids)]
    else:
        with Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_chunk, chunks),
                    total=n_workers,
                    desc="Workers",
                )
            )

    all_bins, num_pix = merge_results(results)

    for wp_name, (threshold, wp_label) in WORKING_POINTS.items():
        wp_dir = PLOT_DIR / wp_name
        wp_dir.mkdir(parents=True, exist_ok=True)

        wp = all_bins[wp_name]
        plot_efficiency(wp, wp_label, threshold, wp_dir)
        plot_pred_metric(wp, "pred_pur", "Purity", wp_label, threshold, wp_dir, "pur")
        plot_pred_metric(wp, "pred_fak", "Fake Rate", wp_label, threshold, wp_dir, "fak")
        plot_pred_metric(wp, "pred_dup", "Duplicate Rate", wp_label, threshold, wp_dir, "dup")
        plot_roi_efficiency(wp, wp_label, threshold, wp_dir)
        plot_roi_pred_metric(wp, "pred_pur", "Purity", wp_label, threshold, wp_dir, "pur")
        plot_roi_pred_metric(wp, "pred_fak", "Fake Rate", wp_label, threshold, wp_dir, "fak")
        plot_roi_pred_metric(wp, "pred_dup", "Duplicate Rate", wp_label, threshold, wp_dir, "dup")
        print(f"  [{wp_name}] plots saved to {wp_dir}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pixel_sharing(num_pix, PLOT_DIR)
    print(f"  Pixel sharing plot saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
