from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from atlasify import atlasify
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Matplotlib defaults
# -----------------------------------------------------------------------------
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 16,
})

# -----------------------------------------------------------------------------
# Labels / config
# -----------------------------------------------------------------------------
SUBHEADING_BASE = r"$\sqrt{s} = 13\,\mathrm{TeV},\; Z'\!\rightarrow q\bar{q}$"


@dataclass(frozen=True)
class Field:
    name: str  # suffix under targets['sudo_{name}'] or special
    label: str  # axis label
    scale: str  # 'linear' or 'log' (x-axis)
    bins: np.ndarray  # bin edges


# Per-pseudotrack (truth 'sudo_*' arrays, masked by sudo_valid)
TRACK_FIELDS: tuple[Field, ...] = (
    Field("pt", r"Pseudotrack $p_\mathrm{T}$ [GeV]", "log", np.geomspace(2, 3.5e3, 64)),
    Field("bhad_pt", r"b-hadron $p_\mathrm{T}$ [GeV]", "log", np.geomspace(150, 5e3, 64)),
    Field("eta", r"Pseudotrack $\eta$", "linear", np.linspace(-2.5, 2.5, 64)),
    Field("phi", r"Pseudotrack $\phi$", "linear", np.linspace(-np.pi, np.pi, 64)),
    Field("deta", r"Pseudotrack $-\,$RoI axis $\Delta \eta$", "linear", np.linspace(-0.05, 0.05, 64)),
    Field("dphi", r"Pseudotrack $-\,$RoI axis $\Delta \phi$", "linear", np.linspace(-0.05, 0.05, 64)),
)

# Per-RoI (one value per sample)
ROI_FIELDS: tuple[Field, ...] = (
    Field("energy", r"RoI Energy [GeV]", "log", np.geomspace(150, 5e3, 64)),
    Field("n_sudo", r"Number of Particles in RoI", "linear", np.arange(0, 50, 1) - 0.5),  # integer bins 0..101
    Field("n_pix", r"Number of Pixel hits per RoI", "linear", np.arange(0, 101, 5) - 0.5),  # 0..250
    Field("n_sct", r"Number of SCT hits per RoI", "linear", np.arange(0, 201, 5) - 0.5),  # 0..500
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _init_histograms(fields: Iterable[Field]) -> dict[str, np.ndarray]:
    return {f.name: np.zeros(len(f.bins) - 1, dtype=float) for f in fields}


def _accumulate_hist(counts: np.ndarray, values: np.ndarray, bins: np.ndarray) -> None:
    if values.size == 0:
        return
    h, _ = np.histogram(values, bins=bins)
    counts += h


def _stairs(ax, counts: np.ndarray, bins: np.ndarray, *, label: str | None = None):
    ax.stairs(counts, bins, label=label, color="black")


def _finalize(ax, xscale: str, xlabel: str, ylabel: str, subheading: str):
    ax.set_xscale(xscale)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    atlasify("Simulation Internal", subheading, sub_font_size=16)
    ax.legend(loc="best")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    eval_path: Path = Path(
        "/share/rcifdata/maxhart/hepattn/logs/TIDE_32trk_F32_fixed_nodrop_20250926-T150450/ckpts/epoch=003-train_loss=0.54949_prepped_eval.h5"
    ),
    outdir: Path = Path("./plots_quick_hists"),
    max_samples: int = 10_000,
    title_suffix: str | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Accumulators
    track_hists = _init_histograms(TRACK_FIELDS)
    roi_hists = _init_histograms(ROI_FIELDS)
    roi_bins = {f.name: f.bins for f in ROI_FIELDS}

    n_events = 0

    with h5py.File(eval_path, "r") as f:
        keys = list(f.keys())
        for i, sample_id in tqdm(enumerate(keys), total=min(len(keys), max_samples)):
            if i >= max_samples:
                break

            g = f[sample_id]
            targets = g["targets"]

            # Truth pseudotrack validity mask
            sudo_valid = targets["sudo_valid"][0].astype(bool, copy=False)

            # Per-track quantities (masked by validity)
            for fld in TRACK_FIELDS:
                arr = targets[f"sudo_{fld.name}"][0]
                vals = arr[sudo_valid]
                _accumulate_hist(track_hists[fld.name], vals, fld.bins)

            # Per-RoI values (one per sample)
            roi_energy_gev = float(targets["roi_energy"][0]) / 1000.0  # to GeV
            _accumulate_hist(roi_hists["energy"], np.array([roi_energy_gev]), roi_bins["energy"])

            n_sudo = int(np.count_nonzero(sudo_valid))
            _accumulate_hist(roi_hists["n_sudo"], np.array([n_sudo]), roi_bins["n_sudo"])

            # Pixel/SCT hits per RoI
            n_pix = int(np.count_nonzero(targets["pix_valid"][0].astype(bool, copy=False)))
            _accumulate_hist(roi_hists["n_pix"], np.array([n_pix]), roi_bins["n_pix"])

            # SCT validity key might be 'sct_valid' (expected); fallback to 'strip_valid' if present
            if "sct_valid" in targets:
                sct_valid_arr = targets["sct_valid"][0]
            elif "strip_valid" in targets:
                sct_valid_arr = targets["strip_valid"][0]
            else:
                sct_valid_arr = None

            if sct_valid_arr is not None:
                n_sct = int(np.count_nonzero(sct_valid_arr.astype(bool, copy=False)))
                _accumulate_hist(roi_hists["n_sct"], np.array([n_sct]), roi_bins["n_sct"])

            n_events += 1

    # Compose subheading
    extra = f"\nSample of {n_events} RoIs"
    if title_suffix:
        extra += f"\n{title_suffix}"
    subheading = SUBHEADING_BASE + extra

    # --------
    # Plots
    # --------
    # Track-level
    for fld in TRACK_FIELDS:
        counts = track_hists[fld.name]
        fig, ax = plt.subplots(figsize=(8, 6))
        _stairs(ax, counts, fld.bins, label="Truth Particles")
        _finalize(ax, fld.scale, fld.label, "Count", f"{fld.label} Distribution", subheading)
        fig.tight_layout()
        fig.savefig(outdir / f"sudo_{fld.name}.png")
        plt.close(fig)

    # RoI-level
    for fld in ROI_FIELDS:
        counts = roi_hists[fld.name]
        fig, ax = plt.subplots(figsize=(8, 6))
        _stairs(ax, counts, fld.bins, label="RoI")
        _finalize(ax, fld.scale, fld.label, "Count", f"RoI {fld.label}", subheading)

        # Special ticks/limits for integer-count histograms
        if fld.name == "n_sudo":
            ax.set_xticks(np.arange(0, 50, 5))
            ax.set_xlim(-0.5, 50.5)
        elif fld.name == "n_pix":
            ax.set_xlim(-0.5, 100.5)
        elif fld.name == "n_sct":
            ax.set_xlim(-0.5, 200.5)

        fig.tight_layout()
        fig.savefig(outdir / f"roi_{fld.name}.png")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick histograms for truth pseudotracks and RoI-level variables")
    parser.add_argument("--eval-path", type=Path, help="Path to prepared evaluation HDF5")
    parser.add_argument("--outdir", type=Path, default=Path("./plots_quick_hists"), help="Output directory for plots")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples to process")
    parser.add_argument("--title-suffix", type=str, default=None, help="Optional extra line in the plot subheading")

    args = parser.parse_args()

    main(
        eval_path=args.eval_path,
        outdir=args.outdir,
        max_samples=args.max_samples,
        title_suffix=args.title_suffix,
    )
