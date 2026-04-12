# ruff: noqa
"""Plot TIDE ROI event displays with ATLAS-style formatting.

Usage:
    pixi run python src/reconstruct_anything/experiments/tide/scripts/plot_roi.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from atlasify import atlasify
from reconstruct_anything.experiments.tide.data.data import ROIDataModule

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
    "legend.fontsize": 14,
})

SAVE_FORMAT = "png"
SUB_FONTSIZE = 16
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"


def plot_roi_display(inputs, targets, batch_idx, plot_dir):
    """Plot an ROI event display showing pixel hits coloured by track assignment."""
    track = "sudo"

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    colormap = plt.cm.tab10
    cycler = [colormap(i) for i in range(colormap.N)]

    hit_x = inputs["pix_r"][batch_idx]
    hit_y = inputs["pix_dphi"][batch_idx]
    hit_valid = inputs["pix_valid"][batch_idx]
    mask = targets[f"{track}_pix_valid"][batch_idx]
    track_valid = targets[f"{track}_valid"][batch_idx]

    # Left: r vs dphi
    ax[0].scatter(hit_x[hit_valid], hit_y[hit_valid], s=16.0, marker="s", fc="none", ec="black", linewidths=0.5)
    for track_idx in range(track_valid.shape[-1]):
        if not track_valid[track_idx]:
            continue
        color = cycler[track_idx % len(cycler)]
        tx = hit_x[mask[track_idx]]
        ty = hit_y[mask[track_idx]]
        sort_idx = torch.argsort(tx)
        ax[0].plot(tx[sort_idx], ty[sort_idx], color=color, linewidth=1.5)

    ax[0].set_xlabel(r"Pixel $r$ [mm]")
    ax[0].set_ylabel(r"Pixel $\Delta\phi$")
    ax[0].grid(zorder=0, alpha=0.25, linestyle="--")

    # Right: eta vs phi
    hit_eta = inputs["pix_eta"][batch_idx]
    hit_phi = inputs["pix_phi"][batch_idx]

    ax[1].scatter(hit_eta[hit_valid], hit_phi[hit_valid], s=16.0, marker="s", fc="none", ec="black", linewidths=0.5)
    for track_idx in range(track_valid.shape[-1]):
        if not track_valid[track_idx]:
            continue
        color = cycler[track_idx % len(cycler)]
        tx = hit_eta[mask[track_idx]]
        ty = hit_phi[mask[track_idx]]
        sort_idx = torch.argsort(tx)
        ax[1].plot(tx[sort_idx], ty[sort_idx], color=color, linewidth=1.5)

    ax[1].set_xlabel(r"Pixel $\eta$")
    ax[1].set_ylabel(r"Pixel $\phi$")
    ax[1].grid(zorder=0, alpha=0.25, linestyle="--")

    num_pix = hit_valid.sum().item()
    num_tracks = track_valid.sum().item()
    roi_id = targets["sample_id"][batch_idx].item()

    atlasify(
        "Simulation Internal",
        rf"$\sqrt{{s}} = 13\,\mathrm{{TeV}},\; Z'\!\rightarrow q\bar{{q}}$"
        + f"\nROI {roi_id}: {num_pix} pixel hits, {num_tracks} tracks",
        sub_font_size=SUB_FONTSIZE,
    )

    fig.savefig(plot_dir / f"roi_display.{SAVE_FORMAT}")
    plt.close(fig)


def main():
    torch.manual_seed(42)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0
    config["batch_size"] = 100
    config["num_test"] = 1000
    config["test_dir"] = "/share/rcifdata/maxhart/data/ambi/train/"

    datamodule = ROIDataModule(**config)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    inputs, targets = next(iter(dataloader))

    # Pick the ROI with the most tracks
    batch_idx = torch.argmax(targets["sudo_valid"].sum(-1))
    print(f"Plotting ROI {targets['sample_id'][batch_idx].item()} "
          f"({targets['sudo_valid'][batch_idx].sum().item()} tracks, "
          f"{inputs['pix_valid'][batch_idx].sum().item()} pixel hits)")

    plot_roi_display(inputs, targets, batch_idx, PLOT_DIR)
    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
