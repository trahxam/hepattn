from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plotting import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300

BINS = {
    "pos.x": np.linspace(-0.015, 0.015, 64),
    "pos.y": np.linspace(-0.015, 0.015, 64),
    "pos.z": np.linspace(-0.25, 0.25, 64),
    "pos.r": np.linspace(0.0125, 0.015, 64),
    "pos.eta": np.linspace(-4, 4, 64),
    "pos.phi": np.linspace(-np.pi, np.pi, 64),
    "time": np.linspace(0, 2.5, 64),
}

LABELS = {
    "pos.x": "VTXD Hit $x$ [m]",
    "pos.y": "VTXD Hit $y$ [m]",
    "pos.z": "VTXD Hit $z$ [m]",
    "pos.r": "VTXD Hit $r$ [m]",
    "pos.eta": r"VTXD Hit $\eta$",
    "pos.phi": r"VTXD Hit $\phi$",
    "time": "VTXD Hit time [ns]",
}

PLOT_GROUPS = {
    "vtxd_pandora_hits_xyz": ["pos.x", "pos.y", "pos.z"],
    "vtxd_pandora_hits_retaphi": ["pos.r", "pos.eta", "pos.phi"],
    "vtxd_pandora_hits_time": ["time"],
}

CATEGORIES = {
    "on_pandora": "On truth particle + Pandora track",
    "off_pandora": "On truth particle, not on Pandora track",
}

COLOURS = {
    "on_pandora": "tab:blue",
    "off_pandora": "tab:orange",
}

hists = {field: {cat: CountingHistogram(bins) for cat in CATEGORIES} for field, bins in BINS.items()}

config_path = Path("src/hepattn/experiments/cld/configs/trkfit.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 10
config["batch_size"] = 10
config["num_test"] = 1000

datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()
data_iterator = iter(dataloader)

for _ in tqdm(range(50)):
    try:
        inputs, targets = next(data_iterator)
    except StopIteration:
        break

    # [batch, n_particle, n_vtxd] -> [batch, n_vtxd]: True if any truth particle has this hit
    on_particle = targets["particle_vtxd_valid"].any(dim=-2)

    # [batch, n_pandora, n_vtxd] -> [batch, n_vtxd]: True if any pandora track contains this hit
    on_pandora = targets["pandora_vtxd_valid"].any(dim=-2)

    innermost = inputs["vtxd_pos.r"] < 0.015

    on_mask = (innermost & on_particle & on_pandora).numpy()
    off_mask = (innermost & on_particle & ~on_pandora).numpy()

    for field in BINS:
        values = inputs[f"vtxd_{field}"].numpy()
        hists[field]["on_pandora"].fill(values[on_mask])
        hists[field]["off_pandora"].fill(values[off_mask])

out_dir = Path("src/hepattn/experiments/cld/plots/data")
out_dir.mkdir(parents=True, exist_ok=True)

for plot_name, fields in PLOT_GROUPS.items():
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 3))
    if len(fields) == 1:
        axes = [axes]

    for ax_idx, field in enumerate(fields):
        ax = axes[ax_idx]
        for cat_name, cat_label in CATEGORIES.items():
            hist = hists[field][cat_name]
            plot_hist_to_ax(
                ax,
                hist.counts,
                hist.bins,
                label=cat_label,
                color=COLOURS[cat_name],
                vertical_lines=True,
            )
        ax.set_yscale("log")
        ax.set_xlabel(LABELS[field])
        ax.set_ylabel("Count")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{plot_name}.png")
    plt.close(fig)
