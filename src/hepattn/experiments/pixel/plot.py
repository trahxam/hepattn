from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import colors

from hepattn.experiments.pixel.data import PixelClusterDataModule

plt.rcParams["figure.dpi"] = 300

config_path = Path("src/hepattn/experiments/pixel/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 0
config["batch_size"] = 100

datamodule = PixelClusterDataModule(**config)
datamodule.setup(stage="test")


dataloader = datamodule.test_dataloader()

inputs, targets = next(iter(dataloader))

particle_masks = {
    "All": targets["particle_valid"],
    "Primary": targets["particle_primary"].to(torch.bool),
    "Secondary": targets["particle_secondary"].to(torch.bool),
    "No Truth": targets["particle_notruth"].to(torch.bool),
    "Primary or Secondary": targets["particle_primary"].to(torch.bool) | targets["particle_secondary"].to(torch.bool),
}

# Plot the particle counts
fig, ax = plt.subplots()
fig.set_size_inches(8, 2)

for mask_name, particle_mask in particle_masks.items():
    cluster_num_particles = particle_mask.sum(-1)
    ax.hist(
        cluster_num_particles,
        bins=np.arange(0, 16) - 0.5,
        label=mask_name,
        histtype="step",
    )

ax.legend(fontsize=10)
ax.set_yscale("log")
ax.set_xticks(np.arange(0, 16))
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.set_ylabel("Count")
ax.set_xlabel("Number of Pixels on Cluster")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_num_particles_hist.png")

# Plot the particle fields

fields = ["x", "y", "theta", "phi", "p"]

field_bins = {
    "x": np.linspace(-8, 8, 24),
    "y": np.linspace(-4, 4, 24),
    "theta": np.linspace(-np.pi, np.pi, 24),
    "phi": np.linspace(-np.pi, np.pi, 24),
    "p": np.logspace(-1, 3, 24),
}

field_symbols = {
    "x": r"$x$",
    "y": r"$y$",
    "theta": r"$\theta$",
    "phi": r"$\phi$",
    "p": r"$p$ [GeV]",
}

log_fields = ["p"]

fig, ax = plt.subplots(nrows=1, ncols=len(fields))
fig.set_size_inches(8, 2)

for i, field in enumerate(fields):
    for mask_name, particle_mask in particle_masks.items():
        ax[i].hist(
            targets[f"particle_{field}"][particle_mask.to(torch.bool)],
            bins=field_bins[field],
            label=mask_name,
            histtype="step",
        )

        if field in log_fields:
            ax[i].set_xscale("log")

    ax[i].set_xlabel("Particle " + field_symbols[field])
    ax[i].set_yscale("log")
    ax[i].grid(zorder=0, alpha=0.25, linestyle="--")

ax[0].set_ylabel("Count")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cluster_particle_fields_hist.png")

# Plot the cell charge

fig, ax = plt.subplots()
fig.set_size_inches(8, 2)

ax.hist(
    inputs["cell_charge"][inputs["cell_valid"]],
    bins=np.logspace(-2, 0, 64),
    histtype="step",
    color="cornflowerblue",
)

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_ylabel("Count")
ax.set_xlabel("Pixel Charge [ke / 100]")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/charge.png")

# Plot the cell coordinates

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(8, 2)

ax[0].hist(
    inputs["cell_x"][inputs["cell_valid"]],
    bins=np.linspace(-5, 5, 11),
    histtype="step",
    color="cornflowerblue",
)

ax[1].hist(
    inputs["cell_y"][inputs["cell_valid"]],
    bins=np.linspace(-5, 5, 11),
    histtype="step",
    color="cornflowerblue",
)

ax[0].set_xlabel(r"Pixel $x$ Index Position")
ax[0].set_ylabel("Count")
ax[0].set_yscale("log")

ax[1].set_xlabel(r"Pixel $y$ Index Position")
ax[1].set_ylabel("Count")
ax[1].set_yscale("log")

fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/cell_local_xy.png")

# Plot some examples

fig, ax = plt.subplots(6, 6)
fig.set_size_inches(8, 8)

ax = ax.flatten()

idx = torch.argsort(targets["particle_valid"].sum(-1), descending=True)

cell_charge = inputs["cell_charge"][inputs["cell_valid"]]
norm = colors.Normalize(vmin=torch.min(cell_charge), vmax=torch.min(cell_charge))

for ax_idx in range(len(ax)):
    i = idx[ax_idx]

    extent = 4.5

    ax[ax_idx].set_xticks(np.linspace(-extent, extent, int(2 * extent + 1)))
    ax[ax_idx].set_yticks(np.linspace(-extent, extent, int(2 * extent + 1)))

    ax[ax_idx].grid(alpha=0.5, zorder=-100)
    ax[ax_idx].grid(alpha=0.5, zorder=-100)

    # ax[ax_idx].spines['left'].set_visible(False)
    # ax[ax_idx].spines['right'].set_visible(False)
    # ax[ax_idx].spines['top'].set_visible(False)
    # ax[ax_idx].spines['bottom'].set_visible(False)

    ax[ax_idx].set_xticklabels([])
    ax[ax_idx].set_yticklabels([])

    ax[ax_idx].imshow(
        np.zeros(shape=(10, 10)) * np.nan,
        extent=(-extent, extent, -extent, extent),
    )

    ax[ax_idx].scatter(
        inputs["cell_x"][i][inputs["cell_valid"][i]],
        inputs["cell_y"][i][inputs["cell_valid"][i]],
        c=inputs["cell_charge"][i][inputs["cell_valid"][i]],
        marker="s",
        ec="black",
        s=64,
    )

    particles_x = targets["particle_x"][i][targets["particle_valid"][i]]
    particles_y = targets["particle_y"][i][targets["particle_valid"][i]]

    particles_phi = targets["particle_phi"][i][targets["particle_valid"][i]]
    particles_theta = targets["particle_theta"][i][targets["particle_valid"][i]]

    particles_dx = particles_phi / torch.sqrt(particles_phi**2 + particles_theta**2)
    particles_dy = particles_theta / torch.sqrt(particles_phi**2 + particles_theta**2)

    ax[ax_idx].scatter(particles_x, particles_y, c="cornflowerblue", ec="black", s=32, linewidths=1)

    num_particles = targets["particle_valid"][i].sum(-1)

    for particle_idx in range(num_particles):
        ax[ax_idx].arrow(
            particles_x[particle_idx],
            particles_y[particle_idx],
            particles_dx[particle_idx],
            particles_dy[particle_idx],
            width=0.01,
        )


fig.tight_layout()
fig.savefig("src/hepattn/experiments/pixel/plots/examples.png")
