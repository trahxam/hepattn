# ruff: noqa: E402

import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from hepattn.experiments.pixel.utils.cluster_plot import load_data_config, load_first_batch, set_seed

plt.rcParams["figure.dpi"] = 300


def _append_missing(existing: list[str] | None, required: list[str]) -> list[str]:
    values = list(existing) if existing is not None else []
    for item in required:
        if item not in values:
            values.append(item)
    return values


def ensure_hist_fields(data_cfg: dict):
    inputs = data_cfg.setdefault("inputs", {})
    targets = data_cfg.setdefault("targets", {})

    inputs["cluster"] = _append_missing(
        inputs.get("cluster"),
        ["width_x", "width_y", "global_x", "global_y", "global_z", "global_r", "global_eta", "global_phi"],
    )
    inputs["pixel"] = _append_missing(inputs.get("pixel"), ["x", "y", "charge"])
    targets["particle"] = _append_missing(
        targets.get("particle"),
        ["x", "y", "theta", "phi", "p", "primary", "secondary", "notruth", "class_label"],
    )
    targets["cluster"] = _append_missing(targets.get("cluster"), ["multiplicity"])


def parse_args():
    parser = ArgumentParser(description="Plot pixel-cluster histograms.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "base.yaml",
        help="Path to the pixel config yaml (expects top-level `data` section).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which dataloader split to inspect.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override dataloader workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size to draw for plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plots",
        help="Directory where histogram figures will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic batch loading.",
    )
    return parser.parse_args()


def save_figure(fig, output_path: Path, show: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_histograms(inputs: dict, targets: dict, output_dir: Path, show: bool):
    particle_masks = {
        "All": targets["particle_valid"],
        "Primary": targets["particle_primary"].to(torch.bool) & targets["particle_valid"],
        "Secondary": targets["particle_secondary"].to(torch.bool) & targets["particle_valid"],
        "No Truth": targets["particle_notruth"].to(torch.bool) & targets["particle_valid"],
        "Primary or Secondary": (
            targets["particle_primary"].to(torch.bool) | targets["particle_secondary"].to(torch.bool)
        )
        & targets["particle_valid"],
    }

    print("Average query occupancy:", targets["particle_valid"].float().mean().item())

    cluster_multiplicities = targets["cluster_multiplicity"].cpu().numpy()
    multiplicities, multiplicity_counts = np.unique(cluster_multiplicities, return_counts=True)
    num_clusters = int(targets["cluster_valid"].sum().item())

    print(f"\nCluster multiplicities ({num_clusters} clusters)\n")
    single_cluster_count = multiplicity_counts[0]
    for multiplicity, multiplicity_count in zip(multiplicities, multiplicity_counts, strict=False):
        multiplicity_str = str(int(multiplicity)).ljust(4)
        count_str = str(int(multiplicity_count)).ljust(8)
        count_frac = multiplicity_count / len(cluster_multiplicities)
        pct_str = 100 * count_frac
        weight_str = single_cluster_count / multiplicity_count
        print(f"{multiplicity_str} | {count_str} | {pct_str:.2f}% | {weight_str:.2f}")

    particle_class_name_to_label = {
        "hadron": 1,
        "photon": 2,
        "electron": 3,
        "muon": 4,
        "tau": 5,
        "other": 6,
    }
    particle_class_label_to_name = {value: key for key, value in particle_class_name_to_label.items()}

    particle_class_labels = targets["particle_class_label"][targets["particle_valid"]].cpu().numpy()
    class_labels, class_counts = np.unique(particle_class_labels, return_counts=True)
    num_particles = float(targets["particle_valid"].sum().item())

    print(f"\nParticle classes ({int(num_particles)} particles)\n")
    for class_label, class_count in zip(class_labels, class_counts, strict=False):
        class_name = particle_class_label_to_name[int(class_label)].ljust(12)
        count_str = str(int(class_count)).ljust(8)
        pct_str = 100 * class_count / num_particles
        weight = num_particles / class_count
        print(f"{class_name} | {count_str} | {pct_str:.2f}% | {weight:.2f}")

    print(f"\nParticle truth types ({int(num_particles)} particles)\n")

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    for mask_name, particle_mask in particle_masks.items():
        pct = 100 * particle_mask.sum().item() / num_particles
        num_type = particle_mask.sum().item()
        weight = num_particles / num_type
        print(f"{mask_name.ljust(24)}| {str(num_type).ljust(6)} | {pct:.2f}% | {weight:.2f}")

        cluster_num_particles = particle_mask.sum(-1).cpu().numpy()
        ax.hist(
            cluster_num_particles,
            bins=np.arange(0, 16) - 0.5,
            label=mask_name,
            histtype="step",
            density=False,
        )

    ax.legend(fontsize=6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, 16))
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_ylabel("Density")
    ax.set_xlabel("Number of Particles of Given Origin on Cluster")
    save_figure(fig, output_dir / "cluster_num_particles_hist.png", show)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8, 2)

    axes[0].hist(inputs["cluster_width_x"].cpu().numpy(), bins=32, histtype="step", color="cornflowerblue")
    axes[1].hist(inputs["cluster_width_y"].cpu().numpy(), bins=32, histtype="step", color="cornflowerblue")

    axes[0].grid(alpha=0.25, linestyle="--")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[0].set_xlabel(r"Cluster Width $x$")
    axes[1].set_xlabel(r"Cluster Width $y$")
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    save_figure(fig, output_dir / "cluster_width.png", show)

    field_aliases = {
        "global_r": r"Cluster Global $r$",
        "global_eta": r"Cluster Global $\eta$",
        "global_phi": r"Cluster Global $\phi$",
        "global_x": r"Cluster Global $x$",
        "global_y": r"Cluster Global $y$",
        "global_z": r"Cluster Global $z$",
    }

    fields = ["global_r", "global_eta", "global_phi"]
    fig, axes = plt.subplots(nrows=1, ncols=len(fields))
    fig.set_size_inches(8, 2)

    for idx, field in enumerate(fields):
        axes[idx].hist(inputs[f"cluster_{field}"].cpu().numpy(), bins=32, histtype="step", color="cornflowerblue")
        axes[idx].set_xlabel(field_aliases[field])
        axes[idx].set_ylabel("Count")
        axes[idx].grid(alpha=0.25, linestyle="--")
    save_figure(fig, output_dir / "cluster_global_coords_angular.png", show)

    fields = ["global_x", "global_y", "global_z"]
    fig, axes = plt.subplots(nrows=1, ncols=len(fields))
    fig.set_size_inches(8, 2)

    for idx, field in enumerate(fields):
        axes[idx].hist(inputs[f"cluster_{field}"].cpu().numpy(), bins=32, histtype="step", color="cornflowerblue")
        axes[idx].set_xlabel(field_aliases[field])
        axes[idx].set_ylabel("Count")
        axes[idx].grid(alpha=0.25, linestyle="--")
    save_figure(fig, output_dir / "cluster_global_coords_cartesian.png", show)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8, 4)

    mask = torch.abs(inputs["cluster_global_eta"]) >= 2.5
    axes[0].scatter(
        inputs["cluster_global_x"][mask].cpu().numpy(),
        inputs["cluster_global_y"][mask].cpu().numpy(),
        alpha=0.5,
        s=1.0,
        color="cornflowerblue",
    )
    axes[1].scatter(
        inputs["cluster_global_z"][mask].cpu().numpy(),
        inputs["cluster_global_y"][mask].cpu().numpy(),
        alpha=0.5,
        s=1.0,
        color="cornflowerblue",
    )
    axes[0].set_xlabel(r"Cluster Global $x$")
    axes[0].set_ylabel(r"Cluster Global $y$")
    axes[1].set_xlabel(r"Cluster Global $z$")
    axes[1].set_ylabel(r"Cluster Global $y$")
    save_figure(fig, output_dir / "cluster_higheta.png", show)

    field_bins = {
        "x": np.linspace(-8, 8, 24),
        "y": np.linspace(-4, 4, 24),
        "theta": np.linspace(-np.pi, np.pi, 24),
        "phi": np.linspace(-np.pi / 4, np.pi / 4, 24),
        "p": np.logspace(-1, 4, 24),
    }
    field_symbols = {
        "x": r"$x$",
        "y": r"$y$",
        "theta": r"$\theta$",
        "phi": r"$\phi$",
        "p": r"$p$ [GeV]",
    }
    log_fields = {"p"}

    fields = ["x", "y"]
    fig, axes = plt.subplots(nrows=1, ncols=len(fields))
    fig.set_size_inches(8, 2)

    for idx, field in enumerate(fields):
        for mask_name, particle_mask in particle_masks.items():
            axes[idx].hist(
                targets[f"particle_{field}"][particle_mask.to(torch.bool)].cpu().numpy(),
                bins=field_bins[field],
                label=mask_name,
                histtype="step",
                density=True,
            )
            if field in log_fields:
                axes[idx].set_xscale("log")

        axes[idx].set_xlabel("Particle " + field_symbols[field])
        axes[idx].set_yscale("log")
        axes[idx].grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=6)
    save_figure(fig, output_dir / "cluster_particle_xy_hist.png", show)

    fields = ["theta", "phi", "p"]
    fig, axes = plt.subplots(nrows=1, ncols=len(fields))
    fig.set_size_inches(8, 2)

    for idx, field in enumerate(fields):
        for mask_name, particle_mask in particle_masks.items():
            axes[idx].hist(
                targets[f"particle_{field}"][particle_mask.to(torch.bool)].cpu().numpy(),
                bins=field_bins[field],
                label=mask_name,
                histtype="step",
                density=True,
            )
            if field in log_fields:
                axes[idx].set_xscale("log")

        axes[idx].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        axes[idx].tick_params(axis="y", which="both", bottom=False, top=False, labelbottom=False)
        axes[idx].set_xlabel("Particle " + field_symbols[field])
        axes[idx].set_yscale("log")
        axes[idx].grid(alpha=0.25, linestyle="--")

    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=6)
    save_figure(fig, output_dir / "cluster_particle_angles_hist.png", show)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)
    ax.hist(
        inputs["pixel_charge"][inputs["pixel_valid"]].cpu().numpy(),
        bins=np.logspace(-2, 0, 64),
        histtype="step",
        color="cornflowerblue",
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Density")
    ax.set_xlabel("Pixel Charge [ke / 100]")
    ax.grid(alpha=0.25, linestyle="--")
    save_figure(fig, output_dir / "pixel_charge.png", show)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8, 2)
    axes[0].hist(
        inputs["pixel_x"][inputs["pixel_valid"]].cpu().numpy(),
        bins=np.linspace(-5, 5, 11),
        histtype="step",
        color="cornflowerblue",
        density=True,
    )
    axes[1].hist(
        inputs["pixel_y"][inputs["pixel_valid"]].cpu().numpy(),
        bins=np.linspace(-5, 5, 11),
        histtype="step",
        color="cornflowerblue",
        density=True,
    )
    axes[0].set_xlabel(r"Pixel $x$ Index Position")
    axes[0].set_ylabel("Density")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[1].set_xlabel(r"Pixel $y$ Index Position")
    axes[1].set_ylabel("Density")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25, linestyle="--")
    save_figure(fig, output_dir / "pixel_local_xy.png", show)


def main():
    args = parse_args()
    set_seed(args.seed)
    data_cfg = load_data_config(args.config, batch_size=args.batch_size, num_workers=args.num_workers)
    data_cfg["seed"] = int(args.seed)
    ensure_hist_fields(data_cfg)
    inputs, targets = load_first_batch(data_cfg, split=args.split)
    plot_histograms(inputs, targets, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
