import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from hepattn.experiments.pixel.data import PixelClusterDataModule, PixelClusterDataset

OUTPUT_DPI = 320
GRID_BUFFER_CELLS = 1.0
PLOT_MAX_MULTIPLICITY = 8


def _append_missing(existing: list[str] | None, required: list[str]) -> list[str]:
    values = list(existing) if existing is not None else []
    for item in required:
        if item not in values:
            values.append(item)
    return values


def ensure_plot_fields(data_cfg: dict):
    inputs = data_cfg.setdefault("inputs", {})
    targets = data_cfg.setdefault("targets", {})

    inputs["cluster"] = _append_missing(inputs.get("cluster"), ["pitch_vector"])
    inputs["pixel"] = _append_missing(inputs.get("pixel"), ["x", "y", "charge", "pitch_y"])
    targets["particle"] = _append_missing(
        targets.get("particle"),
        ["x", "y", "phi", "theta", "p", "primary", "secondary", "notruth", "class_label"],
    )


def load_data_config(config_path: Path, batch_size: int, num_workers: int) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if "data" not in config:
        raise KeyError(f"Config {config_path} does not contain a top-level `data` section.")

    data_cfg = dict(config["data"])
    data_cfg["batch_size"] = batch_size
    data_cfg["num_workers"] = num_workers
    data_cfg["cluster_max_multiplicity"] = min(int(data_cfg.get("cluster_max_multiplicity", PLOT_MAX_MULTIPLICITY)), PLOT_MAX_MULTIPLICITY)
    ensure_plot_fields(data_cfg)
    return data_cfg


def _point_range(tensors: list[torch.Tensor]) -> tuple[float, float]:
    valid_tensors = [tensor for tensor in tensors if tensor.numel() > 0]
    if not valid_tensors:
        return 0.0, 0.0

    min_value = min(float(tensor.min().item()) for tensor in valid_tensors)
    max_value = max(float(tensor.max().item()) for tensor in valid_tensors)
    return min_value, max_value


def _grid_bounds(
    x_tensors: list[torch.Tensor],
    y_tensors: list[torch.Tensor],
    buffer_cells: float = GRID_BUFFER_CELLS,
) -> tuple[float, float, float, float, float, float, float, float]:
    x_min, x_max = _point_range(x_tensors)
    y_min, y_max = _point_range(y_tensors)

    grid_x_min = math.floor(x_min - buffer_cells)
    grid_x_max = math.ceil(x_max + buffer_cells)
    grid_y_min = math.floor(y_min - buffer_cells)
    grid_y_max = math.ceil(y_max + buffer_cells)

    return (
        grid_x_min,
        grid_x_max,
        grid_y_min,
        grid_y_max,
        grid_x_min - 0.5,
        grid_x_max + 0.5,
        grid_y_min - 0.5,
        grid_y_max + 0.5,
    )


def _format_pitch_label(pitch: float) -> str:
    if np.isclose(pitch, round(pitch)):
        return str(round(pitch))
    return f"{pitch:.2f}".rstrip("0").rstrip(".")


def _cluster_pitch_vector_title(cluster_pitch_vector: torch.Tensor) -> str:
    values = cluster_pitch_vector.detach().cpu().reshape(-1).tolist()
    formatted_values = ", ".join(_format_pitch_label(float(value)) for value in values)
    return f"cluster_pitch_vector\n[{formatted_values}]"


def _configure_example_axis(
    axis,
    grid_x_min: float,
    grid_x_max: float,
    grid_y_min: float,
    grid_y_max: float,
    y_tick_labels: list[str] | None = None,
):
    x_centers = np.arange(grid_x_min, grid_x_max + 1e-9, 1.0)
    y_centers = np.arange(grid_y_min, grid_y_max + 1e-9, 1.0)
    x_edges = np.arange(grid_x_min - 0.5, grid_x_max + 0.5 + 1e-9, 1.0)
    y_edges = np.arange(grid_y_min - 0.5, grid_y_max + 0.5 + 1e-9, 1.0)

    axis.set_xticks(x_centers)
    axis.set_yticks(y_centers)
    axis.set_xticks(x_edges, minor=True)
    axis.set_yticks(y_edges, minor=True)
    axis.grid(which="minor", alpha=0.5)
    axis.tick_params(which="minor", length=0)
    axis.tick_params(which="major", length=0, pad=1.5)
    axis.set_xticklabels([])
    axis.set_yticklabels([] if y_tick_labels is None else y_tick_labels, fontsize=6)
    axis.set_xlim(grid_x_min - 0.5, grid_x_max + 0.5)
    axis.set_ylim(grid_y_min - 0.5, grid_y_max + 0.5)
    axis.set_aspect("equal")


def _pitch_y_tick_labels(grid_y_min: float, grid_y_max: float, cluster_pitch_vector: torch.Tensor) -> list[str]:
    axis_rows = np.arange(grid_y_min, grid_y_max + 1e-9, 1.0)
    pitch_values = PixelClusterDataset.pixel_y_to_pitch_y(axis_rows, cluster_pitch_vector.detach().cpu().numpy())
    return [_format_pitch_label(float(pitch)) for pitch in pitch_values]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_example_indices(inputs: dict, targets: dict, num_examples: int):
    total = int(inputs["pixel_valid"].shape[0])
    requested = min(int(num_examples), total)
    if requested <= 0:
        return torch.empty(0, dtype=torch.long)

    occupancy = inputs["pixel_valid"].sum(-1)
    ranked = torch.argsort(occupancy, descending=True)

    if "particle_valid" not in targets:
        return ranked[:requested]

    particle_valid = targets["particle_valid"].to(torch.bool)
    multiplicity = particle_valid.sum(-1).to(torch.long)

    selected = []
    used = set()

    # Prefer one example from each multiplicity up to the plotting cap.
    preferred = list(range(1, min(PLOT_MAX_MULTIPLICITY, requested) + 1))
    for multiplicity_value in preferred:
        ranked_with_multiplicity = ranked[multiplicity[ranked] == multiplicity_value]
        if ranked_with_multiplicity.numel() == 0:
            continue

        idx = int(ranked_with_multiplicity[0].item())
        if idx not in used:
            selected.append(idx)
            used.add(idx)

        if len(selected) >= requested:
            break

    if len(selected) < requested:
        for idx in ranked.tolist():
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
            if len(selected) >= requested:
                break

    return torch.tensor(selected, dtype=torch.long)


def build_dataloader(data_cfg: dict, split: str):
    datamodule = PixelClusterDataModule(**data_cfg)
    if split in {"train", "val"}:
        datamodule.setup(stage="fit")
        return datamodule.train_dataloader() if split == "train" else datamodule.val_dataloader()

    datamodule.setup(stage="test")
    return datamodule.test_dataloader()


def load_first_batch(data_cfg: dict, split: str):
    dataloader = build_dataloader(data_cfg, split)
    try:
        return next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("No batches were loaded from the dataloader.") from exc


def _finalize(fig, output_path: Path, show: bool, plt_mod, tight: bool = True):
    pdf_path = output_path.with_suffix(".pdf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    for axis in fig.get_axes():
        axis.set_facecolor("white")
    if tight:
        fig.tight_layout()
    fig.savefig(pdf_path, dpi=OUTPUT_DPI, facecolor="white", edgecolor="white")
    print(f"Saved: {pdf_path}")
    if show:
        plt_mod.show()
    else:
        plt_mod.close(fig)


def plot_charge_examples(
    inputs: dict,
    targets: dict,
    num_examples: int,
    output_dir: Path,
    show: bool,
    nrows: int = 2,
    ncols: int = 4,
    show_pitch_y_labels: bool = True,
    show_cluster_pitch_vector_title: bool = True,
):
    required_fields = ["pixel_x", "pixel_y", "pixel_charge", "pixel_valid"]
    if any(field not in inputs for field in required_fields):
        print("Skipping examples plot: missing pixel fields in dataloader inputs.")
        return

    idx = select_example_indices(inputs, targets, num_examples)
    max_examples = int(idx.numel())
    if max_examples <= 0:
        return

    nrows = max(1, int(nrows))
    ncols = max(2, int(ncols))
    total_slots = nrows * ncols
    example_slots = max(1, total_slots - 1)
    if max_examples > example_slots:
        nrows = math.ceil((max_examples + 1) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(-1)
    example_axes = axes[:-1]
    key_ax = axes[-1]
    max_examples = min(max_examples, len(example_axes))

    norm = colors.LogNorm(vmin=0.01, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    for ax_idx in range(max_examples):
        axis = example_axes[ax_idx]
        sample_idx = int(idx[ax_idx])
        pix_valid = inputs["pixel_valid"][sample_idx].to(torch.bool)
        pix_x = inputs["pixel_x"][sample_idx][pix_valid]
        pix_y = inputs["pixel_y"][sample_idx][pix_valid]
        pix_charge = inputs["pixel_charge"][sample_idx][pix_valid]

        if len(pix_x) == 0:
            axis.axis("off")
            continue

        part_valid = targets["particle_valid"][sample_idx].to(torch.bool)
        particles_x = targets["particle_x"][sample_idx][part_valid]
        particles_y = targets["particle_y"][sample_idx][part_valid]
        particles_phi = targets["particle_phi"][sample_idx][part_valid]
        particles_theta = targets["particle_theta"][sample_idx][part_valid]
        particles_primary = targets["particle_primary"][sample_idx][part_valid].to(torch.bool)
        particles_secondary = targets["particle_secondary"][sample_idx][part_valid].to(torch.bool)
        particles_notruth = targets["particle_notruth"][sample_idx][part_valid].to(torch.bool)

        denom = torch.sqrt(particles_phi**2 + particles_theta**2).clamp_min(1e-6)
        particles_dx = particles_phi / denom
        particles_dy = particles_theta / denom
        particles_end_x = particles_x + particles_dx
        particles_end_y = particles_y + particles_dy

        _, _, _, _, x_min, x_max, y_min, y_max = _grid_bounds(
            [pix_x, particles_x, particles_end_x],
            [pix_y, particles_y, particles_end_y],
        )

        grid_x_min = x_min + 0.5
        grid_x_max = x_max - 0.5
        grid_y_min = y_min + 0.5
        grid_y_max = y_max - 0.5

        y_tick_labels = None
        if show_pitch_y_labels and "cluster_pitch_vector" in inputs:
            y_tick_labels = _pitch_y_tick_labels(grid_y_min, grid_y_max, inputs["cluster_pitch_vector"][sample_idx])

        _configure_example_axis(
            axis,
            grid_x_min,
            grid_x_max,
            grid_y_min,
            grid_y_max,
            y_tick_labels=y_tick_labels,
        )
        if show_cluster_pitch_vector_title and "cluster_pitch_vector" in inputs:
            axis.set_title(_cluster_pitch_vector_title(inputs["cluster_pitch_vector"][sample_idx]), fontsize=5.5, pad=2.0)

        for pixel_idx in range(len(pix_x)):
            x = pix_x[pixel_idx].item()
            y = pix_y[pixel_idx].item()
            charge = max(pix_charge[pixel_idx].item(), 1e-6)
            pixel_patch = Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, color=cmap(norm(charge)), zorder=1)
            axis.add_patch(pixel_patch)

        for particle_idx in range(len(particles_x)):
            if particles_primary[particle_idx]:
                color = "crimson"
            elif particles_secondary[particle_idx]:
                color = "darkorange"
            elif particles_notruth[particle_idx]:
                color = "darkgray"
            else:
                color = "white"

            axis.arrow(
                particles_x[particle_idx].item(),
                particles_y[particle_idx].item(),
                particles_dx[particle_idx].item(),
                particles_dy[particle_idx].item(),
                width=0.03,
                head_width=0.35,
                head_length=0.35,
                length_includes_head=True,
                fc=color,
                ec="black",
                linewidth=0.6,
                zorder=4,
            )
            axis.scatter(
                particles_x[particle_idx].item(),
                particles_y[particle_idx].item(),
                c=color,
                ec="black",
                s=64,
                linewidths=1,
                zorder=6,
            )

    for axis in example_axes[max_examples:]:
        axis.axis("off")

    key_ax.axis("off")
    custom_markers = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markeredgecolor="black", markersize=10)
        for label, color in [("Primary", "crimson"), ("Secondary", "darkorange"), ("No Truth", "darkgray")]
    ]
    key_ax.legend(handles=custom_markers, loc="upper center", fontsize=8, frameon=False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = key_ax.inset_axes([0.08, 0.10, 0.84, 0.15])
    colorbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    colorbar.set_label("Pixel Charge [ke / 100]", fontsize=8)

    _finalize(fig, output_dir / "examples.png", show, plt, tight=True)


def plot_charge_examples_3d(
    inputs: dict,
    targets: dict,
    num_examples: int,
    output_dir: Path,
    show: bool,
    nrows: int = 2,
    ncols: int = 4,
    show_cluster_pitch_vector_title: bool = True,
):
    required_fields = ["pixel_x", "pixel_y", "pixel_charge", "pixel_valid"]
    if any(field not in inputs for field in required_fields):
        print("Skipping 3D examples plot: missing pixel fields in dataloader inputs.")
        return

    idx = select_example_indices(inputs, targets, num_examples)
    max_examples = int(idx.numel())
    if max_examples <= 0:
        return

    nrows = max(1, int(nrows))
    ncols = max(2, int(ncols))
    total_slots = nrows * ncols
    example_slots = max(1, total_slots - 1)
    if max_examples > example_slots:
        nrows = math.ceil((max_examples + 1) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.15, nrows * 2.15))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(-1)
    example_axes = axes[:-1]
    key_ax = axes[-1]
    max_examples = min(max_examples, len(example_axes))

    norm = colors.LogNorm(vmin=0.01, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    plane_z = 0.0
    marker_radius = 0.28
    arrow_lift = plane_z + 1e-4
    arrow_length = 2.6

    def project_point(x: float, y: float, z: float):
        az = np.deg2rad(-58.0)
        el = np.deg2rad(34.0)
        cam_dist = 9.0

        x1 = np.cos(az) * x - np.sin(az) * y
        y1 = np.sin(az) * x + np.cos(az) * y
        z1 = z

        x2 = x1
        y2 = np.cos(el) * y1 + np.sin(el) * z1
        z2 = -np.sin(el) * y1 + np.cos(el) * z1
        denom = max(1e-6, cam_dist - z2)
        scale = cam_dist / denom
        return x2 * scale, y2 * scale

    def draw_projected_arrow(ax, start_xyz, end_xyz, end_plane_xyz, color):
        us, vs = project_point(*start_xyz)
        ue, ve = project_point(*end_xyz)
        up, vp = project_point(*end_plane_xyz)

        direction = np.array([ue - us, ve - vs], dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return us, vs, ue, ve
        direction /= direction_norm
        normal = np.array([-direction[1], direction[0]])

        head_len = max(0.22, min(0.46, 0.27 * direction_norm))
        head_width = 1.0 * head_len
        shaft_end = np.array([ue, ve]) - direction * head_len

        ax.plot([us, up], [vs, vp], color=color, alpha=0.45, linewidth=1.2, linestyle=(0, (2, 2)), zorder=5)
        ax.plot([us, shaft_end[0]], [vs, shaft_end[1]], color=color, linewidth=2.8, zorder=11, solid_capstyle="round")

        head = np.array(
            [
                [ue, ve],
                [shaft_end[0] + 0.52 * head_width * normal[0], shaft_end[1] + 0.52 * head_width * normal[1]],
                [shaft_end[0] - 0.52 * head_width * normal[0], shaft_end[1] - 0.52 * head_width * normal[1]],
            ]
        )
        ax.add_patch(
            Polygon(
                head,
                closed=True,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                joinstyle="miter",
                zorder=12,
            )
        )
        return us, vs, ue, ve

    for ax_idx in range(max_examples):
        ax = example_axes[ax_idx]
        sample_idx = int(idx[ax_idx])
        pix_valid = inputs["pixel_valid"][sample_idx].to(torch.bool)
        pix_x = inputs["pixel_x"][sample_idx][pix_valid]
        pix_y = inputs["pixel_y"][sample_idx][pix_valid]
        pix_charge = inputs["pixel_charge"][sample_idx][pix_valid]

        if len(pix_x) == 0:
            ax.axis("off")
            continue

        u_vals = []
        v_vals = []

        part_valid = targets["particle_valid"][sample_idx].to(torch.bool)
        particles_x = targets["particle_x"][sample_idx][part_valid]
        particles_y = targets["particle_y"][sample_idx][part_valid]
        particles_phi = targets["particle_phi"][sample_idx][part_valid]
        particles_theta = targets["particle_theta"][sample_idx][part_valid]
        particles_primary = targets["particle_primary"][sample_idx][part_valid].to(torch.bool)
        particles_secondary = targets["particle_secondary"][sample_idx][part_valid].to(torch.bool)
        particles_notruth = targets["particle_notruth"][sample_idx][part_valid].to(torch.bool)

        grid_x_min, grid_x_max, grid_y_min, grid_y_max, _, _, _, _ = _grid_bounds(
            [pix_x, particles_x],
            [pix_y, particles_y],
        )

        x_boundary_min = grid_x_min - 0.5
        x_boundary_max = grid_x_max + 0.5
        y_boundary_min = grid_y_min - 0.5
        y_boundary_max = grid_y_max + 0.5

        for gx in np.arange(x_boundary_min, x_boundary_max + 1e-9, 1.0):
            u0, v0 = project_point(gx, y_boundary_min, plane_z)
            u1, v1 = project_point(gx, y_boundary_max, plane_z)
            ax.plot([u0, u1], [v0, v1], color="black", alpha=0.25, linewidth=0.9, zorder=0)
            u_vals.extend([u0, u1])
            v_vals.extend([v0, v1])
        for gy in np.arange(y_boundary_min, y_boundary_max + 1e-9, 1.0):
            u0, v0 = project_point(x_boundary_min, gy, plane_z)
            u1, v1 = project_point(x_boundary_max, gy, plane_z)
            ax.plot([u0, u1], [v0, v1], color="black", alpha=0.25, linewidth=0.9, zorder=0)
            u_vals.extend([u0, u1])
            v_vals.extend([v0, v1])

        for pixel_idx in range(len(pix_x)):
            x = pix_x[pixel_idx].item()
            y = pix_y[pixel_idx].item()
            charge = max(pix_charge[pixel_idx].item(), 1e-6)
            corners = [
                project_point(x - 0.5, y - 0.5, plane_z),
                project_point(x + 0.5, y - 0.5, plane_z),
                project_point(x + 0.5, y + 0.5, plane_z),
                project_point(x - 0.5, y + 0.5, plane_z),
            ]
            tile = Polygon(
                corners,
                closed=True,
                facecolor=cmap(norm(charge)),
                edgecolor="none",
                linewidth=0.0,
                zorder=1,
            )
            ax.add_patch(tile)
            u_vals.extend([point[0] for point in corners])
            v_vals.extend([point[1] for point in corners])

        for particle_idx in range(len(particles_x)):
            if particles_primary[particle_idx]:
                color = "crimson"
            elif particles_secondary[particle_idx]:
                color = "darkorange"
            elif particles_notruth[particle_idx]:
                color = "darkgray"
            else:
                color = "white"

            px = particles_x[particle_idx].item()
            py = particles_y[particle_idx].item()
            pz = arrow_lift
            phi = particles_phi[particle_idx].item()
            theta = particles_theta[particle_idx].item()

            marker_angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
            marker_points = [
                project_point(
                    px + marker_radius * np.cos(angle),
                    py + marker_radius * np.sin(angle),
                    plane_z,
                )
                for angle in marker_angles
            ]
            marker = Polygon(
                marker_points,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=1.3,
                zorder=8,
            )

            vec = np.array([phi, theta, 1.0], dtype=float)
            vec /= max(1e-6, np.linalg.norm(vec))

            start_x = px
            start_y = py
            start_z = pz
            end_x = start_x + arrow_length * vec[0]
            end_y = start_y + arrow_length * vec[1]
            end_z = start_z + arrow_length * vec[2]
            end_plane_x = start_x + arrow_length * vec[0]
            end_plane_y = start_y + arrow_length * vec[1]
            end_plane_z = plane_z

            um, vm = project_point(px, py, plane_z)
            us, vs, ue, ve = draw_projected_arrow(
                ax,
                (start_x, start_y, start_z),
                (end_x, end_y, end_z),
                (end_plane_x, end_plane_y, end_plane_z),
                color,
            )
            ax.add_patch(marker)
            u_vals.append(um)
            v_vals.append(vm)
            u_vals.extend([us, ue])
            v_vals.extend([vs, ve])

        if u_vals and v_vals:
            pad_u = 0.45
            pad_v = 0.45
            ax.set_xlim(min(u_vals) - pad_u, max(u_vals) + pad_u)
            ax.set_ylim(min(v_vals) - pad_v, max(v_vals) + pad_v)
        ax.set_aspect("equal")
        if show_cluster_pitch_vector_title and "cluster_pitch_vector" in inputs:
            ax.set_title(_cluster_pitch_vector_title(inputs["cluster_pitch_vector"][sample_idx]), fontsize=5.5, pad=2.5)
        ax.axis("off")

    for axis in example_axes[max_examples:]:
        axis.axis("off")

    key_ax.axis("off")
    custom_markers = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markeredgecolor="black", markersize=10)
        for label, color in [("Primary", "crimson"), ("Secondary", "darkorange"), ("No Truth", "darkgray")]
    ]
    key_ax.legend(handles=custom_markers, loc="center", bbox_to_anchor=(0.5, 0.62), fontsize=8, frameon=False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = key_ax.inset_axes([0.14, 0.30, 0.72, 0.12])
    colorbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    colorbar.set_label("Pixel Charge [ke / 100]", fontsize=8)

    subplot_top = 0.93 if show_cluster_pitch_vector_title and "cluster_pitch_vector" in inputs else 0.97
    subplot_hspace = 0.08 if show_cluster_pitch_vector_title and "cluster_pitch_vector" in inputs else 0.015
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=subplot_top, wspace=0.015, hspace=subplot_hspace)
    _finalize(fig, output_dir / "examples_3d.png", show, plt, tight=False)
