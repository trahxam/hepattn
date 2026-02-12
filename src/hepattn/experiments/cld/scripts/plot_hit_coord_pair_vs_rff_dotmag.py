import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule

BASE_COORD_NAMES = ["x", "y", "z", "r", "eta", "phi", "sinphi", "cosphi"]
BASE_COORD_LABELS = {
    "x": "x",
    "y": "y",
    "z": "z",
    "r": "r",
    "eta": "eta",
    "phi": "phi",
    "sinphi": "sin(phi)",
    "cosphi": "cos(phi)",
}

GROUP_SPECS = [
    {
        "category": "single",
        "name": "vtxd",
        "subsystems": ["vtxd"],
        "title": "VTXD",
    },
    {
        "category": "single",
        "name": "trkr",
        "subsystems": ["trkr"],
        "title": "TRKR",
    },
    {
        "category": "single",
        "name": "ecal",
        "subsystems": ["ecal"],
        "title": "ECAL",
    },
    {
        "category": "single",
        "name": "hcal",
        "subsystems": ["hcal"],
        "title": "HCAL",
    },
    {
        "category": "combined",
        "name": "all",
        "subsystems": ["vtxd", "trkr", "ecal", "hcal"],
        "title": "ALL (vtxd+trkr+ecal+hcal)",
    },
    {
        "category": "combined",
        "name": "vtxd_trkr",
        "subsystems": ["vtxd", "trkr"],
        "title": "VTXD+TRKR",
    },
    {
        "category": "combined",
        "name": "trkr_ecal",
        "subsystems": ["trkr", "ecal"],
        "title": "TRKR+ECAL",
    },
    {
        "category": "combined",
        "name": "ecal_hcal",
        "subsystems": ["ecal", "hcal"],
        "title": "ECAL+HCAL",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CLD hit-pair diagnostics: for each coordinate, build a 2D histogram over "
            "(coord_i, coord_j) with color set by mean |normalized RFF dot|. "
            "Produces plots for requested subsystem groups using unscaled coordinates, "
            "plus directed cross-system plots for two-subsystem groups."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/hepattn/experiments/cld/configs/unified.yaml"),
    )
    parser.add_argument("--event-index", type=int, default=0, help="Zero-based start index in test dataloader.")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events to aggregate.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--coords",
        type=str,
        nargs="+",
        choices=BASE_COORD_NAMES,
        default=BASE_COORD_NAMES,
        help="Coordinates to plot.",
    )
    parser.add_argument("--dim-per-field", type=int, default=32, help="RFF embedding width per coordinate.")
    parser.add_argument("--rff-scale", type=float, default=1.0, help="RFF scale for random Fourier features.")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help=("Max unique hit pairs per event. <= 0 uses all unique pairs. When sampled, pairs are mirrored to include both (i,j) and (j,i)."),
    )
    parser.add_argument(
        "--pair-block-size",
        type=int,
        default=1024,
        help="Block size for exact all-pairs accumulation.",
    )
    parser.add_argument("--num-bins", type=int, default=120, help="Histogram bins per coordinate axis.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src/hepattn/experiments/cld/plots/data"),
    )
    parser.add_argument("--out-prefix", type=str, default="hit_coord_pair_vs_rff_dotmag")
    return parser.parse_args()


def load_events(config_path, event_index, num_events, num_workers):
    cfg = yaml.safe_load(config_path.read_text())["data"]
    cfg["num_workers"] = num_workers
    cfg["batch_size"] = 1
    cfg["num_test"] = max(event_index + num_events, 1)

    datamodule = CLDDataModule(**cfg)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    data_iter = iter(dataloader)
    events = []
    try:
        for idx in range(event_index + num_events):
            inputs, targets = next(data_iter)
            if idx >= event_index:
                events.append((inputs, targets))
    except StopIteration:
        pass

    if len(events) < num_events:
        raise RuntimeError(f"Requested {num_events} events from index {event_index}, but only loaded {len(events)}")

    return events


def init_rff_base_matrices(coord_names, dim_per_field, seed):
    half_dim = dim_per_field // 2
    if half_dim == 0:
        raise ValueError("--dim-per-field must be >= 2 for random Fourier features")

    gen = torch.Generator()
    gen.manual_seed(seed)
    return {coord_name: torch.randn((1, half_dim), generator=gen) for coord_name in sorted(coord_names)}


def random_fourier_pos_enc(xs, base_matrix, scale, dim_per_field):
    matrix = ((1.0 / scale) * base_matrix).to(device=xs.device, dtype=xs.dtype)
    proj = (2.0 * torch.pi * xs.unsqueeze(-1)) * matrix
    enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    if enc.shape[-1] < dim_per_field:
        pad = torch.zeros((*enc.shape[:-1], dim_per_field - enc.shape[-1]), dtype=enc.dtype, device=enc.device)
        enc = torch.cat([enc, pad], dim=-1)
    return enc


def pair_dot_products(embeddings, pairs):
    if pairs.numel() == 0:
        return torch.empty((0,), dtype=embeddings.dtype)

    left = embeddings[pairs[:, 0]]
    right = embeddings[pairs[:, 1]]
    dot = (left * right).sum(dim=-1)
    emb_dim = int(embeddings.shape[-1])
    if emb_dim <= 0:
        return dot
    return dot / float(emb_dim)


def pair_id_to_pair_indices(num_hits, pair_ids):
    pair_ids = np.asarray(pair_ids, dtype=np.int64)
    counts = np.arange(num_hits - 1, 0, -1, dtype=np.int64)
    prefix = np.empty(num_hits, dtype=np.int64)
    prefix[0] = 0
    prefix[1:] = np.cumsum(counts, dtype=np.int64)

    idx_i = np.searchsorted(prefix, pair_ids, side="right") - 1
    offset = pair_ids - prefix[idx_i]
    idx_j = idx_i + 1 + offset
    return np.stack((idx_i, idx_j), axis=1)


def sample_pairs_without_replacement(num_hits, max_pairs, rng):
    total_pairs = int(num_hits * (num_hits - 1) // 2)
    target_pairs = min(int(max_pairs), total_pairs)
    if target_pairs <= 0:
        return torch.empty((0, 2), dtype=torch.long), total_pairs

    pair_ids = rng.choice(total_pairs, size=target_pairs, replace=False)
    pair_indices = pair_id_to_pair_indices(num_hits=num_hits, pair_ids=pair_ids)
    return torch.as_tensor(pair_indices, dtype=torch.long), total_pairs


def sample_rect_pairs_without_replacement(num_source_hits, num_target_hits, max_pairs, rng):
    total_pairs = int(num_source_hits * num_target_hits)
    target_pairs = min(int(max_pairs), total_pairs)
    if target_pairs <= 0:
        return torch.empty((0, 2), dtype=torch.long), total_pairs

    pair_ids = rng.choice(total_pairs, size=target_pairs, replace=False)
    source_idx = pair_ids // num_target_hits
    target_idx = pair_ids % num_target_hits
    pair_indices = np.stack((source_idx, target_idx), axis=1)
    return torch.as_tensor(pair_indices, dtype=torch.long), total_pairs


def build_coord_edges_from_tensors(coord_name, values_iter, num_bins):
    if coord_name == "phi":
        return np.linspace(-np.pi, np.pi, num_bins + 1)
    if coord_name in {"sinphi", "cosphi"}:
        return np.linspace(-1.0, 1.0, num_bins + 1)

    min_coord = np.inf
    max_coord = -np.inf
    for values in values_iter:
        if values.numel() == 0:
            continue
        min_coord = min(min_coord, float(values.min().item()))
        max_coord = max(max_coord, float(values.max().item()))

    if not np.isfinite(min_coord) or not np.isfinite(max_coord):
        min_coord, max_coord = -1.0, 1.0

    if min_coord == max_coord:
        span = max(abs(min_coord), 1e-3)
        min_coord -= 0.05 * span
        max_coord += 0.05 * span
    else:
        pad = 0.02 * (max_coord - min_coord)
        min_coord -= pad
        max_coord += pad

    return np.linspace(min_coord, max_coord, num_bins + 1)


def build_coord_edges(event_data, coord_name, num_bins):
    return build_coord_edges_from_tensors(
        coord_name=coord_name,
        values_iter=(payload["coords"][coord_name] for payload in event_data),
        num_bins=num_bins,
    )


def accumulate_hist_sampled_pairs(values, embeddings, pairs, coord_edges):
    dotmag_sum = np.zeros((len(coord_edges) - 1, len(coord_edges) - 1), dtype=np.float64)
    pair_counts = np.zeros((len(coord_edges) - 1, len(coord_edges) - 1), dtype=np.int64)
    if pairs.numel() == 0:
        return dotmag_sum, pair_counts

    x_forward = values[pairs[:, 0]]
    y_forward = values[pairs[:, 1]]
    dot_magnitude = pair_dot_products(embeddings, pairs).abs()

    x_coords = torch.cat([x_forward, y_forward], dim=0)
    y_coords = torch.cat([y_forward, x_forward], dim=0)
    dot_magnitude = torch.cat([dot_magnitude, dot_magnitude], dim=0)

    x_np = x_coords.detach().cpu().numpy()
    y_np = y_coords.detach().cpu().numpy()
    dotmag_np = dot_magnitude.detach().cpu().numpy()

    counts_hist, _, _ = np.histogram2d(x_np, y_np, bins=(coord_edges, coord_edges))
    sum_hist, _, _ = np.histogram2d(x_np, y_np, bins=(coord_edges, coord_edges), weights=dotmag_np)
    pair_counts += counts_hist.astype(np.int64)
    dotmag_sum += sum_hist.astype(np.float64)
    return dotmag_sum, pair_counts


def accumulate_hist_all_pairs(values, embeddings, coord_edges, pair_block_size):
    dotmag_sum = np.zeros((len(coord_edges) - 1, len(coord_edges) - 1), dtype=np.float64)
    pair_counts = np.zeros((len(coord_edges) - 1, len(coord_edges) - 1), dtype=np.int64)
    num_hits = int(values.shape[0])
    emb_dim = int(embeddings.shape[-1])

    if num_hits < 2 or emb_dim <= 0:
        return dotmag_sum, pair_counts

    for i_start in range(0, num_hits, pair_block_size):
        i_end = min(num_hits, i_start + pair_block_size)
        values_i = values[i_start:i_end]
        embeddings_i = embeddings[i_start:i_end]

        for j_start in range(i_start, num_hits, pair_block_size):
            j_end = min(num_hits, j_start + pair_block_size)
            values_j = values[j_start:j_end]
            embeddings_j = embeddings[j_start:j_end]

            dots = ((embeddings_i @ embeddings_j.T) / float(emb_dim)).abs()

            if i_start == j_start:
                if values_i.shape[0] < 2:
                    continue
                tri_rows, tri_cols = torch.triu_indices(values_i.shape[0], values_j.shape[0], offset=1)
                x_upper = values_i[tri_rows]
                y_upper = values_j[tri_cols]
                dots_upper = dots[tri_rows, tri_cols]
            else:
                x_upper = values_i.repeat_interleave(values_j.shape[0])
                y_upper = values_j.repeat(values_i.shape[0])
                dots_upper = dots.reshape(-1)

            x_coords = torch.cat([x_upper, y_upper], dim=0)
            y_coords = torch.cat([y_upper, x_upper], dim=0)
            dots_flat = torch.cat([dots_upper, dots_upper], dim=0)

            if dots_flat.numel() == 0:
                continue

            x_np = x_coords.detach().cpu().numpy()
            y_np = y_coords.detach().cpu().numpy()
            dots_np = dots_flat.detach().cpu().numpy()

            counts_hist, _, _ = np.histogram2d(x_np, y_np, bins=(coord_edges, coord_edges))
            sum_hist, _, _ = np.histogram2d(x_np, y_np, bins=(coord_edges, coord_edges), weights=dots_np)
            pair_counts += counts_hist.astype(np.int64)
            dotmag_sum += sum_hist.astype(np.float64)

    return dotmag_sum, pair_counts


def accumulate_hist_cross_sampled_pairs(source_values, target_values, source_embeddings, target_embeddings, pairs, source_edges, target_edges):
    dotmag_sum = np.zeros((len(source_edges) - 1, len(target_edges) - 1), dtype=np.float64)
    pair_counts = np.zeros((len(source_edges) - 1, len(target_edges) - 1), dtype=np.int64)
    if pairs.numel() == 0:
        return dotmag_sum, pair_counts

    source_idx = pairs[:, 0]
    target_idx = pairs[:, 1]
    x_coords = source_values[source_idx]
    y_coords = target_values[target_idx]

    left = source_embeddings[source_idx]
    right = target_embeddings[target_idx]
    emb_dim = int(source_embeddings.shape[-1])
    if emb_dim <= 0:
        return dotmag_sum, pair_counts
    dot_magnitude = ((left * right).sum(dim=-1) / float(emb_dim)).abs()

    x_np = x_coords.detach().cpu().numpy()
    y_np = y_coords.detach().cpu().numpy()
    dotmag_np = dot_magnitude.detach().cpu().numpy()

    counts_hist, _, _ = np.histogram2d(x_np, y_np, bins=(source_edges, target_edges))
    sum_hist, _, _ = np.histogram2d(x_np, y_np, bins=(source_edges, target_edges), weights=dotmag_np)
    pair_counts += counts_hist.astype(np.int64)
    dotmag_sum += sum_hist.astype(np.float64)
    return dotmag_sum, pair_counts


def accumulate_hist_cross_all_pairs(
    source_values,
    target_values,
    source_embeddings,
    target_embeddings,
    source_edges,
    target_edges,
    pair_block_size,
):
    dotmag_sum = np.zeros((len(source_edges) - 1, len(target_edges) - 1), dtype=np.float64)
    pair_counts = np.zeros((len(source_edges) - 1, len(target_edges) - 1), dtype=np.int64)
    num_source_hits = int(source_values.shape[0])
    num_target_hits = int(target_values.shape[0])
    emb_dim = int(source_embeddings.shape[-1])

    if num_source_hits < 1 or num_target_hits < 1 or emb_dim <= 0:
        return dotmag_sum, pair_counts

    for i_start in range(0, num_source_hits, pair_block_size):
        i_end = min(num_source_hits, i_start + pair_block_size)
        source_vals_block = source_values[i_start:i_end]
        source_emb_block = source_embeddings[i_start:i_end]

        for j_start in range(0, num_target_hits, pair_block_size):
            j_end = min(num_target_hits, j_start + pair_block_size)
            target_vals_block = target_values[j_start:j_end]
            target_emb_block = target_embeddings[j_start:j_end]

            dots = ((source_emb_block @ target_emb_block.T) / float(emb_dim)).abs()
            x_coords = source_vals_block.repeat_interleave(target_vals_block.shape[0])
            y_coords = target_vals_block.repeat(source_vals_block.shape[0])
            dots_flat = dots.reshape(-1)
            if dots_flat.numel() == 0:
                continue

            x_np = x_coords.detach().cpu().numpy()
            y_np = y_coords.detach().cpu().numpy()
            dots_np = dots_flat.detach().cpu().numpy()

            counts_hist, _, _ = np.histogram2d(x_np, y_np, bins=(source_edges, target_edges))
            sum_hist, _, _ = np.histogram2d(x_np, y_np, bins=(source_edges, target_edges), weights=dots_np)
            pair_counts += counts_hist.astype(np.int64)
            dotmag_sum += sum_hist.astype(np.float64)

    return dotmag_sum, pair_counts


def _plot_single_histogram(
    ax,
    coord_name,
    dotmag_sum,
    pair_counts,
    coord_edges,
    xlabel_fontsize=10,
    title_fontsize=11,
):
    label = BASE_COORD_LABELS[coord_name]
    mean_dotmag = np.full(pair_counts.shape, np.nan, dtype=np.float64)
    nonzero = pair_counts > 0
    if np.any(nonzero):
        mean_dotmag[nonzero] = dotmag_sum[nonzero] / pair_counts[nonzero]
    mean_dotmag = np.clip(mean_dotmag, 0.0, 1.0)
    masked = np.ma.masked_invalid(mean_dotmag)

    if masked.count() > 0:
        mesh = ax.pcolormesh(
            coord_edges,
            coord_edges,
            masked.T,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Mean |normalized RFF dot|")
    else:
        ax.text(0.5, 0.5, "No pairs in range", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(f"{label}_i", fontsize=xlabel_fontsize)
    ax.set_ylabel(f"{label}_j", fontsize=xlabel_fontsize)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_title(f"{label}: coord_i vs coord_j | color=mean |dot|", fontsize=title_fontsize)


def plot_coord_histogram(
    coord_name,
    dotmag_sum,
    pair_counts,
    coord_edges,
    out_path,
    group_title,
    used_events,
    total_pairs_used,
    total_pairs_available,
):
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    _plot_single_histogram(
        ax=ax,
        coord_name=coord_name,
        dotmag_sum=dotmag_sum,
        pair_counts=pair_counts,
        coord_edges=coord_edges,
    )
    fig.suptitle(
        (f"CLD hit pairs [{group_title}] | events={used_events} | pairs used={total_pairs_used:,} / available={total_pairs_available:,}"),
        fontsize=11,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_coord_histogram_grid(
    coord_names,
    dotmag_sum_by_coord,
    pair_counts_by_coord,
    coord_edges_by_coord,
    out_path,
    group_title,
    used_events,
    total_pairs_used,
    total_pairs_available,
):
    num_coords = len(coord_names)
    ncols = 3
    nrows = int(np.ceil(num_coords / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False)
    axes_list = list(axes.flat)

    for ax, coord_name in zip(axes_list, coord_names, strict=False):
        _plot_single_histogram(
            ax=ax,
            coord_name=coord_name,
            dotmag_sum=dotmag_sum_by_coord[coord_name],
            pair_counts=pair_counts_by_coord[coord_name],
            coord_edges=coord_edges_by_coord[coord_name],
            xlabel_fontsize=9,
            title_fontsize=10,
        )

    for ax in axes_list[num_coords:]:
        ax.axis("off")

    fig.suptitle(
        (
            f"CLD hit-pair coordinates [{group_title}] | color=mean |RFF dot| "
            f"| events={used_events} | pairs used={total_pairs_used:,} / available={total_pairs_available:,}"
        ),
        fontsize=12,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_single_cross_histogram(
    ax,
    coord_name,
    dotmag_sum,
    pair_counts,
    source_edges,
    target_edges,
    source_subsystem,
    target_subsystem,
    xlabel_fontsize=10,
    title_fontsize=11,
):
    label = BASE_COORD_LABELS[coord_name]
    mean_dotmag = np.full(pair_counts.shape, np.nan, dtype=np.float64)
    nonzero = pair_counts > 0
    if np.any(nonzero):
        mean_dotmag[nonzero] = dotmag_sum[nonzero] / pair_counts[nonzero]
    mean_dotmag = np.clip(mean_dotmag, 0.0, 1.0)
    masked = np.ma.masked_invalid(mean_dotmag)

    if masked.count() > 0:
        mesh = ax.pcolormesh(
            source_edges,
            target_edges,
            masked.T,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Mean |normalized RFF dot|")
    else:
        ax.text(0.5, 0.5, "No pairs in range", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(f"{label}_{source_subsystem}", fontsize=xlabel_fontsize)
    ax.set_ylabel(f"{label}_{target_subsystem}", fontsize=xlabel_fontsize)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_title(
        f"{label}: {source_subsystem}->{target_subsystem} | color=mean |dot|",
        fontsize=title_fontsize,
    )


def plot_cross_coord_histogram(
    coord_name,
    dotmag_sum,
    pair_counts,
    source_edges,
    target_edges,
    out_path,
    group_title,
    source_subsystem,
    target_subsystem,
    used_events,
    total_pairs_used,
    total_pairs_available,
):
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    _plot_single_cross_histogram(
        ax=ax,
        coord_name=coord_name,
        dotmag_sum=dotmag_sum,
        pair_counts=pair_counts,
        source_edges=source_edges,
        target_edges=target_edges,
        source_subsystem=source_subsystem,
        target_subsystem=target_subsystem,
    )
    fig.suptitle(
        (
            f"CLD cross-type hit pairs [{group_title}] | {source_subsystem}->{target_subsystem} "
            f"| events={used_events} | pairs used={total_pairs_used:,} / available={total_pairs_available:,}"
        ),
        fontsize=11,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_cross_coord_histogram_grid(
    coord_names,
    dotmag_sum_by_coord,
    pair_counts_by_coord,
    source_edges_by_coord,
    target_edges_by_coord,
    out_path,
    group_title,
    source_subsystem,
    target_subsystem,
    used_events,
    total_pairs_used,
    total_pairs_available,
):
    num_coords = len(coord_names)
    ncols = 3
    nrows = int(np.ceil(num_coords / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False)
    axes_list = list(axes.flat)

    for ax, coord_name in zip(axes_list, coord_names, strict=False):
        _plot_single_cross_histogram(
            ax=ax,
            coord_name=coord_name,
            dotmag_sum=dotmag_sum_by_coord[coord_name],
            pair_counts=pair_counts_by_coord[coord_name],
            source_edges=source_edges_by_coord[coord_name],
            target_edges=target_edges_by_coord[coord_name],
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
            xlabel_fontsize=9,
            title_fontsize=10,
        )

    for ax in axes_list[num_coords:]:
        ax.axis("off")

    fig.suptitle(
        (
            f"CLD cross-type coordinates [{group_title}] | {source_subsystem}->{target_subsystem} "
            f"| color=mean |RFF dot| | events={used_events} "
            f"| pairs used={total_pairs_used:,} / available={total_pairs_available:,}"
        ),
        fontsize=12,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_text_report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def extract_subsystem_coords(inputs, subsystem):
    hit_valid_key = f"{subsystem}_valid"
    required_pos_keys = [
        f"{subsystem}_pos.x",
        f"{subsystem}_pos.y",
        f"{subsystem}_pos.z",
        f"{subsystem}_pos.r",
        f"{subsystem}_pos.eta",
        f"{subsystem}_pos.phi",
    ]

    if hit_valid_key not in inputs:
        return None, "missing input valid mask"
    missing = [key for key in required_pos_keys if key not in inputs]
    if missing:
        return None, f"missing coordinate fields: {', '.join(missing)}"

    hit_valid = inputs[hit_valid_key][0].bool()
    num_hits = int(hit_valid.sum().item())
    if num_hits < 1:
        return None, "fewer than 1 valid hit"

    phi = inputs[f"{subsystem}_pos.phi"][0][hit_valid].float()
    coords = {
        "x": inputs[f"{subsystem}_pos.x"][0][hit_valid].float(),
        "y": inputs[f"{subsystem}_pos.y"][0][hit_valid].float(),
        "z": inputs[f"{subsystem}_pos.z"][0][hit_valid].float(),
        "r": inputs[f"{subsystem}_pos.r"][0][hit_valid].float(),
        "eta": inputs[f"{subsystem}_pos.eta"][0][hit_valid].float(),
        "phi": phi,
        "sinphi": torch.sin(phi),
        "cosphi": torch.cos(phi),
    }
    return {"num_hits": num_hits, "coords": coords}, None


def build_group_event_payload(inputs, group_subsystems):
    subsystem_views = []
    for subsystem in group_subsystems:
        subsystem_view, reason = extract_subsystem_coords(inputs=inputs, subsystem=subsystem)
        if subsystem_view is None:
            return None, f"{subsystem}: {reason}"
        subsystem_views.append(subsystem_view)

    total_hits = int(sum(view["num_hits"] for view in subsystem_views))
    if total_hits < 2:
        return None, "fewer than 2 total hits"

    coords = {}
    for coord_name in BASE_COORD_NAMES:
        coords[coord_name] = torch.cat([view["coords"][coord_name] for view in subsystem_views], dim=0)

    return {"num_hits": total_hits, "coords": coords}, None


def build_group_event_data(events, group_subsystems):
    event_data = []
    skipped_events = 0
    skipped_reasons = {}
    total_hits = 0

    for inputs, _targets in events:
        payload, reason = build_group_event_payload(inputs=inputs, group_subsystems=group_subsystems)
        if payload is None:
            skipped_events += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        event_data.append(payload)
        total_hits += payload["num_hits"]

    return event_data, skipped_events, skipped_reasons, total_hits


def build_cross_group_event_payload(inputs, source_subsystem, target_subsystem):
    source_view, source_reason = extract_subsystem_coords(inputs=inputs, subsystem=source_subsystem)
    if source_view is None:
        return None, f"{source_subsystem}: {source_reason}"

    target_view, target_reason = extract_subsystem_coords(inputs=inputs, subsystem=target_subsystem)
    if target_view is None:
        return None, f"{target_subsystem}: {target_reason}"

    return {
        "num_source_hits": source_view["num_hits"],
        "num_target_hits": target_view["num_hits"],
        "source_coords": source_view["coords"],
        "target_coords": target_view["coords"],
    }, None


def build_cross_group_event_data(events, source_subsystem, target_subsystem):
    event_data = []
    skipped_events = 0
    skipped_reasons = {}
    total_source_hits = 0
    total_target_hits = 0

    for inputs, _targets in events:
        payload, reason = build_cross_group_event_payload(
            inputs=inputs,
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
        )
        if payload is None:
            skipped_events += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        event_data.append(payload)
        total_source_hits += payload["num_source_hits"]
        total_target_hits += payload["num_target_hits"]

    return event_data, skipped_events, skipped_reasons, total_source_hits, total_target_hits


def build_pair_sampling_plan(event_data, max_pairs, rng, group_name):
    pair_plan = []
    total_pairs_available = 0
    total_pairs_used = 0
    sampled_events = 0

    for event_idx, payload in enumerate(event_data):
        num_hits = payload["num_hits"]
        available_pairs = int(num_hits * (num_hits - 1))
        sampled_pairs = None
        used_pairs = available_pairs

        if available_pairs > 0:
            unique_pairs_available = available_pairs // 2
            if max_pairs > 0 and unique_pairs_available > max_pairs:
                sampled_pairs, _ = sample_pairs_without_replacement(num_hits=num_hits, max_pairs=max_pairs, rng=rng)
                used_pairs = int(2 * sampled_pairs.shape[0])
                sampled_events += 1

            pair_mode = "sampled" if sampled_pairs is not None else "all"
            print(f"[{group_name}] Event {event_idx}: hits={num_hits:,}, pairs used={used_pairs:,} / available={available_pairs:,} ({pair_mode})")

        pair_plan.append({
            "sampled_pairs": sampled_pairs,
            "available_pairs": available_pairs,
            "used_pairs": used_pairs,
        })
        total_pairs_available += available_pairs
        total_pairs_used += used_pairs

    return pair_plan, total_pairs_used, total_pairs_available, sampled_events


def build_cross_pair_sampling_plan(event_data, max_pairs, rng, group_name, source_subsystem, target_subsystem):
    pair_plan = []
    total_pairs_available = 0
    total_pairs_used = 0
    sampled_events = 0

    for event_idx, payload in enumerate(event_data):
        num_source_hits = payload["num_source_hits"]
        num_target_hits = payload["num_target_hits"]
        available_pairs = int(num_source_hits * num_target_hits)
        sampled_pairs = None
        used_pairs = available_pairs

        if available_pairs > 0:
            if max_pairs > 0 and available_pairs > max_pairs:
                sampled_pairs, _ = sample_rect_pairs_without_replacement(
                    num_source_hits=num_source_hits,
                    num_target_hits=num_target_hits,
                    max_pairs=max_pairs,
                    rng=rng,
                )
                used_pairs = int(sampled_pairs.shape[0])
                sampled_events += 1

            pair_mode = "sampled" if sampled_pairs is not None else "all"
            print(
                (
                    f"[cross:{group_name}] Event {event_idx}: "
                    f"{source_subsystem}_hits={num_source_hits:,}, {target_subsystem}_hits={num_target_hits:,}, "
                    f"pairs used={used_pairs:,} / available={available_pairs:,} ({pair_mode})"
                )
            )

        pair_plan.append({
            "sampled_pairs": sampled_pairs,
            "available_pairs": available_pairs,
            "used_pairs": used_pairs,
        })
        total_pairs_available += available_pairs
        total_pairs_used += used_pairs

    return pair_plan, total_pairs_used, total_pairs_available, sampled_events


def run_group_plots(
    *,
    group_spec,
    event_data,
    pair_plan,
    coord_names,
    rff_base_matrices,
    args,
    output_root,
    used_events,
    skipped_events,
    total_hits,
    skipped_reasons,
    total_pairs_used,
    total_pairs_available,
    sampled_events,
):
    coord_edges_by_coord = {
        coord_name: build_coord_edges(event_data=event_data, coord_name=coord_name, num_bins=args.num_bins) for coord_name in coord_names
    }
    dotmag_sum_by_coord = {
        coord_name: np.zeros((len(coord_edges_by_coord[coord_name]) - 1, len(coord_edges_by_coord[coord_name]) - 1), dtype=np.float64)
        for coord_name in coord_names
    }
    pair_counts_by_coord = {
        coord_name: np.zeros((len(coord_edges_by_coord[coord_name]) - 1, len(coord_edges_by_coord[coord_name]) - 1), dtype=np.int64)
        for coord_name in coord_names
    }

    for payload, event_pair_plan in zip(event_data, pair_plan, strict=False):
        if event_pair_plan["used_pairs"] <= 0:
            continue

        sampled_pairs = event_pair_plan["sampled_pairs"]
        for coord_name in coord_names:
            values = payload["coords"][coord_name]
            embeddings = random_fourier_pos_enc(
                xs=values,
                base_matrix=rff_base_matrices[coord_name],
                scale=args.rff_scale,
                dim_per_field=args.dim_per_field,
            )

            if sampled_pairs is None:
                dotmag_sum, pair_counts = accumulate_hist_all_pairs(
                    values=values,
                    embeddings=embeddings,
                    coord_edges=coord_edges_by_coord[coord_name],
                    pair_block_size=args.pair_block_size,
                )
            else:
                dotmag_sum, pair_counts = accumulate_hist_sampled_pairs(
                    values=values,
                    embeddings=embeddings,
                    pairs=sampled_pairs,
                    coord_edges=coord_edges_by_coord[coord_name],
                )

            dotmag_sum_by_coord[coord_name] += dotmag_sum
            pair_counts_by_coord[coord_name] += pair_counts

    group_output_root = output_root / group_spec["category"] / group_spec["name"]
    written_paths = []

    for coord_name in coord_names:
        out_path = group_output_root / "per_coord" / f"{coord_name}.png"
        plot_coord_histogram(
            coord_name=coord_name,
            dotmag_sum=dotmag_sum_by_coord[coord_name],
            pair_counts=pair_counts_by_coord[coord_name],
            coord_edges=coord_edges_by_coord[coord_name],
            out_path=out_path,
            group_title=group_spec["title"],
            used_events=used_events,
            total_pairs_used=total_pairs_used,
            total_pairs_available=total_pairs_available,
        )
        written_paths.append(out_path)

    grid_out_path = group_output_root / "all_coords.png"
    plot_coord_histogram_grid(
        coord_names=coord_names,
        dotmag_sum_by_coord=dotmag_sum_by_coord,
        pair_counts_by_coord=pair_counts_by_coord,
        coord_edges_by_coord=coord_edges_by_coord,
        out_path=grid_out_path,
        group_title=group_spec["title"],
        used_events=used_events,
        total_pairs_used=total_pairs_used,
        total_pairs_available=total_pairs_available,
    )
    written_paths.append(grid_out_path)

    report_lines = [
        "CLD coord_i vs coord_j with color=mean |RFF dot| summary",
        f"Group: {group_spec['name']} ({group_spec['title']})",
        f"Subsystems: {', '.join(group_spec['subsystems'])}",
        f"Event index: {args.event_index}",
        f"Requested events: {args.num_events}",
        f"Used/skipped events: {used_events}/{skipped_events}",
        f"Total valid hits in used events: {total_hits:,}",
        f"Coordinates: {', '.join(coord_names)}",
        f"RFF dim per coordinate: {args.dim_per_field}",
        f"RFF scale: {args.rff_scale}",
        (
            "Pair mode: all ordered pairs (i,j) and (j,i), with i!=j"
            if args.max_pairs <= 0
            else f"Pair mode: cap at {args.max_pairs:,} unique pairs/event, mirrored to ordered pairs"
        ),
        f"Events using sampled pairs: {sampled_events}",
        f"Total pairs used / available: {total_pairs_used:,} / {total_pairs_available:,}",
        f"Pair block size (all-pairs mode): {args.pair_block_size}",
        f"Coordinate bins per axis: {args.num_bins}",
        "Color metric: mean absolute normalized RFF dot product per bin",
    ]

    if skipped_reasons:
        report_lines.append("Skipped event reasons:")
        for reason, count in sorted(skipped_reasons.items()):
            report_lines.append(f"  {reason}: {count}")

    report_path = group_output_root / "summary.txt"
    write_text_report(report_path, report_lines)
    written_paths.append(report_path)

    return written_paths, report_path


def run_cross_group_plots(
    *,
    group_spec,
    event_data,
    pair_plan,
    coord_names,
    rff_base_matrices,
    args,
    output_root,
    source_subsystem,
    target_subsystem,
    used_events,
    skipped_events,
    total_source_hits,
    total_target_hits,
    skipped_reasons,
    total_pairs_used,
    total_pairs_available,
    sampled_events,
):
    source_edges_by_coord = {
        coord_name: build_coord_edges_from_tensors(
            coord_name=coord_name,
            values_iter=(payload["source_coords"][coord_name] for payload in event_data),
            num_bins=args.num_bins,
        )
        for coord_name in coord_names
    }
    target_edges_by_coord = {
        coord_name: build_coord_edges_from_tensors(
            coord_name=coord_name,
            values_iter=(payload["target_coords"][coord_name] for payload in event_data),
            num_bins=args.num_bins,
        )
        for coord_name in coord_names
    }
    dotmag_sum_by_coord = {
        coord_name: np.zeros(
            (
                len(source_edges_by_coord[coord_name]) - 1,
                len(target_edges_by_coord[coord_name]) - 1,
            ),
            dtype=np.float64,
        )
        for coord_name in coord_names
    }
    pair_counts_by_coord = {
        coord_name: np.zeros(
            (
                len(source_edges_by_coord[coord_name]) - 1,
                len(target_edges_by_coord[coord_name]) - 1,
            ),
            dtype=np.int64,
        )
        for coord_name in coord_names
    }

    for payload, event_pair_plan in zip(event_data, pair_plan, strict=False):
        if event_pair_plan["used_pairs"] <= 0:
            continue

        sampled_pairs = event_pair_plan["sampled_pairs"]
        for coord_name in coord_names:
            source_values = payload["source_coords"][coord_name]
            target_values = payload["target_coords"][coord_name]
            source_embeddings = random_fourier_pos_enc(
                xs=source_values,
                base_matrix=rff_base_matrices[coord_name],
                scale=args.rff_scale,
                dim_per_field=args.dim_per_field,
            )
            target_embeddings = random_fourier_pos_enc(
                xs=target_values,
                base_matrix=rff_base_matrices[coord_name],
                scale=args.rff_scale,
                dim_per_field=args.dim_per_field,
            )

            if sampled_pairs is None:
                dotmag_sum, pair_counts = accumulate_hist_cross_all_pairs(
                    source_values=source_values,
                    target_values=target_values,
                    source_embeddings=source_embeddings,
                    target_embeddings=target_embeddings,
                    source_edges=source_edges_by_coord[coord_name],
                    target_edges=target_edges_by_coord[coord_name],
                    pair_block_size=args.pair_block_size,
                )
            else:
                dotmag_sum, pair_counts = accumulate_hist_cross_sampled_pairs(
                    source_values=source_values,
                    target_values=target_values,
                    source_embeddings=source_embeddings,
                    target_embeddings=target_embeddings,
                    pairs=sampled_pairs,
                    source_edges=source_edges_by_coord[coord_name],
                    target_edges=target_edges_by_coord[coord_name],
                )

            dotmag_sum_by_coord[coord_name] += dotmag_sum
            pair_counts_by_coord[coord_name] += pair_counts

    group_output_root = output_root / "cross_type_ordered" / group_spec["name"]
    written_paths = []

    for coord_name in coord_names:
        out_path = group_output_root / "per_coord" / f"{coord_name}.png"
        plot_cross_coord_histogram(
            coord_name=coord_name,
            dotmag_sum=dotmag_sum_by_coord[coord_name],
            pair_counts=pair_counts_by_coord[coord_name],
            source_edges=source_edges_by_coord[coord_name],
            target_edges=target_edges_by_coord[coord_name],
            out_path=out_path,
            group_title=group_spec["title"],
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
            used_events=used_events,
            total_pairs_used=total_pairs_used,
            total_pairs_available=total_pairs_available,
        )
        written_paths.append(out_path)

    grid_out_path = group_output_root / "all_coords.png"
    plot_cross_coord_histogram_grid(
        coord_names=coord_names,
        dotmag_sum_by_coord=dotmag_sum_by_coord,
        pair_counts_by_coord=pair_counts_by_coord,
        source_edges_by_coord=source_edges_by_coord,
        target_edges_by_coord=target_edges_by_coord,
        out_path=grid_out_path,
        group_title=group_spec["title"],
        source_subsystem=source_subsystem,
        target_subsystem=target_subsystem,
        used_events=used_events,
        total_pairs_used=total_pairs_used,
        total_pairs_available=total_pairs_available,
    )
    written_paths.append(grid_out_path)

    report_lines = [
        "CLD cross-type coord_source vs coord_target with color=mean |RFF dot| summary",
        f"Group: {group_spec['name']} ({group_spec['title']})",
        f"Source subsystem (x-axis): {source_subsystem}",
        f"Target subsystem (y-axis): {target_subsystem}",
        f"Event index: {args.event_index}",
        f"Requested events: {args.num_events}",
        f"Used/skipped events: {used_events}/{skipped_events}",
        f"Total valid source hits in used events: {total_source_hits:,}",
        f"Total valid target hits in used events: {total_target_hits:,}",
        f"Coordinates: {', '.join(coord_names)}",
        f"RFF dim per coordinate: {args.dim_per_field}",
        f"RFF scale: {args.rff_scale}",
        (
            "Pair mode: all ordered cross-system pairs (source_hit, target_hit)"
            if args.max_pairs <= 0
            else f"Pair mode: cap at {args.max_pairs:,} source-target pairs/event"
        ),
        f"Events using sampled pairs: {sampled_events}",
        f"Total pairs used / available: {total_pairs_used:,} / {total_pairs_available:,}",
        f"Pair block size (all-pairs mode): {args.pair_block_size}",
        f"Coordinate bins per axis: {args.num_bins}",
        "Color metric: mean absolute normalized RFF dot product per bin",
    ]

    if skipped_reasons:
        report_lines.append("Skipped event reasons:")
        for reason, count in sorted(skipped_reasons.items()):
            report_lines.append(f"  {reason}: {count}")

    report_path = group_output_root / "summary.txt"
    write_text_report(report_path, report_lines)
    written_paths.append(report_path)

    return written_paths, report_path


def main():
    args = parse_args()

    if args.dim_per_field < 2:
        raise ValueError("--dim-per-field must be >= 2")
    if args.rff_scale <= 0:
        raise ValueError("--rff-scale must be > 0")
    if args.pair_block_size <= 0:
        raise ValueError("--pair-block-size must be > 0")
    if args.num_bins <= 1:
        raise ValueError("--num-bins must be > 1")

    coord_names = list(dict.fromkeys(args.coords))
    rng = np.random.default_rng(args.seed)

    events = load_events(
        config_path=args.config,
        event_index=args.event_index,
        num_events=args.num_events,
        num_workers=args.num_workers,
    )

    output_root = args.out_dir / args.out_prefix / "unscaled"
    output_root.mkdir(parents=True, exist_ok=True)

    rff_base_matrices = init_rff_base_matrices(coord_names=coord_names, dim_per_field=args.dim_per_field, seed=args.seed)

    group_summary_rows = []
    cross_summary_rows = []

    for group_spec in GROUP_SPECS:
        event_data, skipped_events, skipped_reasons, total_hits = build_group_event_data(
            events=events,
            group_subsystems=group_spec["subsystems"],
        )

        if not event_data:
            print(f"[{group_spec['name']}] Skipping group: no usable events")
            group_summary_rows.append({
                "group": group_spec["name"],
                "subsystems": "+".join(group_spec["subsystems"]),
                "used_events": 0,
                "skipped_events": len(events),
                "total_hits": 0,
                "total_pairs_used": 0,
                "total_pairs_available": 0,
                "report": "(no output)",
            })
            continue

        pair_plan, total_pairs_used, total_pairs_available, sampled_events = build_pair_sampling_plan(
            event_data=event_data,
            max_pairs=args.max_pairs,
            rng=rng,
            group_name=group_spec["name"],
        )

        used_events = len(event_data)
        written_paths, report_path = run_group_plots(
            group_spec=group_spec,
            event_data=event_data,
            pair_plan=pair_plan,
            coord_names=coord_names,
            rff_base_matrices=rff_base_matrices,
            args=args,
            output_root=output_root,
            used_events=used_events,
            skipped_events=skipped_events,
            total_hits=total_hits,
            skipped_reasons=skipped_reasons,
            total_pairs_used=total_pairs_used,
            total_pairs_available=total_pairs_available,
            sampled_events=sampled_events,
        )

        for path in written_paths:
            print(f"[{group_spec['name']}] Wrote {path}")

        group_summary_rows.append({
            "group": group_spec["name"],
            "subsystems": "+".join(group_spec["subsystems"]),
            "used_events": used_events,
            "skipped_events": skipped_events,
            "total_hits": total_hits,
            "total_pairs_used": total_pairs_used,
            "total_pairs_available": total_pairs_available,
            "report": str(report_path),
        })

    pair_group_specs = [spec for spec in GROUP_SPECS if spec["category"] == "combined" and len(spec["subsystems"]) == 2]
    for group_spec in pair_group_specs:
        source_subsystem = group_spec["subsystems"][1]
        target_subsystem = group_spec["subsystems"][0]

        event_data, skipped_events, skipped_reasons, total_source_hits, total_target_hits = build_cross_group_event_data(
            events=events,
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
        )

        if not event_data:
            print(f"[cross:{group_spec['name']}] Skipping group: no usable events")
            cross_summary_rows.append({
                "group": group_spec["name"],
                "direction": f"{source_subsystem}->{target_subsystem}",
                "used_events": 0,
                "skipped_events": len(events),
                "total_source_hits": 0,
                "total_target_hits": 0,
                "total_pairs_used": 0,
                "total_pairs_available": 0,
                "report": "(no output)",
            })
            continue

        pair_plan, total_pairs_used, total_pairs_available, sampled_events = build_cross_pair_sampling_plan(
            event_data=event_data,
            max_pairs=args.max_pairs,
            rng=rng,
            group_name=group_spec["name"],
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
        )

        used_events = len(event_data)
        written_paths, report_path = run_cross_group_plots(
            group_spec=group_spec,
            event_data=event_data,
            pair_plan=pair_plan,
            coord_names=coord_names,
            rff_base_matrices=rff_base_matrices,
            args=args,
            output_root=output_root,
            source_subsystem=source_subsystem,
            target_subsystem=target_subsystem,
            used_events=used_events,
            skipped_events=skipped_events,
            total_source_hits=total_source_hits,
            total_target_hits=total_target_hits,
            skipped_reasons=skipped_reasons,
            total_pairs_used=total_pairs_used,
            total_pairs_available=total_pairs_available,
            sampled_events=sampled_events,
        )

        for path in written_paths:
            print(f"[cross:{group_spec['name']}] Wrote {path}")

        cross_summary_rows.append({
            "group": group_spec["name"],
            "direction": f"{source_subsystem}->{target_subsystem}",
            "used_events": used_events,
            "skipped_events": skipped_events,
            "total_source_hits": total_source_hits,
            "total_target_hits": total_target_hits,
            "total_pairs_used": total_pairs_used,
            "total_pairs_available": total_pairs_available,
            "report": str(report_path),
        })

    summary_lines = [
        "CLD RFF dot-magnitude plot run summary",
        f"Config: {args.config}",
        f"Event index: {args.event_index}",
        f"Requested events: {args.num_events}",
        f"Coordinates: {', '.join(coord_names)}",
        "",
        "Per-group outputs:",
    ]
    for row in group_summary_rows:
        summary_lines.extend([
            (
                f"  - {row['group']} [{row['subsystems']}]: used/skipped events "
                f"{row['used_events']}/{row['skipped_events']}, "
                f"hits={row['total_hits']:,}, pairs={row['total_pairs_used']:,}/{row['total_pairs_available']:,}"
            ),
            f"    report: {row['report']}",
        ])

    summary_lines.extend([
        "",
        "Cross-type ordered outputs (source->target only):",
    ])
    for row in cross_summary_rows:
        summary_lines.extend([
            (
                f"  - {row['group']} [{row['direction']}]: used/skipped events "
                f"{row['used_events']}/{row['skipped_events']}, "
                f"source_hits={row['total_source_hits']:,}, target_hits={row['total_target_hits']:,}, "
                f"pairs={row['total_pairs_used']:,}/{row['total_pairs_available']:,}"
            ),
            f"    report: {row['report']}",
        ])

    top_summary_path = output_root / "summary.txt"
    write_text_report(top_summary_path, summary_lines)
    print(f"Wrote {top_summary_path}")


if __name__ == "__main__":
    main()
