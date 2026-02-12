import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hepattn.experiments.trackml.data import TrackMLDataset

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
DEFAULT_SCALE_TABLE = Path("src/hepattn/experiments/trackml/eval/plots/hit_posenc_and_rff_scale/original/tables/optimal_scales.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "TrackML hit-pair diagnostics: for each coordinate, build a 2D histogram "
            "over (coord_i, coord_j) with color set by |normalized RFF dot product|. "
            "Writes both original-coordinate and scaled-coordinate runs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/hepattn/experiments/trackml/configs/tracking.yaml"),
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--event-index", type=int, default=0, help="Zero-based start index in the selected split.")
    parser.add_argument("--num-events", type=int, default=1, help="Number of events to aggregate.")
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
        help=("Max hit pairs per event. <= 0 uses all pairs. Useful when events are large and all-pairs is too expensive."),
    )
    parser.add_argument(
        "--pair-block-size",
        type=int,
        default=1024,
        help="Block size for exact all-pairs accumulation.",
    )
    parser.add_argument("--num-bins-diff", type=int, default=120, help="Histogram bins per coordinate axis.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src/hepattn/experiments/trackml/eval/plots"),
    )
    parser.add_argument("--out-prefix", type=str, default="hit_coord_pair_vs_rff_dotmag")
    parser.add_argument(
        "--coord-scale-table",
        type=Path,
        default=DEFAULT_SCALE_TABLE,
        help=("Path to the RFF scale-scan table containing per-coordinate best_scale values. Scaled run uses coord_scaled = coord / best_scale."),
    )
    return parser.parse_args()


def load_dataset(config_path, split, min_num_events):
    cfg = yaml.safe_load(config_path.read_text())["data"]
    dir_key = f"{split}_dir"
    hit_eval_key = f"hit_eval_{split}"
    num_events_key = f"num_{split}"

    if dir_key not in cfg:
        raise KeyError(f"Missing data.{dir_key} in {config_path}")

    config_num_events = int(cfg.get(num_events_key, -1))
    dataset_num_events = -1 if config_num_events < 0 else max(config_num_events, int(min_num_events))

    dataset_kwargs = {
        "dirpath": cfg[dir_key],
        "inputs": cfg["inputs"],
        "targets": cfg["targets"],
        "num_events": dataset_num_events,
        "hit_eval_path": cfg.get(hit_eval_key),
        "hit_volume_ids": cfg.get("hit_volume_ids"),
        "feature_volume_ids": cfg.get("feature_volume_ids"),
        "particle_min_pt": cfg.get("particle_min_pt", 1.0),
        "particle_max_abs_eta": cfg.get("particle_max_abs_eta", 2.5),
        "particle_min_num_hits": cfg.get("particle_min_num_hits", 3),
        "event_max_num_particles": cfg.get("event_max_num_particles", 1000),
        "strict_max_objects": cfg.get("strict_max_objects", False),
        "dummy_data": cfg.get("dummy_data", False),
    }
    return TrackMLDataset(**dataset_kwargs)


def load_events(dataset, event_index, num_events):
    if event_index < 0:
        raise ValueError("--event-index must be >= 0")
    if num_events <= 0:
        raise ValueError("--num-events must be > 0")

    end_index = event_index + num_events
    if end_index > len(dataset):
        raise RuntimeError(f"Requested events [{event_index}, {end_index}), but dataset has {len(dataset)} events.")

    return [dataset[idx] for idx in range(event_index, end_index)]


def extract_hit_coords(inputs):
    if "hit_valid" not in inputs:
        return None, "missing hit_valid"

    required_coord_fields = ["x", "y", "z", "r", "eta", "phi"]
    missing_fields = [field for field in required_coord_fields if f"hit_{field}" not in inputs]
    if missing_fields:
        return None, f"missing hit fields: {', '.join(missing_fields)}"

    hit_valid = inputs["hit_valid"][0].bool()
    num_hits = int(hit_valid.sum().item())
    if num_hits < 2:
        return None, "fewer than 2 valid hits"

    phi = inputs["hit_phi"][0][hit_valid].float()
    coords = {
        "x": inputs["hit_x"][0][hit_valid].float(),
        "y": inputs["hit_y"][0][hit_valid].float(),
        "z": inputs["hit_z"][0][hit_valid].float(),
        "r": inputs["hit_r"][0][hit_valid].float(),
        "eta": inputs["hit_eta"][0][hit_valid].float(),
        "phi": phi,
        "sinphi": torch.sin(phi),
        "cosphi": torch.cos(phi),
    }
    return {"num_hits": num_hits, "coords": coords}, None


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


def build_coord_edges(event_data, coord_name, num_bins, force_data_range=False):
    if not force_data_range:
        if coord_name == "phi":
            return np.linspace(-np.pi, np.pi, num_bins + 1)
        if coord_name in {"sinphi", "cosphi"}:
            return np.linspace(-1.0, 1.0, num_bins + 1)

    min_coord = np.inf
    max_coord = -np.inf
    for payload in event_data:
        values = payload["coords"][coord_name]
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


def accumulate_hist_sampled_pairs(values, embeddings, pairs, coord_edges):
    dotmag_sum = np.zeros((len(coord_edges) - 1, len(coord_edges) - 1), dtype=np.float64)
    pair_counts = np.zeros((len(coord_edges) - 1, len(coord_edges) - 1), dtype=np.int64)
    if pairs.numel() == 0:
        return dotmag_sum, pair_counts

    x_forward = values[pairs[:, 0]]
    y_forward = values[pairs[:, 1]]
    dot_magnitude = pair_dot_products(embeddings, pairs).abs()

    # Include both directions for every pair: (i, j) and (j, i).
    left_coords = torch.cat([x_forward, y_forward], dim=0)
    right_coords = torch.cat([y_forward, x_forward], dim=0)
    dot_magnitude = torch.cat([dot_magnitude, dot_magnitude], dim=0)

    left_np = left_coords.detach().cpu().numpy()
    right_np = right_coords.detach().cpu().numpy()
    dotmag_np = dot_magnitude.detach().cpu().numpy()

    counts_hist, _, _ = np.histogram2d(
        left_np,
        right_np,
        bins=(coord_edges, coord_edges),
    )
    sum_hist, _, _ = np.histogram2d(
        left_np,
        right_np,
        bins=(coord_edges, coord_edges),
        weights=dotmag_np,
    )
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

            # Include both directions for every pair: (i, j) and (j, i).
            x_coords = torch.cat([x_upper, y_upper], dim=0)
            y_coords = torch.cat([y_upper, x_upper], dim=0)
            dots_flat = torch.cat([dots_upper, dots_upper], dim=0)

            if dots_flat.numel() == 0:
                continue

            x_np = x_coords.detach().cpu().numpy()
            y_np = y_coords.detach().cpu().numpy()
            dots_np = dots_flat.detach().cpu().numpy()

            counts_hist, _, _ = np.histogram2d(
                x_np,
                y_np,
                bins=(coord_edges, coord_edges),
            )
            sum_hist, _, _ = np.histogram2d(
                x_np,
                y_np,
                bins=(coord_edges, coord_edges),
                weights=dots_np,
            )
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
    run_label,
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
        (f"TrackML hit pairs [{run_label}] | events={used_events} | pairs used={total_pairs_used:,} / available={total_pairs_available:,}"),
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
    run_label,
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
            f"TrackML hit-pair coordinates with color=mean |RFF dot| [{run_label}] "
            f"| events={used_events} | pairs used={total_pairs_used:,} / available={total_pairs_available:,}"
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


def load_coord_scales_from_table(scale_table_path, coord_names):
    if not scale_table_path.exists():
        raise FileNotFoundError(f"Scale table not found: {scale_table_path}")

    coord_set = set(coord_names)
    scales = {}
    for line in scale_table_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        coord_name = parts[0]
        if coord_name not in coord_set:
            continue

        try:
            scale_value = float(parts[2])
        except ValueError:
            continue

        if np.isfinite(scale_value) and scale_value > 0:
            scales[coord_name] = float(scale_value)

    missing = [coord_name for coord_name in coord_names if coord_name not in scales]
    if missing:
        raise RuntimeError(f"Failed to read positive scales for coordinates from {scale_table_path}: missing {', '.join(missing)}")
    return scales


def apply_coord_scaling_to_event_data(event_data, coord_scales):
    scaled_event_data = []
    for payload in event_data:
        scaled_coords = {}
        for coord_name, values in payload["coords"].items():
            scale_value = float(coord_scales.get(coord_name, 1.0))
            if (not np.isfinite(scale_value)) or (scale_value <= 0.0):
                scale_value = 1.0
            scaled_coords[coord_name] = values / scale_value

        scaled_event_data.append({"num_hits": payload["num_hits"], "coords": scaled_coords})
    return scaled_event_data


def build_pair_sampling_plan(event_data, max_pairs, rng):
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
                sampled_pairs, _ = sample_pairs_without_replacement(
                    num_hits=num_hits,
                    max_pairs=max_pairs,
                    rng=rng,
                )
                used_pairs = int(2 * sampled_pairs.shape[0])
                sampled_events += 1

            pair_mode = "sampled" if sampled_pairs is not None else "all"
            print(f"Event {event_idx}: hits={num_hits:,}, pairs used={used_pairs:,} / available={available_pairs:,} ({pair_mode})")

        pair_plan.append({
            "sampled_pairs": sampled_pairs,
            "available_pairs": available_pairs,
            "used_pairs": used_pairs,
        })
        total_pairs_available += available_pairs
        total_pairs_used += used_pairs

    return pair_plan, total_pairs_used, total_pairs_available, sampled_events


def run_plot_pass(
    *,
    event_data,
    coord_names,
    pair_plan,
    rff_base_matrices,
    args,
    output_root,
    run_label,
    used_events,
    skipped_events,
    total_hits,
    skipped_reasons,
    total_pairs_used,
    total_pairs_available,
    sampled_events,
    coord_scales=None,
    scale_table_path=None,
):
    coord_edges_by_coord = {
        coord_name: build_coord_edges(
            event_data=event_data,
            coord_name=coord_name,
            num_bins=args.num_bins_diff,
            force_data_range=(coord_scales is not None),
        )
        for coord_name in coord_names
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

    written_paths = []
    for coord_name in coord_names:
        out_path = output_root / "per_coord" / f"{coord_name}.png"
        plot_coord_histogram(
            coord_name=coord_name,
            dotmag_sum=dotmag_sum_by_coord[coord_name],
            pair_counts=pair_counts_by_coord[coord_name],
            coord_edges=coord_edges_by_coord[coord_name],
            out_path=out_path,
            run_label=run_label,
            used_events=used_events,
            total_pairs_used=total_pairs_used,
            total_pairs_available=total_pairs_available,
        )
        written_paths.append(out_path)

    grid_out_path = output_root / "all_coords.png"
    plot_coord_histogram_grid(
        coord_names=coord_names,
        dotmag_sum_by_coord=dotmag_sum_by_coord,
        pair_counts_by_coord=pair_counts_by_coord,
        coord_edges_by_coord=coord_edges_by_coord,
        out_path=grid_out_path,
        run_label=run_label,
        used_events=used_events,
        total_pairs_used=total_pairs_used,
        total_pairs_available=total_pairs_available,
    )
    written_paths.append(grid_out_path)

    report_lines = [
        "TrackML coord_i vs coord_j with color=mean |RFF dot| summary",
        f"Run label: {run_label}",
        f"Split: {args.split}",
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
        f"Coordinate bins per axis: {args.num_bins_diff}",
        "Color metric: mean absolute normalized RFF dot product per bin",
    ]

    if coord_scales is not None:
        report_lines.extend(("", "Coordinate scaling:"))
        if scale_table_path is not None:
            report_lines.append(f"Scale table: {scale_table_path}")
        report_lines.append("coord_scaled = coord / scale")
        report_lines.extend(f"  {coord_name}: {coord_scales[coord_name]:.6g}" for coord_name in coord_names)

    if skipped_reasons:
        report_lines.append("Skipped event reasons:")
        for reason, count in sorted(skipped_reasons.items()):
            report_lines.append(f"  {reason}: {count}")

    report_path = output_root / "summary.txt"
    write_text_report(report_path, report_lines)
    written_paths.append(report_path)
    return written_paths


def main():
    args = parse_args()

    if args.dim_per_field < 2:
        raise ValueError("--dim-per-field must be >= 2")
    if args.rff_scale <= 0:
        raise ValueError("--rff-scale must be > 0")
    if args.pair_block_size <= 0:
        raise ValueError("--pair-block-size must be > 0")
    if args.num_bins_diff <= 1:
        raise ValueError("--num-bins-diff must be > 1")

    coord_names = list(dict.fromkeys(args.coords))

    dataset = load_dataset(args.config, args.split, min_num_events=args.event_index + args.num_events)
    events = load_events(dataset, args.event_index, args.num_events)

    event_data = []
    skipped_events = 0
    skipped_reasons = {}
    total_hits = 0
    for inputs, _targets in events:
        payload, reason = extract_hit_coords(inputs=inputs)
        if payload is None:
            skipped_events += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        event_data.append(payload)
        total_hits += payload["num_hits"]

    if not event_data:
        raise RuntimeError("No usable events found. Try increasing --num-events or relaxing data cuts.")

    used_events = len(event_data)
    rng = np.random.default_rng(args.seed)
    pair_plan, total_pairs_used, total_pairs_available, sampled_events = build_pair_sampling_plan(
        event_data=event_data,
        max_pairs=args.max_pairs,
        rng=rng,
    )

    rff_base_matrices = init_rff_base_matrices(
        coord_names=coord_names,
        dim_per_field=args.dim_per_field,
        seed=args.seed,
    )

    output_root = args.out_dir / args.out_prefix
    output_root.mkdir(parents=True, exist_ok=True)

    original_written_paths = run_plot_pass(
        event_data=event_data,
        coord_names=coord_names,
        pair_plan=pair_plan,
        rff_base_matrices=rff_base_matrices,
        args=args,
        output_root=output_root,
        run_label="original",
        used_events=used_events,
        skipped_events=skipped_events,
        total_hits=total_hits,
        skipped_reasons=skipped_reasons,
        total_pairs_used=total_pairs_used,
        total_pairs_available=total_pairs_available,
        sampled_events=sampled_events,
        coord_scales=None,
        scale_table_path=None,
    )

    coord_scales = load_coord_scales_from_table(
        scale_table_path=args.coord_scale_table,
        coord_names=coord_names,
    )
    scaled_event_data = apply_coord_scaling_to_event_data(
        event_data=event_data,
        coord_scales=coord_scales,
    )
    scaled_output_root = output_root / "scaled"
    scaled_written_paths = run_plot_pass(
        event_data=scaled_event_data,
        coord_names=coord_names,
        pair_plan=pair_plan,
        rff_base_matrices=rff_base_matrices,
        args=args,
        output_root=scaled_output_root,
        run_label="scaled",
        used_events=used_events,
        skipped_events=skipped_events,
        total_hits=total_hits,
        skipped_reasons=skipped_reasons,
        total_pairs_used=total_pairs_used,
        total_pairs_available=total_pairs_available,
        sampled_events=sampled_events,
        coord_scales=coord_scales,
        scale_table_path=args.coord_scale_table,
    )

    for path in original_written_paths:
        print(f"[original] Wrote {path}")
    for path in scaled_written_paths:
        print(f"[scaled] Wrote {path}")

    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
