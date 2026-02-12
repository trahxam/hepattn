#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance

from hepattn.experiments.trackml.data import TrackMLDataset
from hepattn.models.posenc import pos_enc

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
DEFAULT_DIM_VALUES = [2**k for k in range(1, 10)]  # 2..512

PAIR_FIELD_SETS = {
    "retaphi": {
        "fields": {
            "r": "r",
            "eta": "eta",
            "phi": "phi",
        },
        "title_suffix": "(r, eta, phi)",
    },
    "xyz": {
        "fields": {
            "x": "x",
            "y": "y",
            "z": "z",
        },
        "title_suffix": "(x, y, z)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "TrackML hit-only positional-encoding diagnostics: "
            "(1) pair-dot distributions, (2) RFF-scale sweep, and "
            "(3) scaled-coordinate RFF-dimension sweep."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/hepattn/experiments/trackml/configs/tracking.yaml"),
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--event-index", type=int, default=0, help="Zero-based start index in the selected split.")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events to aggregate.")
    parser.add_argument("--dim-per-field", type=int, default=32, help="Embedding width per field.")
    parser.add_argument("--alpha", type=float, default=1000.0, help="Sinusoidal PE alpha.")
    parser.add_argument("--base", type=float, default=100.0, help="Sinusoidal PE log base.")
    parser.add_argument("--rff-scale", type=float, default=1.0, help="Default RFF scale for pair-dot plots.")
    parser.add_argument(
        "--curve-num-scales",
        type=int,
        default=32,
        help="Number of log-spaced scales for RFF sweep curves.",
    )
    parser.add_argument(
        "--curve-scale-min-mult",
        type=float,
        default=0.1,
        help="Minimum multiplier relative to --rff-scale for sweep curves.",
    )
    parser.add_argument(
        "--curve-scale-max-mult",
        type=float,
        default=100.0,
        help="Maximum multiplier relative to --rff-scale for sweep curves.",
    )
    parser.add_argument(
        "--dim-values",
        type=int,
        nargs="+",
        default=DEFAULT_DIM_VALUES,
        help="Embedding dimensions to scan on scaled coordinates (e.g. 2 4 8 ... 512).",
    )
    parser.add_argument(
        "--max-same-pairs",
        type=int,
        default=25000,
        help="Subsample same-particle pairs to this cap per event (<=0 uses all).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src/hepattn/experiments/trackml/eval/plots"),
    )
    parser.add_argument("--out-prefix", type=str, default="hit_posenc_and_rff_scale")
    parser.add_argument("--pdf-path", type=Path, default=None, help="Optional custom summary PDF path.")
    parser.add_argument("--skip-pdf", action="store_true", help="If set, skip writing summary PDF.")
    return parser.parse_args()


def load_dataset(config_path, split, min_num_events):
    cfg = yaml.safe_load(config_path.read_text())["data"]
    dir_key = f"{split}_dir"
    hit_eval_key = f"hit_eval_{split}"
    num_events_key = f"num_{split}"

    if dir_key not in cfg:
        raise KeyError(f"Missing data.{dir_key} in {config_path}")

    config_num_events = int(cfg.get(num_events_key, -1))
    if config_num_events < 0:
        dataset_num_events = -1
    else:
        dataset_num_events = max(config_num_events, int(min_num_events))

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
        raise RuntimeError(
            f"Requested events [{event_index}, {end_index}), but dataset has {len(dataset)} events."
        )

    return [dataset[idx] for idx in range(event_index, end_index)]


def collect_same_particle_pairs(particle_hit_valid, particle_valid):
    pairs = []
    for hit_mask in particle_hit_valid[particle_valid]:
        hit_indices = torch.nonzero(hit_mask, as_tuple=False).flatten()
        if hit_indices.numel() >= 2:
            pairs.append(torch.combinations(hit_indices, r=2))

    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)

    pairs_tensor = torch.cat(pairs, dim=0)
    pairs_tensor = torch.sort(pairs_tensor, dim=1).values
    return torch.unique(pairs_tensor, dim=0)


def subsample_pairs(pairs, max_pairs, rng):
    if max_pairs <= 0 or pairs.shape[0] <= max_pairs:
        return pairs

    keep = rng.choice(pairs.shape[0], size=max_pairs, replace=False)
    keep_idx = torch.as_tensor(keep, dtype=torch.long)
    return pairs[keep_idx]


def sample_not_same_pairs(hit_to_particle, num_pairs, rng):
    num_hits = int(hit_to_particle.shape[0])
    if num_hits < 2 or num_pairs <= 0:
        return torch.empty((0, 2), dtype=torch.long)

    target = min(int(num_pairs), int(num_hits * (num_hits - 1) // 2))
    pair_set = set()
    attempt_limit = max(200, 20 * target)
    attempts = 0

    while len(pair_set) < target and attempts < attempt_limit:
        batch_size = max(1024, 4 * (target - len(pair_set)))
        a = rng.integers(0, num_hits, size=batch_size)
        b = rng.integers(0, num_hits, size=batch_size)
        valid = a != b

        if not np.any(valid):
            attempts += batch_size
            continue

        a = a[valid]
        b = b[valid]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        candidates = np.stack((lo, hi), axis=1)
        candidates = np.unique(candidates, axis=0)
        pair_tensor = torch.as_tensor(candidates, dtype=torch.long)

        if pair_tensor.numel() == 0:
            attempts += batch_size
            continue

        share_particle = (hit_to_particle[pair_tensor[:, 0]] & hit_to_particle[pair_tensor[:, 1]]).any(dim=1)
        diff_pairs = pair_tensor[~share_particle]

        for i, j in diff_pairs.tolist():
            pair_set.add((int(i), int(j)))
            if len(pair_set) >= target:
                break

        attempts += batch_size

    if not pair_set:
        return torch.empty((0, 2), dtype=torch.long)

    return torch.tensor(list(pair_set), dtype=torch.long)


def extract_hit_view(inputs, targets):
    if "hit_valid" not in inputs:
        return None, "missing hit_valid"
    if "particle_hit_valid" not in targets:
        return None, "missing particle_hit_valid"

    hit_valid = inputs["hit_valid"][0].bool()
    num_hits = int(hit_valid.sum().item())
    if num_hits < 1:
        return None, "fewer than 1 valid hit"

    required_coord_fields = ["x", "y", "z", "r", "eta", "phi"]
    missing_fields = [field for field in required_coord_fields if f"hit_{field}" not in inputs]
    if missing_fields:
        return None, f"missing hit fields: {', '.join(missing_fields)}"

    particle_hit_valid = targets["particle_hit_valid"][0][:, hit_valid].bool()
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
    return {
        "num_hits": num_hits,
        "coords": coords,
        "particle_hit_valid": particle_hit_valid,
    }, None


def prepare_event_data(inputs, targets, max_same_pairs, rng):
    hit_view, reason = extract_hit_view(inputs=inputs, targets=targets)
    if hit_view is None:
        return None, reason

    if "particle_valid" not in targets:
        return None, "missing particle_valid"

    if hit_view["num_hits"] < 2:
        return None, "fewer than 2 valid hits"

    particle_valid = targets["particle_valid"][0].bool()
    same_pairs = collect_same_particle_pairs(hit_view["particle_hit_valid"], particle_valid)
    same_pairs = subsample_pairs(same_pairs, max_same_pairs, rng)
    if same_pairs.numel() == 0:
        return None, "no same-particle hit pairs"

    hit_to_particle = hit_view["particle_hit_valid"][particle_valid].T.contiguous()
    not_same_pairs = sample_not_same_pairs(
        hit_to_particle=hit_to_particle,
        num_pairs=same_pairs.shape[0],
        rng=rng,
    )
    if not_same_pairs.shape[0] == 0:
        return None, "failed to sample non-same-particle hit pairs"

    return {
        "num_hits": hit_view["num_hits"],
        "coords": hit_view["coords"],
        "same_pairs": same_pairs,
        "not_same_pairs": not_same_pairs,
    }, None


def init_rff_base_matrices(coord_names, dim_per_field, seed):
    half_dim = dim_per_field // 2
    if half_dim == 0:
        raise ValueError("--dim-per-field must be >= 2 for random Fourier features")

    gen = torch.Generator()
    gen.manual_seed(seed)
    return {coord_name: torch.randn((1, half_dim), generator=gen) for coord_name in sorted(coord_names)}


def init_rff_base_matrices_max_half(coord_names, max_half_dim, seed):
    if max_half_dim <= 0:
        raise ValueError("Maximum half-dimension must be > 0")

    gen = torch.Generator()
    gen.manual_seed(seed)
    return {coord_name: torch.randn((1, max_half_dim), generator=gen) for coord_name in sorted(coord_names)}


def random_fourier_pos_enc(xs, base_matrix, scale, dim_per_field):
    matrix = ((1.0 / scale) * base_matrix).to(device=xs.device, dtype=xs.dtype)
    proj = (2.0 * torch.pi * xs.unsqueeze(-1)) * matrix
    enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    if enc.shape[-1] < dim_per_field:
        pad = torch.zeros((*enc.shape[:-1], dim_per_field - enc.shape[-1]), dtype=enc.dtype, device=enc.device)
        enc = torch.cat([enc, pad], dim=-1)
    return enc


def random_fourier_pos_enc_trig(xs, base_matrix, scale, max_half_dim):
    if max_half_dim <= 0:
        raise ValueError(f"Maximum half-dimension must be > 0, got {max_half_dim}")

    matrix = ((1.0 / scale) * base_matrix[:, :max_half_dim]).to(device=xs.device, dtype=xs.dtype)
    proj = (2.0 * torch.pi * xs.unsqueeze(-1)) * matrix
    return torch.sin(proj), torch.cos(proj)


def random_fourier_pos_enc_multi(xs, base_matrix, scales, dim_per_field):
    scales_tensor = torch.as_tensor(scales, device=xs.device, dtype=xs.dtype)
    matrix = ((1.0 / scales_tensor)[:, None, None] * base_matrix.to(device=xs.device, dtype=xs.dtype)[None, :, :])
    proj = (2.0 * torch.pi * xs)[None, :, None] * matrix
    enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    if enc.shape[-1] < dim_per_field:
        pad = torch.zeros((*enc.shape[:-1], dim_per_field - enc.shape[-1]), dtype=enc.dtype, device=enc.device)
        enc = torch.cat([enc, pad], dim=-1)
    return enc


def encode_field_set(coords, field_map, encoding_name, dim_per_field, alpha, base, rff_base_matrices, rff_scale):
    encodings = {}
    per_field = []

    for field_name, coord_name in field_map.items():
        values = coords[coord_name]
        if encoding_name == "sinusoidal":
            field_encoding = pos_enc(values, dim=dim_per_field, alpha=alpha, base=base)
        else:
            field_encoding = random_fourier_pos_enc(
                values,
                base_matrix=rff_base_matrices[coord_name],
                scale=rff_scale,
                dim_per_field=dim_per_field,
            )
        encodings[field_name] = field_encoding
        per_field.append(field_encoding)

    encodings["combined"] = torch.cat(per_field, dim=-1)
    return encodings


def pair_dot_products(embeddings_left, pairs):
    if pairs.numel() == 0:
        return torch.empty((0,), dtype=embeddings_left.dtype)

    left = embeddings_left[pairs[:, 0]]
    right = embeddings_left[pairs[:, 1]]
    dot = (left * right).sum(dim=-1)
    emb_dim = int(embeddings_left.shape[-1])
    if emb_dim <= 0:
        return dot
    return dot / float(emb_dim)


def pair_dot_products_multi(embeddings_left, pairs):
    num_scales = embeddings_left.shape[0]
    if pairs.numel() == 0:
        return torch.empty((num_scales, 0), dtype=embeddings_left.dtype, device=embeddings_left.device)

    pairs = pairs.to(device=embeddings_left.device)
    left = embeddings_left[:, pairs[:, 0], :]
    right = embeddings_left[:, pairs[:, 1], :]
    dot = (left * right).sum(dim=-1)
    emb_dim = int(embeddings_left.shape[-1])
    if emb_dim <= 0:
        return dot
    return dot / float(emb_dim)


def pair_dot_products_multi_from_trig(sin_vals, cos_vals, pairs, half_dim_indices, dim_values):
    num_dims = int(half_dim_indices.shape[0])
    if pairs.numel() == 0:
        return torch.empty((num_dims, 0), dtype=sin_vals.dtype, device=sin_vals.device)

    pairs = pairs.to(device=sin_vals.device)
    dim_idx = half_dim_indices.to(device=sin_vals.device)

    left_sin = sin_vals[pairs[:, 0]]
    right_sin = sin_vals[pairs[:, 1]]
    left_cos = cos_vals[pairs[:, 0]]
    right_cos = cos_vals[pairs[:, 1]]

    per_frequency_dot = (left_sin * right_sin) + (left_cos * right_cos)
    cumulative_dot = torch.cumsum(per_frequency_dot, dim=-1)
    selected = cumulative_dot[:, dim_idx].T.contiguous()

    # Normalize by requested embedding dimension so curves are comparable across dims.
    dim_values_t = torch.as_tensor(dim_values, dtype=selected.dtype, device=selected.device)
    selected = selected / dim_values_t[:, None]
    return selected


def make_bins(a, b, num_bins=60):
    if a.size == 0 and b.size == 0:
        return np.linspace(-1.0, 1.0, num_bins)

    min_val = np.inf
    max_val = -np.inf
    if a.size:
        min_val = min(min_val, float(np.min(a)))
        max_val = max(max_val, float(np.max(a)))
    if b.size:
        min_val = min(min_val, float(np.min(b)))
        max_val = max(max_val, float(np.max(b)))

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        min_val, max_val = -1.0, 1.0

    if min_val == max_val:
        span = max(abs(min_val), 1e-3)
        min_val -= 0.05 * span
        max_val += 0.05 * span
    else:
        pad = 0.02 * (max_val - min_val)
        min_val -= pad
        max_val += pad

    return np.linspace(min_val, max_val, num_bins)


def plot_dot_products(scores_same, scores_random, out_path, field_order, title_prefix, num_events, hit_summary):
    field_titles = {
        "combined": "Combined PE",
        "r": "r PE",
        "eta": "eta PE",
        "phi": "phi PE",
        "x": "x PE",
        "y": "y PE",
        "z": "z PE",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_fields = ["combined"] + list(field_order)

    for ax, field in zip(axes.flat, plot_fields):
        same = scores_same[field].detach().cpu().numpy()
        random = scores_random[field].detach().cpu().numpy()
        bins = make_bins(same, random)

        wdist = float("nan")
        if same.size and random.size:
            wdist = float(wasserstein_distance(same, random))

        if same.size:
            ax.hist(
                same,
                bins=bins,
                histtype="step",
                linewidth=1.6,
                color="tab:blue",
                density=True,
                label=f"same particle ({same.size:,})",
            )
        if random.size:
            ax.hist(
                random,
                bins=bins,
                histtype="step",
                linewidth=1.6,
                color="tab:orange",
                density=True,
                label=f"not same particle ({random.size:,})",
            )

        if np.isfinite(wdist):
            ax.set_title(f"{field_titles[field]} | W={wdist:.3g}")
        else:
            ax.set_title(f"{field_titles[field]} | W=n/a")
        ax.set_xlabel("Normalized dot product (dot/dim)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3, linestyle="--")

    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{title_prefix} | events={num_events} | valid hits={hit_summary} | dot/dim")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_wasserstein_vs_scale_all(curve_results, scale_values, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for coord_name in BASE_COORD_NAMES:
        payload = curve_results.get(coord_name)
        if payload is None:
            continue

        wdist = payload["wdist"]
        finite = np.isfinite(wdist)
        if not np.any(finite):
            continue

        max_idx = int(np.nanargmax(wdist))
        best_scale = float(scale_values[max_idx])
        label = f"{BASE_COORD_LABELS[coord_name]} (max@{best_scale:.3g})"
        ax.plot(scale_values, wdist, linewidth=1.5, marker="o", markersize=3.5, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("RFF scale")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title("TrackML hit-only: Wasserstein vs RFF scale (dot/dim)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_wasserstein_vs_scale_single(scale_values, wdist, coord_name, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    finite = np.isfinite(wdist)
    if np.any(finite):
        max_idx = int(np.nanargmax(wdist))
        best_scale = float(scale_values[max_idx])
        best_wdist = float(wdist[max_idx])

        ax.plot(scale_values, wdist, linewidth=1.8, marker="o", markersize=4.0, color="tab:blue")
        ax.axvline(best_scale, linestyle=":", linewidth=1.2, color="tab:red")
        title_suffix = f"max={best_wdist:.4g} @ scale={best_scale:.4g}"
    else:
        title_suffix = "no finite values"
        ax.text(0.5, 0.5, "No finite Wasserstein values", transform=ax.transAxes, ha="center", va="center")

    ax.set_xscale("log")
    ax.set_xlabel("RFF scale")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(f"{BASE_COORD_LABELS[coord_name]} (dot/dim): {title_suffix}")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_wasserstein_vs_dim_all(curve_results, dim_values, out_path, run_label):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for coord_name in BASE_COORD_NAMES:
        payload = curve_results.get(coord_name)
        if payload is None:
            continue

        wdist = payload["wdist"]
        finite = np.isfinite(wdist)
        if not np.any(finite):
            continue

        max_idx = int(np.nanargmax(wdist))
        best_dim = int(dim_values[max_idx])
        label = f"{BASE_COORD_LABELS[coord_name]} (max@{best_dim})"
        ax.plot(dim_values, wdist, linewidth=1.5, marker="o", markersize=3.5, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(dim_values)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("RFF embedding dimension")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(f"TrackML hit-only ({run_label}): Wasserstein vs RFF embedding dimension")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_wasserstein_vs_dim_single(dim_values, wdist, coord_name, out_path, run_label):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    finite = np.isfinite(wdist)
    if np.any(finite):
        max_idx = int(np.nanargmax(wdist))
        best_dim = int(dim_values[max_idx])
        best_wdist = float(wdist[max_idx])

        ax.plot(dim_values, wdist, linewidth=1.8, marker="o", markersize=4.0, color="tab:blue")
        ax.axvline(best_dim, linestyle=":", linewidth=1.2, color="tab:red")
        title_suffix = f"max={best_wdist:.4g} @ dim={best_dim}"
    else:
        title_suffix = "no finite values"
        ax.text(0.5, 0.5, "No finite Wasserstein values", transform=ax.transAxes, ha="center", va="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks(dim_values)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("RFF embedding dimension")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(f"{BASE_COORD_LABELS[coord_name]} ({run_label}): {title_suffix}")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _add_text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.05, 0.92, title, fontsize=18, fontweight="bold", va="top", ha="left")
    if lines:
        ax.text(0.05, 0.84, "\n".join(lines), fontsize=12, va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def write_summary_pdf(
    plot_entries,
    pdf_path,
    args,
    used_events,
    skipped_events,
    total_hits,
    summary_title="TrackML Hit PosEnc Summary",
):
    if not plot_entries:
        return None

    section_order = {"pair_dots": 0, "sweep": 1, "dim_sweep": 2}
    sorted_entries = sorted(
        plot_entries,
        key=lambda entry: (
            section_order.get(entry["section"], 99),
            entry.get("encoding", ""),
            entry.get("set_name", ""),
            entry.get("coord", ""),
        ),
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _add_text_page(
            pdf,
            summary_title,
            [
                f"Split: {args.split}",
                f"Event start index: {args.event_index}",
                f"Requested events: {args.num_events}",
                f"Used/skipped events: {used_events}/{skipped_events}",
                f"Total valid hits (used events): {total_hits}",
                f"Dim per field: {args.dim_per_field}",
                f"RFF base scale: {args.rff_scale}",
                (
                    "Sweep scales: "
                    f"{args.curve_num_scales} log-spaced in "
                    f"[{args.curve_scale_min_mult}, {args.curve_scale_max_mult}] x base"
                ),
                "Dot-product normalization (all plots): enabled (divide by embedding dim)",
                f"Dimension sweep values: {', '.join(str(int(dim)) for dim in sorted(set(args.dim_values)))}",
                "Dimension sweep dot normalization: enabled (divide by embedding dim)",
                f"Number of plot pages: {len(sorted_entries)}",
            ],
        )

        current_section = None
        for entry in sorted_entries:
            if entry["section"] != current_section:
                current_section = entry["section"]
                section_title_map = {
                    "pair_dots": "Pair Dot Products",
                    "sweep": "RFF Scale Sweep",
                    "dim_sweep": "RFF Dimension Sweep",
                }
                section_title = section_title_map.get(current_section, current_section)
                _add_text_page(pdf, section_title, [])

            fig = plt.figure(figsize=(11, 8.5))
            grid = fig.add_gridspec(2, 1, height_ratios=[0.12, 0.88], hspace=0.02)
            title_ax = fig.add_subplot(grid[0])
            title_ax.axis("off")
            title_ax.text(0.0, 0.9, entry["title"], fontsize=12, fontweight="bold", va="top", ha="left")

            img_ax = fig.add_subplot(grid[1])
            img_ax.axis("off")
            image = plt.imread(entry["path"])
            img_ax.imshow(image)

            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def _format_table_lines(rows, column_order, headers):
    widths = {key: len(headers[key]) for key in column_order}
    for row in rows:
        for key in column_order:
            widths[key] = max(widths[key], len(str(row[key])))

    lines = []
    lines.append("  " + "  ".join(headers[key].ljust(widths[key]) for key in column_order))
    lines.append("  " + "  ".join("-" * widths[key] for key in column_order))
    for row in rows:
        lines.append("  " + "  ".join(str(row[key]).ljust(widths[key]) for key in column_order))
    return lines


def format_optimal_scale_summary(
    curve_results,
    scale_values,
    title="Optimal RFF scales by coordinate (argmax Wasserstein distance):",
):
    rows = []
    for coord_name in BASE_COORD_NAMES:
        payload = curve_results.get(coord_name)
        if payload is None:
            continue

        wdist = payload["wdist"]
        finite = np.isfinite(wdist)
        if np.any(finite):
            max_idx = int(np.nanargmax(wdist))
            best_scale = f"{float(scale_values[max_idx]):.6g}"
            best_wdist = f"{float(wdist[max_idx]):.6g}"
        else:
            best_scale = "n/a"
            best_wdist = "n/a"

        rows.append(
            {
                "coord": coord_name,
                "label": BASE_COORD_LABELS[coord_name],
                "scale": best_scale,
                "wdist": best_wdist,
            }
        )

    lines = [title]
    if not rows:
        lines.append("  (no valid curves)")
        return "\n".join(lines)

    headers = {
        "coord": "coord",
        "label": "label",
        "scale": "best_scale",
        "wdist": "max_wdist",
    }
    lines.extend(_format_table_lines(rows=rows, column_order=["coord", "label", "scale", "wdist"], headers=headers))
    return "\n".join(lines)


def format_optimal_dim_summary(
    curve_results,
    dim_values,
    title="Optimal RFF dimensions by coordinate (argmax Wasserstein distance):",
):
    rows = []
    for coord_name in BASE_COORD_NAMES:
        payload = curve_results.get(coord_name)
        if payload is None:
            continue

        wdist = payload["wdist"]
        finite = np.isfinite(wdist)
        if np.any(finite):
            max_idx = int(np.nanargmax(wdist))
            best_dim = str(int(dim_values[max_idx]))
            best_wdist = f"{float(wdist[max_idx]):.6g}"
        else:
            best_dim = "n/a"
            best_wdist = "n/a"

        rows.append(
            {
                "coord": coord_name,
                "label": BASE_COORD_LABELS[coord_name],
                "dim": best_dim,
                "wdist": best_wdist,
            }
        )

    lines = [title]
    if not rows:
        lines.append("  (no valid curves)")
        return "\n".join(lines)

    headers = {
        "coord": "coord",
        "label": "label",
        "dim": "best_dim",
        "wdist": "max_wdist",
    }
    lines.extend(_format_table_lines(rows=rows, column_order=["coord", "label", "dim", "wdist"], headers=headers))
    return "\n".join(lines)


def derive_best_coord_scales(curve_results, scale_values):
    best_scales = {coord_name: 1.0 for coord_name in BASE_COORD_NAMES}
    source_tags = {coord_name: "fallback" for coord_name in BASE_COORD_NAMES}

    for coord_name in BASE_COORD_NAMES:
        payload = curve_results.get(coord_name)
        if payload is None:
            continue

        wdist = payload["wdist"]
        finite = np.isfinite(wdist)
        if not np.any(finite):
            continue

        max_idx = int(np.nanargmax(wdist))
        best_scale = float(scale_values[max_idx])
        if (not np.isfinite(best_scale)) or (best_scale <= 0.0):
            continue

        best_scales[coord_name] = best_scale
        source_tags[coord_name] = "argmax"

    return best_scales, source_tags


def format_coord_normalization_summary(
    best_scales,
    source_tags,
    title=(
        "Coordinate normalization scales used for scaled run "
        "(coord_scaled = coord / best_scale_original):"
    ),
):
    rows = []
    for coord_name in BASE_COORD_NAMES:
        rows.append(
            {
                "coord": coord_name,
                "label": BASE_COORD_LABELS[coord_name],
                "scale": f"{float(best_scales[coord_name]):.6g}",
                "source": source_tags[coord_name],
            }
        )

    lines = [title]
    headers = {
        "coord": "coord",
        "label": "label",
        "scale": "best_scale_div",
        "source": "source",
    }
    lines.extend(
        _format_table_lines(
            rows=rows,
            column_order=["coord", "label", "scale", "source"],
            headers=headers,
        )
    )
    return "\n".join(lines)


def apply_coord_scaling_to_event_data(event_data, coord_scales):
    scaled_event_data = []
    for event_payload in event_data:
        scaled_coords = {}
        for coord_name, values in event_payload["coords"].items():
            scale_value = float(coord_scales.get(coord_name, 1.0))
            if (not np.isfinite(scale_value)) or (scale_value <= 0.0):
                scale_value = 1.0
            scaled_coords[coord_name] = values / scale_value

        scaled_event_data.append(
            {
                "num_hits": event_payload["num_hits"],
                "coords": scaled_coords,
                "same_pairs": event_payload["same_pairs"],
                "not_same_pairs": event_payload["not_same_pairs"],
            }
        )

    return scaled_event_data


def write_text_report(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def run_pair_dot_plots(
    event_data,
    rff_base_matrices,
    dim_per_field,
    alpha,
    base,
    rff_scale,
    output_root,
    used_events,
    total_hits,
):
    encoding_methods = {
        "sinusoidal": "Sinusoidal",
        "random_fourier": "Random Fourier Features",
    }
    written_paths = []
    plot_entries = []

    def init_score_buffers():
        score_buffers = {}
        for encoding_name in encoding_methods:
            score_buffers[encoding_name] = {}
            for set_name, set_cfg in PAIR_FIELD_SETS.items():
                score_fields = ["combined"] + list(set_cfg["fields"].keys())
                score_buffers[encoding_name][set_name] = {field: [] for field in score_fields}
        return score_buffers

    per_encoding_set_scores_same = init_score_buffers()
    per_encoding_set_scores_random = init_score_buffers()

    for event_payload in event_data:
        for encoding_name in encoding_methods:
            for set_name, set_cfg in PAIR_FIELD_SETS.items():
                encodings = encode_field_set(
                    coords=event_payload["coords"],
                    field_map=set_cfg["fields"],
                    encoding_name=encoding_name,
                    dim_per_field=dim_per_field,
                    alpha=alpha,
                    base=base,
                    rff_base_matrices=rff_base_matrices,
                    rff_scale=rff_scale,
                )
                scores_same = {
                    name: pair_dot_products(emb, event_payload["same_pairs"])
                    for name, emb in encodings.items()
                }
                scores_random = {
                    name: pair_dot_products(emb, event_payload["not_same_pairs"])
                    for name, emb in encodings.items()
                }

                for field_name in scores_same:
                    per_encoding_set_scores_same[encoding_name][set_name][field_name].append(scores_same[field_name])
                    per_encoding_set_scores_random[encoding_name][set_name][field_name].append(scores_random[field_name])

    for encoding_name, encoding_label in encoding_methods.items():
        for set_name, set_cfg in PAIR_FIELD_SETS.items():
            scores_same = {
                field: torch.cat(chunks, dim=0)
                for field, chunks in per_encoding_set_scores_same[encoding_name][set_name].items()
                if chunks
            }
            scores_random = {
                field: torch.cat(chunks, dim=0)
                for field, chunks in per_encoding_set_scores_random[encoding_name][set_name].items()
                if chunks
            }

            expected_fields = ["combined"] + list(set_cfg["fields"].keys())
            if any(field not in scores_same for field in expected_fields):
                continue

            out_path = output_root / "pair_dots" / encoding_name / f"{set_name}.png"
            plot_dot_products(
                scores_same=scores_same,
                scores_random=scores_random,
                out_path=out_path,
                field_order=list(set_cfg["fields"].keys()),
                title_prefix=f"HIT {set_cfg['title_suffix']} [{encoding_label}]",
                num_events=used_events,
                hit_summary=str(total_hits),
            )
            written_paths.append(out_path)
            plot_entries.append(
                {
                    "path": out_path,
                    "section": "pair_dots",
                    "encoding": encoding_name,
                    "set_name": set_name,
                    "title": f"Hit {set_cfg['title_suffix']} [{encoding_label}]",
                }
            )

    return written_paths, plot_entries


def run_rff_scale_sweep(event_data, rff_base_matrices, dim_per_field, scale_values, output_root):
    num_scales = len(scale_values)
    scale_tensor = torch.as_tensor(scale_values, dtype=torch.float32)

    curve_scores_same = {
        coord_name: [[] for _ in range(num_scales)]
        for coord_name in BASE_COORD_NAMES
    }
    curve_scores_not_same = {
        coord_name: [[] for _ in range(num_scales)]
        for coord_name in BASE_COORD_NAMES
    }

    for event_payload in event_data:
        for coord_name in BASE_COORD_NAMES:
            embeddings_multi = random_fourier_pos_enc_multi(
                xs=event_payload["coords"][coord_name],
                base_matrix=rff_base_matrices[coord_name],
                scales=scale_tensor,
                dim_per_field=dim_per_field,
            )
            same_multi = pair_dot_products_multi(embeddings_multi, event_payload["same_pairs"])
            not_same_multi = pair_dot_products_multi(embeddings_multi, event_payload["not_same_pairs"])

            for scale_idx in range(num_scales):
                curve_scores_same[coord_name][scale_idx].append(same_multi[scale_idx])
                curve_scores_not_same[coord_name][scale_idx].append(not_same_multi[scale_idx])

    curve_results = {}
    written_paths = []
    plot_entries = []

    for coord_name in BASE_COORD_NAMES:
        wdist_curve = np.full(num_scales, np.nan, dtype=np.float64)
        for scale_idx in range(num_scales):
            chunks_same = curve_scores_same[coord_name][scale_idx]
            chunks_not = curve_scores_not_same[coord_name][scale_idx]
            if chunks_same and chunks_not:
                same = torch.cat(chunks_same, dim=0).detach().cpu().numpy()
                not_same = torch.cat(chunks_not, dim=0).detach().cpu().numpy()
                if same.size and not_same.size:
                    wdist_curve[scale_idx] = float(wasserstein_distance(same, not_same))

        curve_results[coord_name] = {"wdist": wdist_curve}

        coord_out_path = output_root / "rff_scale_sweep" / f"wasserstein_vs_scale_{coord_name}.png"
        plot_wasserstein_vs_scale_single(
            scale_values=scale_values,
            wdist=wdist_curve,
            coord_name=coord_name,
            out_path=coord_out_path,
        )
        written_paths.append(coord_out_path)
        plot_entries.append(
            {
                "path": coord_out_path,
                "section": "sweep",
                "coord": coord_name,
                "title": f"RFF sweep: {BASE_COORD_LABELS[coord_name]}",
            }
        )

    summary_out_path = output_root / "rff_scale_sweep" / "wasserstein_vs_scale_all_coords.png"
    plot_wasserstein_vs_scale_all(
        curve_results=curve_results,
        scale_values=scale_values,
        out_path=summary_out_path,
    )
    written_paths.append(summary_out_path)
    plot_entries.append(
        {
            "path": summary_out_path,
            "section": "sweep",
            "coord": "all",
            "title": "RFF sweep summary: all coordinates",
        }
    )

    return curve_results, written_paths, plot_entries


def run_rff_dim_sweep(event_data, rff_base_matrices, dim_values, rff_scale, output_root, run_label):
    num_dims = len(dim_values)
    half_dims = (dim_values // 2).astype(np.int64)
    max_half_dim = int(half_dims.max())
    half_dim_indices = torch.as_tensor(half_dims - 1, dtype=torch.long)
    dim_values_f32 = np.asarray(dim_values, dtype=np.float32)

    curve_scores_same = {
        coord_name: [[] for _ in range(num_dims)]
        for coord_name in BASE_COORD_NAMES
    }
    curve_scores_not_same = {
        coord_name: [[] for _ in range(num_dims)]
        for coord_name in BASE_COORD_NAMES
    }

    for event_payload in event_data:
        for coord_name in BASE_COORD_NAMES:
            sin_vals, cos_vals = random_fourier_pos_enc_trig(
                xs=event_payload["coords"][coord_name],
                base_matrix=rff_base_matrices[coord_name],
                scale=rff_scale,
                max_half_dim=max_half_dim,
            )
            same_multi = pair_dot_products_multi_from_trig(
                sin_vals=sin_vals,
                cos_vals=cos_vals,
                pairs=event_payload["same_pairs"],
                half_dim_indices=half_dim_indices,
                dim_values=dim_values_f32,
            )
            not_same_multi = pair_dot_products_multi_from_trig(
                sin_vals=sin_vals,
                cos_vals=cos_vals,
                pairs=event_payload["not_same_pairs"],
                half_dim_indices=half_dim_indices,
                dim_values=dim_values_f32,
            )

            for dim_idx in range(num_dims):
                curve_scores_same[coord_name][dim_idx].append(same_multi[dim_idx])
                curve_scores_not_same[coord_name][dim_idx].append(not_same_multi[dim_idx])

    curve_results = {}
    written_paths = []
    plot_entries = []

    for coord_name in BASE_COORD_NAMES:
        wdist_curve = np.full(num_dims, np.nan, dtype=np.float64)
        for dim_idx in range(num_dims):
            chunks_same = curve_scores_same[coord_name][dim_idx]
            chunks_not = curve_scores_not_same[coord_name][dim_idx]
            if chunks_same and chunks_not:
                same = torch.cat(chunks_same, dim=0).detach().cpu().numpy()
                not_same = torch.cat(chunks_not, dim=0).detach().cpu().numpy()
                if same.size and not_same.size:
                    wdist_curve[dim_idx] = float(wasserstein_distance(same, not_same))

        curve_results[coord_name] = {"wdist": wdist_curve}

        coord_out_path = output_root / "rff_dim_sweep" / f"wasserstein_vs_dim_{coord_name}.png"
        plot_wasserstein_vs_dim_single(
            dim_values=dim_values,
            wdist=wdist_curve,
            coord_name=coord_name,
            out_path=coord_out_path,
            run_label=run_label,
        )
        written_paths.append(coord_out_path)
        plot_entries.append(
            {
                "path": coord_out_path,
                "section": "dim_sweep",
                "coord": coord_name,
                "title": (
                    f"RFF dim sweep ({run_label}, dot/dim): "
                    f"{BASE_COORD_LABELS[coord_name]}"
                ),
            }
        )

    summary_out_path = output_root / "rff_dim_sweep" / "wasserstein_vs_dim_all_coords.png"
    plot_wasserstein_vs_dim_all(
        curve_results=curve_results,
        dim_values=dim_values,
        out_path=summary_out_path,
        run_label=run_label,
    )
    written_paths.append(summary_out_path)
    plot_entries.append(
        {
            "path": summary_out_path,
            "section": "dim_sweep",
            "coord": "all",
            "title": f"RFF dim sweep summary ({run_label}, dot/dim): all coordinates",
        }
    )

    return curve_results, written_paths, plot_entries


def main():
    args = parse_args()

    if args.dim_per_field < 2:
        raise ValueError("--dim-per-field must be >= 2")
    if args.rff_scale <= 0:
        raise ValueError("--rff-scale must be > 0")
    if args.curve_num_scales <= 0:
        raise ValueError("--curve-num-scales must be > 0")
    if args.curve_scale_min_mult <= 0 or args.curve_scale_max_mult <= 0:
        raise ValueError("--curve-scale-min-mult and --curve-scale-max-mult must be > 0")
    if args.curve_scale_min_mult > args.curve_scale_max_mult:
        raise ValueError("--curve-scale-min-mult must be <= --curve-scale-max-mult")
    if not args.dim_values:
        raise ValueError("--dim-values must not be empty")

    dim_values = np.asarray(sorted(set(args.dim_values)), dtype=np.int64)
    for dim_value in dim_values:
        if dim_value < 2:
            raise ValueError(f"All --dim-values must be >= 2, got {dim_value}")
        if (dim_value // 2) == 0:
            raise ValueError(f"All --dim-values must satisfy dim//2 >= 1, got {dim_value}")

    rng = np.random.default_rng(args.seed)

    dataset = load_dataset(args.config, args.split, min_num_events=args.event_index + args.num_events)
    events = load_events(dataset, args.event_index, args.num_events)

    event_data = []
    skipped_events = 0
    total_hits = 0
    for inputs, targets in events:
        prepared, _reason = prepare_event_data(
            inputs=inputs,
            targets=targets,
            max_same_pairs=args.max_same_pairs,
            rng=rng,
        )
        if prepared is None:
            skipped_events += 1
            continue

        event_data.append(prepared)
        total_hits += prepared["num_hits"]

    if not event_data:
        raise RuntimeError("No usable events after pair-building. Try increasing --num-events or relaxing cuts.")

    used_events = len(event_data)
    rff_base_matrices = init_rff_base_matrices(BASE_COORD_NAMES, args.dim_per_field, args.seed)
    rff_dim_base_matrices = init_rff_base_matrices_max_half(
        BASE_COORD_NAMES,
        max_half_dim=int((dim_values // 2).max()),
        seed=args.seed,
    )

    scale_multipliers = np.logspace(
        np.log10(args.curve_scale_min_mult),
        np.log10(args.curve_scale_max_mult),
        args.curve_num_scales,
    )
    scale_values = args.rff_scale * scale_multipliers

    output_root = args.out_dir / args.out_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    original_output_root = output_root / "original"
    scaled_output_root = output_root / "scaled"

    original_pair_written_paths, original_pair_plot_entries = run_pair_dot_plots(
        event_data=event_data,
        rff_base_matrices=rff_base_matrices,
        dim_per_field=args.dim_per_field,
        alpha=args.alpha,
        base=args.base,
        rff_scale=args.rff_scale,
        output_root=original_output_root,
        used_events=used_events,
        total_hits=total_hits,
    )
    curve_results_original, original_sweep_written_paths, original_sweep_plot_entries = run_rff_scale_sweep(
        event_data=event_data,
        rff_base_matrices=rff_base_matrices,
        dim_per_field=args.dim_per_field,
        scale_values=scale_values,
        output_root=original_output_root,
    )
    curve_results_dim_original, original_dim_written_paths, original_dim_plot_entries = run_rff_dim_sweep(
        event_data=event_data,
        rff_base_matrices=rff_dim_base_matrices,
        dim_values=dim_values,
        rff_scale=args.rff_scale,
        output_root=original_output_root,
        run_label="original",
    )

    original_optimal_scale_text = format_optimal_scale_summary(
        curve_results=curve_results_original,
        scale_values=scale_values,
        title="Optimal RFF scales by coordinate (argmax Wasserstein distance) [original coords]:",
    )
    print("")
    print(original_optimal_scale_text)

    original_optimal_scale_path = original_output_root / "tables" / "optimal_scales.txt"
    write_text_report(original_optimal_scale_path, original_optimal_scale_text)

    original_optimal_dim_text = format_optimal_dim_summary(
        curve_results=curve_results_dim_original,
        dim_values=dim_values,
        title=(
            "Optimal RFF dimensions by coordinate (argmax Wasserstein distance) "
            "[original coords; dot normalized by dim]:"
        ),
    )
    print("")
    print(original_optimal_dim_text)

    original_optimal_dim_path = original_output_root / "tables" / "optimal_dims.txt"
    write_text_report(original_optimal_dim_path, original_optimal_dim_text)

    best_coord_scales, source_tags = derive_best_coord_scales(
        curve_results=curve_results_original,
        scale_values=scale_values,
    )
    normalization_text = format_coord_normalization_summary(
        best_scales=best_coord_scales,
        source_tags=source_tags,
        title=(
            "Coordinate normalization scales used for scaled run "
            "(coord_scaled = coord / best_scale_original):"
        ),
    )
    print("")
    print(normalization_text)

    normalization_path = scaled_output_root / "tables" / "coord_normalization_scales.txt"
    write_text_report(normalization_path, normalization_text)

    scaled_event_data = apply_coord_scaling_to_event_data(
        event_data=event_data,
        coord_scales=best_coord_scales,
    )
    scaled_pair_written_paths, scaled_pair_plot_entries = run_pair_dot_plots(
        event_data=scaled_event_data,
        rff_base_matrices=rff_base_matrices,
        dim_per_field=args.dim_per_field,
        alpha=args.alpha,
        base=args.base,
        rff_scale=args.rff_scale,
        output_root=scaled_output_root,
        used_events=used_events,
        total_hits=total_hits,
    )
    curve_results_scaled, scaled_sweep_written_paths, scaled_sweep_plot_entries = run_rff_scale_sweep(
        event_data=scaled_event_data,
        rff_base_matrices=rff_base_matrices,
        dim_per_field=args.dim_per_field,
        scale_values=scale_values,
        output_root=scaled_output_root,
    )
    curve_results_dim_scaled, scaled_dim_written_paths, scaled_dim_plot_entries = run_rff_dim_sweep(
        event_data=scaled_event_data,
        rff_base_matrices=rff_dim_base_matrices,
        dim_values=dim_values,
        rff_scale=args.rff_scale,
        output_root=scaled_output_root,
        run_label="scaled",
    )

    scaled_optimal_scale_text = format_optimal_scale_summary(
        curve_results=curve_results_scaled,
        scale_values=scale_values,
        title=(
            "Optimal RFF scales by coordinate (argmax Wasserstein distance) [scaled coords; "
            "expected near 1]:"
        ),
    )
    print("")
    print(scaled_optimal_scale_text)

    scaled_optimal_scale_path = scaled_output_root / "tables" / "optimal_scales.txt"
    write_text_report(scaled_optimal_scale_path, scaled_optimal_scale_text)

    scaled_optimal_dim_text = format_optimal_dim_summary(
        curve_results=curve_results_dim_scaled,
        dim_values=dim_values,
        title=(
            "Optimal RFF dimensions by coordinate (argmax Wasserstein distance) "
            "[scaled coords; dot normalized by dim]:"
        ),
    )
    print("")
    print(scaled_optimal_dim_text)

    scaled_optimal_dim_path = scaled_output_root / "tables" / "optimal_dims.txt"
    write_text_report(scaled_optimal_dim_path, scaled_optimal_dim_text)

    for path in [*original_pair_written_paths, *original_sweep_written_paths, *original_dim_written_paths]:
        print(f"[original] Wrote {path}")
    print(f"[original] Wrote {original_optimal_scale_path}")
    print(f"[original] Wrote {original_optimal_dim_path}")
    print(f"[scaled] Wrote {normalization_path}")
    for path in [*scaled_pair_written_paths, *scaled_sweep_written_paths, *scaled_dim_written_paths]:
        print(f"[scaled] Wrote {path}")
    print(f"[scaled] Wrote {scaled_optimal_scale_path}")
    print(f"[scaled] Wrote {scaled_optimal_dim_path}")

    if not args.skip_pdf:
        pdf_name = args.pdf_path.name if args.pdf_path is not None else "summary.pdf"

        original_pdf_path = original_output_root / pdf_name
        written_original_pdf = write_summary_pdf(
            plot_entries=[*original_pair_plot_entries, *original_sweep_plot_entries, *original_dim_plot_entries],
            pdf_path=original_pdf_path,
            args=args,
            used_events=used_events,
            skipped_events=skipped_events,
            total_hits=total_hits,
            summary_title="TrackML Hit PosEnc Summary (original)",
        )
        if written_original_pdf is not None:
            print(f"[original] Wrote {written_original_pdf}")

        scaled_pdf_path = scaled_output_root / pdf_name
        written_scaled_pdf = write_summary_pdf(
            plot_entries=[*scaled_pair_plot_entries, *scaled_sweep_plot_entries, *scaled_dim_plot_entries],
            pdf_path=scaled_pdf_path,
            args=args,
            used_events=used_events,
            skipped_events=skipped_events,
            total_hits=total_hits,
            summary_title="TrackML Hit PosEnc Summary (scaled)",
        )
        if written_scaled_pdf is not None:
            print(f"[scaled] Wrote {written_scaled_pdf}")

    print(f"Split: {args.split}")
    print(f"Event index: {args.event_index}, requested num_events: {args.num_events}")
    print(f"Used/skipped events: {used_events}/{skipped_events}")
    print(f"Total valid hits in used events: {total_hits}")
    print(
        "Sweep scales: "
        f"{args.curve_num_scales} log-spaced in "
        f"[{args.curve_scale_min_mult}, {args.curve_scale_max_mult}] x base scale"
    )
    print(f"Dimension sweep values: {', '.join(str(int(dim)) for dim in dim_values)}")
    print("Dimension sweep dot-product normalization: enabled (divide by dim)")
    print(f"Output roots: {original_output_root} (original), {scaled_output_root} (scaled)")


if __name__ == "__main__":
    main()
