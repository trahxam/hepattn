#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance

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


def build_field_sets():
    field_sets = {}
    for coord_name in BASE_COORD_NAMES:
        label = BASE_COORD_LABELS[coord_name]
        field_sets[coord_name] = {"fields": {coord_name: coord_name}, "title_suffix": f"({label})"}
    return field_sets


def build_set_title_map():
    title_map = {}
    for coord_name in BASE_COORD_NAMES:
        label = BASE_COORD_LABELS[coord_name]
        title_map[coord_name] = f"Wasserstein vs RFF Scale ({label})"
    return title_map


def build_set_order():
    ordered_names = list(BASE_COORD_NAMES)
    return {name: idx for idx, name in enumerate(ordered_names)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan random Fourier feature scales and plot Wasserstein distance between "
            "same-particle and non-same-particle hit-pair dot-product distributions."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("src/hepattn/experiments/cld/configs/unified.yaml"))
    parser.add_argument("--event-index", type=int, default=0, help="Zero-based start index in the test dataloader.")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events to aggregate for statistics.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim-per-field", type=int, default=32, help="RFF embedding width per field.")
    parser.add_argument("--rff-scale", type=float, default=1.0, help="Default RFF scale before multipliers are applied.")
    parser.add_argument(
        "--curve-num-scales",
        type=int,
        default=32,
        help="Number of log-spaced scales for Wasserstein-vs-scale summary curves.",
    )
    parser.add_argument(
        "--curve-scale-min-mult",
        type=float,
        default=0.1,
        help="Minimum multiplier (relative to --rff-scale) for Wasserstein-vs-scale summary curves.",
    )
    parser.add_argument(
        "--curve-scale-max-mult",
        type=float,
        default=100.0,
        help="Maximum multiplier (relative to --rff-scale) for Wasserstein-vs-scale summary curves.",
    )
    parser.add_argument(
        "--cross2d-num-scales",
        type=int,
        default=16,
        help="Number of log-spaced scales per axis for cross-pair 2D scans.",
    )
    parser.add_argument(
        "--cross2d-scale-min-mult",
        type=float,
        default=0.1,
        help="Minimum multiplier (relative to --rff-scale) for cross-pair 2D scans.",
    )
    parser.add_argument(
        "--cross2d-scale-max-mult",
        type=float,
        default=100.0,
        help="Maximum multiplier (relative to --rff-scale) for cross-pair 2D scans.",
    )
    parser.add_argument(
        "--cross2d-max-pairs",
        type=int,
        default=5000,
        help="Cap same/non-same pairs per event for cross-pair 2D scans (<=0 uses all).",
    )
    parser.add_argument("--skip-cross2d", action="store_true", help="If set, skip cross-pair 2D scale scan heatmaps.")
    parser.add_argument("--max-same-pairs", type=int, default=25000, help="Subsample same-particle pairs to this cap per event.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("src/hepattn/experiments/cld/plots/data"))
    parser.add_argument("--out-prefix", type=str, default="posenc_rff_scale_sweep")
    parser.add_argument("--pdf-path", type=Path, default=None, help="Optional path for consolidated summary PDF.")
    parser.add_argument("--skip-pdf", action="store_true", help="If set, skip writing the consolidated summary PDF.")
    parser.add_argument(
        "--subsystems",
        nargs="+",
        default=["vtxd", "trkr", "ecal", "hcal"],
        help="Subsystem inputs to plot (e.g. vtxd trkr ecal hcal).",
    )
    parser.add_argument(
        "--cross-subsystem-pairs",
        nargs="+",
        default=["vtxd-trkr", "trkr-ecal", "ecal-hcal"],
        help="Cross-subsystem pairs in the form A-B (e.g. vtxd-trkr trkr-ecal).",
    )
    return parser.parse_args()


def parse_cross_subsystem_pairs(pair_specs):
    parsed_pairs = []
    seen = set()
    for pair_spec in pair_specs:
        split = pair_spec.split("-")
        if len(split) != 2 or (not split[0]) or (not split[1]):
            raise ValueError(
                "Invalid --cross-subsystem-pairs value "
                f"{pair_spec!r}. Expected format A-B, e.g. vtxd-trkr."
            )
        pair = (split[0], split[1])
        if pair in seen:
            continue
        parsed_pairs.append(pair)
        seen.add(pair)
    return parsed_pairs


def apply_coord_scaling_for_subsystem(coords, subsystem, coord_scale_by_subsystem):
    if coord_scale_by_subsystem is None:
        return coords

    subsystem_scales = coord_scale_by_subsystem.get(subsystem, {})
    scaled_coords = {}
    for coord_name, values in coords.items():
        scale_value = float(subsystem_scales.get(coord_name, 1.0))
        if (not np.isfinite(scale_value)) or (scale_value <= 0.0):
            scale_value = 1.0
        scaled_coords[coord_name] = values / scale_value
    return scaled_coords


def undo_coord_scaling_for_subsystem(coords, subsystem, coord_scale_by_subsystem):
    if coord_scale_by_subsystem is None:
        return coords

    subsystem_scales = coord_scale_by_subsystem.get(subsystem, {})
    unscaled_coords = {}
    for coord_name, values in coords.items():
        scale_value = float(subsystem_scales.get(coord_name, 1.0))
        if (not np.isfinite(scale_value)) or (scale_value <= 0.0):
            scale_value = 1.0
        unscaled_coords[coord_name] = values * scale_value
    return unscaled_coords


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
        raise RuntimeError(
            f"Requested {num_events} events from index {event_index}, but only loaded {len(events)}."
        )

    return events


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


def extract_subsystem_view(inputs, targets, subsystem, coord_scale_by_subsystem=None):
    hit_valid_key = f"{subsystem}_valid"
    particle_hit_key = f"particle_{subsystem}_valid"

    if hit_valid_key not in inputs:
        return None, "missing input valid mask"
    if particle_hit_key not in targets:
        return None, "missing particle->hit truth mask"

    hit_valid = inputs[hit_valid_key][0].bool()
    num_hits = int(hit_valid.sum().item())
    if num_hits < 1:
        return None, "fewer than 1 valid hit"

    particle_hit_valid = targets[particle_hit_key][0][:, hit_valid].bool()
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
    coords = apply_coord_scaling_for_subsystem(
        coords=coords,
        subsystem=subsystem,
        coord_scale_by_subsystem=coord_scale_by_subsystem,
    )
    return {
        "num_hits": num_hits,
        "coords": coords,
        "particle_hit_valid": particle_hit_valid,
    }, None


def prepare_subsystem_data(inputs, targets, subsystem, max_same_pairs, rng, coord_scale_by_subsystem=None):
    subsystem_view, reason = extract_subsystem_view(
        inputs=inputs,
        targets=targets,
        subsystem=subsystem,
        coord_scale_by_subsystem=coord_scale_by_subsystem,
    )
    if subsystem_view is None:
        return None, reason

    if subsystem_view["num_hits"] < 2:
        return None, "fewer than 2 valid hits"

    particle_valid = targets["particle_valid"][0].bool()
    same_pairs = collect_same_particle_pairs(subsystem_view["particle_hit_valid"], particle_valid)
    same_pairs = subsample_pairs(same_pairs, max_same_pairs, rng)
    if same_pairs.numel() == 0:
        return None, "no same-particle hit pairs"

    hit_to_particle = subsystem_view["particle_hit_valid"][particle_valid].T.contiguous()
    not_same_pairs = sample_not_same_pairs(hit_to_particle=hit_to_particle, num_pairs=same_pairs.shape[0], rng=rng)
    if not_same_pairs.shape[0] == 0:
        return None, "failed to sample non-same-particle hit pairs"

    return {
        "num_hits": subsystem_view["num_hits"],
        "coords": subsystem_view["coords"],
        "same_pairs": same_pairs,
        "not_same_pairs": not_same_pairs,
    }, None


def collect_cross_system_same_pairs(particle_hit_valid_a, particle_hit_valid_b, particle_valid, max_pairs, rng):
    per_particle = []
    pair_counts = []

    for hit_mask_a, hit_mask_b in zip(particle_hit_valid_a[particle_valid], particle_hit_valid_b[particle_valid]):
        idx_a = torch.nonzero(hit_mask_a, as_tuple=False).flatten()
        idx_b = torch.nonzero(hit_mask_b, as_tuple=False).flatten()

        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue

        count = int(idx_a.numel() * idx_b.numel())
        per_particle.append((idx_a, idx_b, count))
        pair_counts.append(count)

    if not per_particle:
        return torch.empty((0, 2), dtype=torch.long)

    total_pairs = int(sum(pair_counts))
    target_pairs = total_pairs if max_pairs <= 0 else min(total_pairs, int(max_pairs))
    sampled_pairs = []

    if target_pairs >= total_pairs:
        for idx_a, idx_b, _count in per_particle:
            left = idx_a.repeat_interleave(idx_b.numel())
            right = idx_b.repeat(idx_a.numel())
            sampled_pairs.append(torch.stack((left, right), dim=1))
    else:
        probs = np.asarray(pair_counts, dtype=np.float64)
        probs = probs / probs.sum()
        samples_per_particle = rng.multinomial(target_pairs, probs)

        for (idx_a, idx_b, count), n_sample in zip(per_particle, samples_per_particle):
            if n_sample <= 0:
                continue

            n_sample = int(min(n_sample, count))
            num_b = int(idx_b.numel())
            flat_choices = rng.choice(count, size=n_sample, replace=False)
            flat_idx = torch.as_tensor(flat_choices, dtype=torch.long)
            left = idx_a[flat_idx // num_b]
            right = idx_b[flat_idx % num_b]
            sampled_pairs.append(torch.stack((left, right), dim=1))

    if not sampled_pairs:
        return torch.empty((0, 2), dtype=torch.long)

    pairs = torch.cat(sampled_pairs, dim=0)
    return torch.unique(pairs, dim=0)


def sample_not_same_cross_pairs(hit_to_particle_a, hit_to_particle_b, num_pairs, rng):
    num_hits_a = int(hit_to_particle_a.shape[0])
    num_hits_b = int(hit_to_particle_b.shape[0])
    if num_hits_a < 1 or num_hits_b < 1 or num_pairs <= 0:
        return torch.empty((0, 2), dtype=torch.long)

    target = min(int(num_pairs), int(num_hits_a * num_hits_b))
    pair_set = set()
    attempt_limit = max(200, 20 * target)
    attempts = 0

    while len(pair_set) < target and attempts < attempt_limit:
        batch_size = max(1024, 4 * (target - len(pair_set)))
        a = rng.integers(0, num_hits_a, size=batch_size)
        b = rng.integers(0, num_hits_b, size=batch_size)
        candidates = np.stack((a, b), axis=1)
        candidates = np.unique(candidates, axis=0)
        pair_tensor = torch.as_tensor(candidates, dtype=torch.long)

        if pair_tensor.numel() == 0:
            attempts += batch_size
            continue

        share_particle = (hit_to_particle_a[pair_tensor[:, 0]] & hit_to_particle_b[pair_tensor[:, 1]]).any(dim=1)
        diff_pairs = pair_tensor[~share_particle]

        for i, j in diff_pairs.tolist():
            pair_set.add((int(i), int(j)))
            if len(pair_set) >= target:
                break

        attempts += batch_size

    if not pair_set:
        return torch.empty((0, 2), dtype=torch.long)

    return torch.tensor(list(pair_set), dtype=torch.long)


def prepare_cross_subsystem_data(
    inputs,
    targets,
    subsystem_a,
    subsystem_b,
    max_same_pairs,
    rng,
    coord_scale_by_subsystem=None,
):
    subsystem_a_view, reason_a = extract_subsystem_view(
        inputs=inputs,
        targets=targets,
        subsystem=subsystem_a,
        coord_scale_by_subsystem=coord_scale_by_subsystem,
    )
    if subsystem_a_view is None:
        return None, f"{subsystem_a}: {reason_a}"

    subsystem_b_view, reason_b = extract_subsystem_view(
        inputs=inputs,
        targets=targets,
        subsystem=subsystem_b,
        coord_scale_by_subsystem=coord_scale_by_subsystem,
    )
    if subsystem_b_view is None:
        return None, f"{subsystem_b}: {reason_b}"

    particle_valid = targets["particle_valid"][0].bool()
    same_pairs = collect_cross_system_same_pairs(
        particle_hit_valid_a=subsystem_a_view["particle_hit_valid"],
        particle_hit_valid_b=subsystem_b_view["particle_hit_valid"],
        particle_valid=particle_valid,
        max_pairs=max_same_pairs,
        rng=rng,
    )
    if same_pairs.numel() == 0:
        return None, "no same-particle cross-system hit pairs"

    hit_to_particle_a = subsystem_a_view["particle_hit_valid"][particle_valid].T.contiguous()
    hit_to_particle_b = subsystem_b_view["particle_hit_valid"][particle_valid].T.contiguous()
    not_same_pairs = sample_not_same_cross_pairs(
        hit_to_particle_a=hit_to_particle_a,
        hit_to_particle_b=hit_to_particle_b,
        num_pairs=same_pairs.shape[0],
        rng=rng,
    )
    if not_same_pairs.shape[0] == 0:
        return None, "failed to sample non-same-particle cross-system hit pairs"

    return {
        "num_hits_a": subsystem_a_view["num_hits"],
        "num_hits_b": subsystem_b_view["num_hits"],
        "coords_a": subsystem_a_view["coords"],
        "coords_b": subsystem_b_view["coords"],
        "same_pairs": same_pairs,
        "not_same_pairs": not_same_pairs,
    }, None


def init_rff_base_matrices(coord_names, dim_per_field, seed):
    half_dim = dim_per_field // 2
    if half_dim == 0:
        raise ValueError("--dim-per-field must be >= 2 for random Fourier features.")

    gen = torch.Generator()
    gen.manual_seed(seed)
    matrices = {}
    for coord_name in sorted(coord_names):
        matrices[coord_name] = torch.randn((1, half_dim), generator=gen)
    return matrices


def random_fourier_pos_enc(xs, base_matrix, scale, dim_per_field):
    matrix = ((1 / scale) * base_matrix).to(device=xs.device, dtype=xs.dtype)
    proj = (2.0 * torch.pi * xs.unsqueeze(-1)) * matrix
    enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    if enc.shape[-1] < dim_per_field:
        enc = torch.cat([enc, torch.zeros_like(enc[..., : dim_per_field - enc.shape[-1]])], dim=-1)
    return enc


def random_fourier_pos_enc_multi(xs, base_matrix, scales, dim_per_field):
    scales_tensor = torch.as_tensor(scales, device=xs.device, dtype=xs.dtype)
    matrix = ((1.0 / scales_tensor)[:, None, None] * base_matrix.to(device=xs.device, dtype=xs.dtype)[None, :, :])
    proj = (2.0 * torch.pi * xs)[None, :, None] * matrix
    enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    if enc.shape[-1] < dim_per_field:
        pad = torch.zeros((*enc.shape[:-1], dim_per_field - enc.shape[-1]), device=enc.device, dtype=enc.dtype)
        enc = torch.cat([enc, pad], dim=-1)
    return enc


def encode_field_set_rff(coords, field_map, base_matrices, scale, dim_per_field):
    encodings = {}
    per_field_encodings = []
    for field_name, coord_name in field_map.items():
        values = coords[coord_name]
        field_encoding = random_fourier_pos_enc(values, base_matrix=base_matrices[coord_name], scale=scale, dim_per_field=dim_per_field)
        encodings[field_name] = field_encoding
        per_field_encodings.append(field_encoding)
    encodings["combined"] = torch.cat(per_field_encodings, dim=-1)
    return encodings


def pair_dot_products(embeddings_left, pairs, embeddings_right=None):
    if pairs.numel() == 0:
        return torch.empty((0,), dtype=embeddings_left.dtype)

    if embeddings_right is None:
        embeddings_right = embeddings_left

    left = embeddings_left[pairs[:, 0]]
    right = embeddings_right[pairs[:, 1]]
    return (left * right).sum(dim=-1)


def pair_dot_products_multi(embeddings_left, pairs, embeddings_right=None):
    num_scales = embeddings_left.shape[0]
    if pairs.numel() == 0:
        return torch.empty((num_scales, 0), dtype=embeddings_left.dtype, device=embeddings_left.device)

    if embeddings_right is None:
        embeddings_right = embeddings_left

    pairs = pairs.to(device=embeddings_left.device)
    left = embeddings_left[:, pairs[:, 0], :]
    right = embeddings_right[:, pairs[:, 1], :]
    return (left * right).sum(dim=-1)


def plot_wasserstein_vs_scale(curve_data, out_path, set_name):
    title_map = build_set_title_map()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    def line_sort_key(item):
        pair_name, payload = item
        return (0 if payload["kind"] == "intra" else 1, pair_name)

    for pair_name, payload in sorted(curve_data.items(), key=line_sort_key):
        scales = payload["scales"]
        wdist = payload["wdist"]
        finite = np.isfinite(wdist)
        if not np.any(finite):
            continue
        max_idx = int(np.nanargmax(wdist))
        max_scale = float(scales[max_idx])
        legend_label = f"{pair_name} (max@{max_scale:.3g})"

        linestyle = "-" if payload["kind"] == "intra" else "--"
        marker = "o" if payload["kind"] == "intra" else "s"
        ax.plot(
            scales,
            wdist,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.4,
            markersize=3.5,
            label=legend_label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("RFF scale")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(title_map.get(set_name, f"Wasserstein vs RFF Scale ({set_name})"))
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _set_display_label(set_name):
    return BASE_COORD_LABELS.get(set_name, set_name)


def plot_cross_pair_wasserstein_2d(wdist_grid, scale_values_x, scale_values_y, out_path, subsystem_x, subsystem_y, set_name):
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    masked = np.ma.masked_invalid(wdist_grid)
    best = None

    if masked.count() > 0:
        mesh = ax.pcolormesh(scale_values_x, scale_values_y, masked, shading="auto", cmap="viridis")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Wasserstein distance")

        best_flat_idx = int(np.nanargmax(wdist_grid))
        best_y_idx, best_x_idx = np.unravel_index(best_flat_idx, wdist_grid.shape)
        best_x = float(scale_values_x[best_x_idx])
        best_y = float(scale_values_y[best_y_idx])
        best_wdist = float(wdist_grid[best_y_idx, best_x_idx])
        ax.plot(best_x, best_y, marker="x", markersize=9, markeredgewidth=2, color="red")
        ax.axvline(best_x, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axhline(best_y, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
        best = {
            "best_scale_x": best_x,
            "best_scale_y": best_y,
            "best_wdist": best_wdist,
        }
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"best {subsystem_x}: {best_x:.6g}",
                    f"best {subsystem_y}: {best_y:.6g}",
                    f"max W: {best_wdist:.6g}",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
        )
        title_suffix = f"max={best_wdist:.4g} @ ({best_x:.3g}, {best_y:.3g})"
    else:
        ax.text(0.5, 0.5, "No finite Wasserstein values", ha="center", va="center", transform=ax.transAxes)
        title_suffix = "no finite values"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{subsystem_x} RFF scale")
    ax.set_ylabel(f"{subsystem_y} RFF scale")
    ax.set_title(f"{subsystem_x}-{subsystem_y} 2D scan ({_set_display_label(set_name)}): {title_suffix}")
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return best


def _add_text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.05, 0.92, title, fontsize=18, fontweight="bold", va="top", ha="left")
    if lines:
        ax.text(0.05, 0.84, "\n".join(lines), fontsize=12, va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def write_summary_pdf(plot_entries, pdf_path, args, cross_pairs, summary_title="RFF Scale Sweep Summary"):
    if not plot_entries:
        return None

    category_order = {"intra": 0, "cross": 1, "curve": 2, "cross2d": 3}
    set_order = build_set_order()

    sorted_entries = sorted(
        plot_entries,
        key=lambda entry: (
            category_order.get(entry["category"], 99),
            entry["group_name"],
            set_order.get(entry["set_name"], 99),
        ),
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _add_text_page(
            pdf,
            summary_title,
            [
                f"Event start index: {args.event_index}",
                f"Number of events: {args.num_events}",
                f"RFF base scale: {args.rff_scale}",
                "Curve scales: "
                f"{args.curve_num_scales} log-spaced in [{args.curve_scale_min_mult}, {args.curve_scale_max_mult}] x base",
                f"Intra subsystems: {', '.join(args.subsystems)}",
                f"Cross pairs: {', '.join([f'{a}-{b}' for a, b in cross_pairs])}",
                f"Number of plot pages: {len(sorted_entries)}",
            ],
        )

        current_category = None
        current_group = None

        for entry in sorted_entries:
            if entry["category"] != current_category:
                current_category = entry["category"]
                current_group = None
                category_titles = {
                    "intra": "Intra-Subsystem Comparisons",
                    "cross": "Cross-Subsystem Comparisons",
                    "curve": "Wasserstein-vs-Scale Curves",
                    "cross2d": "Cross-Pair 2D Scale Scans",
                }
                category_title = category_titles.get(current_category, current_category)
                _add_text_page(pdf, category_title, [])

            if entry["group_name"] != current_group:
                current_group = entry["group_name"]
                _add_text_page(
                    pdf,
                    f"Group: {current_group}",
                    [
                        f"Events used: {entry['num_events']}",
                        f"Valid hits summary: {entry['hit_summary']}",
                    ],
                )

            fig = plt.figure(figsize=(11, 8.5))
            grid = fig.add_gridspec(2, 1, height_ratios=[0.12, 0.88], hspace=0.02)
            title_ax = fig.add_subplot(grid[0])
            title_ax.axis("off")
            title_ax.text(0.0, 0.9, entry["title_prefix"], fontsize=12, fontweight="bold", va="top", ha="left")
            title_ax.text(0.0, 0.45, f"Set: {entry['set_name']}", fontsize=10, va="top", ha="left")

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


def collect_optimal_scale_rows(curve_results):
    set_order = build_set_order()
    rows = []

    for set_name in sorted(curve_results, key=lambda name: set_order.get(name, 999)):
        set_curve_data = curve_results[set_name]
        if not set_curve_data:
            continue

        for pair_name, payload in sorted(set_curve_data.items(), key=lambda item: (0 if item[1]["kind"] == "intra" else 1, item[0])):
            scales = payload["scales"]
            wdist = payload["wdist"]
            finite = np.isfinite(wdist)
            if np.any(finite):
                max_idx = int(np.nanargmax(wdist))
                best_scale = f"{float(scales[max_idx]):.6g}"
                best_wdist = f"{float(wdist[max_idx]):.6g}"
            else:
                best_scale = "n/a"
                best_wdist = "n/a"

            rows.append(
                {
                    "set": set_name,
                    "pair": pair_name,
                    "kind": payload["kind"],
                    "scale": best_scale,
                    "wdist": best_wdist,
                }
            )

    return rows


def format_optimal_scale_summary(curve_results, title):
    rows = collect_optimal_scale_rows(curve_results)
    lines = [title]
    if not rows:
        lines.append("  (no valid curves)")
        return "\n".join(lines)

    headers = {"set": "set", "pair": "pair", "kind": "kind", "scale": "best_scale", "wdist": "max_wdist"}
    lines.extend(_format_table_lines(rows=rows, column_order=["set", "pair", "kind", "scale", "wdist"], headers=headers))
    return "\n".join(lines)


def format_cross2d_best_summary(rows, title):
    lines = [title]
    if not rows:
        lines.append("  (no valid 2D scans)")
        return "\n".join(lines)

    set_order = build_set_order()
    sorted_rows = sorted(rows, key=lambda row: (row["pair"], set_order.get(row["set"], 999)))
    headers = {
        "pair": "pair",
        "set": "set",
        "scale_x": "best_scale_x",
        "scale_y": "best_scale_y",
        "wdist": "max_wdist",
    }
    lines.extend(
        _format_table_lines(
            rows=sorted_rows,
            column_order=["pair", "set", "scale_x", "scale_y", "wdist"],
            headers=headers,
        )
    )
    return "\n".join(lines)


def build_cross2d_best_lookup(rows):
    lookup = {}
    for row in rows:
        pair_name = row.get("pair")
        set_name = row.get("set")
        if (not pair_name) or (not set_name):
            continue

        try:
            scale_x = float(row.get("scale_x"))
            scale_y = float(row.get("scale_y"))
        except (TypeError, ValueError):
            continue

        if (not np.isfinite(scale_x)) or (not np.isfinite(scale_y)):
            continue
        if scale_x <= 0.0 or scale_y <= 0.0:
            continue

        lookup.setdefault(pair_name, {})[set_name] = {
            "scale_x": scale_x,
            "scale_y": scale_y,
        }

    return lookup


def format_cross2d_pair_summary(rows, title, pair_name):
    pair_rows = [row for row in rows if row["pair"] == pair_name]
    lines = [title]
    if not pair_rows:
        lines.append("  (no valid 2D scans)")
        return "\n".join(lines)

    set_order = build_set_order()
    sorted_rows = sorted(pair_rows, key=lambda row: set_order.get(row["set"], 999))
    subsystem_x, subsystem_y = pair_name.split("-", maxsplit=1)
    headers = {
        "set": "set",
        "scale_x": f"best_scale_{subsystem_x}",
        "scale_y": f"best_scale_{subsystem_y}",
        "wdist": "max_wdist",
    }
    lines.extend(
        _format_table_lines(
            rows=sorted_rows,
            column_order=["set", "scale_x", "scale_y", "wdist"],
            headers=headers,
        )
    )
    return "\n".join(lines)


def write_text_report(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_cross2d_pair_tables(rows, run_output_root, run_label):
    pair_names = sorted({row["pair"] for row in rows})
    for pair_name in pair_names:
        pair_text = format_cross2d_pair_summary(
            rows=rows,
            title=f"Best cross-pair 2D scan points for {pair_name} [run={run_label}]:",
            pair_name=pair_name,
        )
        pair_path = run_output_root / "tables" / "cross2d_pairs" / f"cross2d_optimal_scales_{pair_name}.txt"
        write_text_report(pair_path, pair_text)
        print(f"[{run_label}] Wrote {pair_path}")


def derive_best_intra_scales_by_subsystem(curve_results, subsystems):
    best_scales = {subsystem: {coord_name: 1.0 for coord_name in BASE_COORD_NAMES} for subsystem in subsystems}
    source_tags = {subsystem: {coord_name: "fallback" for coord_name in BASE_COORD_NAMES} for subsystem in subsystems}

    for coord_name in BASE_COORD_NAMES:
        set_curve_data = curve_results.get(coord_name, {})
        for subsystem in subsystems:
            payload = set_curve_data.get(subsystem)
            if payload is None or payload.get("kind") != "intra":
                continue

            scales = payload["scales"]
            wdist = payload["wdist"]
            finite = np.isfinite(wdist)
            if not np.any(finite):
                continue

            max_idx = int(np.nanargmax(wdist))
            best_scale = float(scales[max_idx])
            if (not np.isfinite(best_scale)) or (best_scale <= 0.0):
                continue

            best_scales[subsystem][coord_name] = best_scale
            source_tags[subsystem][coord_name] = "intra_argmax"

    return best_scales, source_tags


def format_coord_normalization_summary(best_scales, source_tags, title):
    rows = []
    for subsystem in sorted(best_scales):
        for coord_name in BASE_COORD_NAMES:
            rows.append(
                {
                    "subsystem": subsystem,
                    "coord": coord_name,
                    "scale": f"{float(best_scales[subsystem][coord_name]):.6g}",
                    "source": source_tags[subsystem][coord_name],
                }
            )

    lines = [title]
    headers = {
        "subsystem": "subsystem",
        "coord": "coord",
        "scale": "intra_best_scale_div",
        "source": "source",
    }
    lines.extend(_format_table_lines(rows=rows, column_order=["subsystem", "coord", "scale", "source"], headers=headers))
    return "\n".join(lines)


def run_cross_pair_2d_scan(
    cross_event_data,
    cross_pairs,
    field_sets,
    cross2d_scale_values,
    rff_base_matrices,
    dim_per_field,
    run_output_root,
    run_label,
    cross2d_reference_best=None,
    cross2d_max_pairs=5000,
    seed=7,
):
    if cross2d_scale_values is None:
        return [], [], []

    rng = np.random.default_rng(seed + 101)
    num_scales = len(cross2d_scale_values)
    best_rows = []
    written_paths = []
    plot_entries = []

    for subsystem_a, subsystem_b in cross_pairs:
        pair_name = f"{subsystem_a}-{subsystem_b}"
        pair_events = cross_event_data.get(pair_name, [])
        if not pair_events:
            print(f"[{run_label}] Skipping 2D scan for {pair_name}: no usable events found")
            continue

        print(f"[{run_label}] 2D scan {pair_name}: using {len(pair_events)} events")

        for set_name, set_cfg in field_sets.items():
            metric_field = list(set_cfg["fields"].keys())[0]
            coord_name = list(set_cfg["fields"].values())[0]
            wdist_grid = np.full((num_scales, num_scales), np.nan, dtype=np.float64)

            ref_scale_x = None
            ref_scale_y = None
            if cross2d_reference_best is not None:
                ref_payload = cross2d_reference_best.get(pair_name, {}).get(set_name)
                if ref_payload is not None:
                    maybe_x = float(ref_payload.get("scale_x", np.nan))
                    maybe_y = float(ref_payload.get("scale_y", np.nan))
                    if np.isfinite(maybe_x) and maybe_x > 0.0:
                        ref_scale_x = maybe_x
                    if np.isfinite(maybe_y) and maybe_y > 0.0:
                        ref_scale_y = maybe_y

            event_payloads = []
            for event_data in pair_events:
                coords_a = event_data["coords_a"]
                coords_b = event_data["coords_b"]
                same_pairs = event_data["same_pairs"]
                not_same_pairs = event_data["not_same_pairs"]

                if ref_scale_x is not None:
                    coords_a = dict(coords_a)
                    coords_a[coord_name] = coords_a[coord_name] / ref_scale_x
                if ref_scale_y is not None:
                    coords_b = dict(coords_b)
                    coords_b[coord_name] = coords_b[coord_name] / ref_scale_y

                if cross2d_max_pairs > 0:
                    same_pairs = subsample_pairs(same_pairs, cross2d_max_pairs, rng)
                    not_same_pairs = subsample_pairs(not_same_pairs, cross2d_max_pairs, rng)

                if same_pairs.numel() == 0 or not_same_pairs.numel() == 0:
                    continue

                event_payloads.append(
                    {
                        "coords_a": coords_a,
                        "coords_b": coords_b,
                        "same_pairs": same_pairs,
                        "not_same_pairs": not_same_pairs,
                    }
                )

            if not event_payloads:
                out_path = run_output_root / "cross_2d" / f"wasserstein_2d_{pair_name}_{set_name}.png"
                best = plot_cross_pair_wasserstein_2d(
                    wdist_grid=wdist_grid,
                    scale_values_x=cross2d_scale_values,
                    scale_values_y=cross2d_scale_values,
                    out_path=out_path,
                    subsystem_x=subsystem_a,
                    subsystem_y=subsystem_b,
                    set_name=set_name,
                )
                written_paths.append(out_path)
                plot_entries.append(
                    {
                        "path": out_path,
                        "category": "cross2d",
                        "group_name": pair_name,
                        "set_name": set_name,
                        "title_prefix": f"Cross 2D scale scan ({pair_name}) [{run_label}]",
                        "num_events": len(pair_events),
                        "hit_summary": "cross-system same vs non-same",
                    }
                )
                best_rows.append(
                    {
                        "pair": pair_name,
                        "set": set_name,
                        "scale_x": "n/a",
                        "scale_y": "n/a",
                        "wdist": "n/a",
                    }
                )
                continue

            # Collect per-grid-cell chunks once, and compute pairwise dot-products for all scale
            # combinations using einsum to avoid repeated gather+multiply in nested loops.
            same_chunks_grid = [[[] for _ in range(num_scales)] for _ in range(num_scales)]
            not_chunks_grid = [[[] for _ in range(num_scales)] for _ in range(num_scales)]

            for event_payload in event_payloads:
                coords_a = event_payload["coords_a"]
                coords_b = event_payload["coords_b"]
                same_pairs = event_payload["same_pairs"]
                not_same_pairs = event_payload["not_same_pairs"]

                enc_cache_a = []
                enc_cache_b = []
                event_enc_a = []
                event_enc_b = []
                for scale_value in cross2d_scale_values:
                    enc_a = encode_field_set_rff(
                        coords=coords_a,
                        field_map=set_cfg["fields"],
                        base_matrices=rff_base_matrices,
                        scale=float(scale_value),
                        dim_per_field=dim_per_field,
                    )[metric_field]
                    enc_b = encode_field_set_rff(
                        coords=coords_b,
                        field_map=set_cfg["fields"],
                        base_matrices=rff_base_matrices,
                        scale=float(scale_value),
                        dim_per_field=dim_per_field,
                    )[metric_field]
                    event_enc_a.append(enc_a)
                    event_enc_b.append(enc_b)
                enc_cache_a = torch.stack(event_enc_a, dim=0)  # [sx, hits_a, dim]
                enc_cache_b = torch.stack(event_enc_b, dim=0)  # [sy, hits_b, dim]

                same_left = enc_cache_a[:, same_pairs[:, 0], :]  # [sx, pairs, dim]
                same_right = enc_cache_b[:, same_pairs[:, 1], :]  # [sy, pairs, dim]
                same_dot = torch.einsum("spd,tpd->stp", same_left, same_right)  # [sx, sy, pairs]

                not_left = enc_cache_a[:, not_same_pairs[:, 0], :]
                not_right = enc_cache_b[:, not_same_pairs[:, 1], :]
                not_dot = torch.einsum("spd,tpd->stp", not_left, not_right)

                for x_idx in range(num_scales):
                    for y_idx in range(num_scales):
                        same_chunks_grid[y_idx][x_idx].append(same_dot[x_idx, y_idx])
                        not_chunks_grid[y_idx][x_idx].append(not_dot[x_idx, y_idx])

            for y_idx in range(num_scales):
                for x_idx in range(num_scales):
                    if same_chunks_grid[y_idx][x_idx] and not_chunks_grid[y_idx][x_idx]:
                        same_np = torch.cat(same_chunks_grid[y_idx][x_idx], dim=0).detach().cpu().numpy()
                        not_np = torch.cat(not_chunks_grid[y_idx][x_idx], dim=0).detach().cpu().numpy()
                        if same_np.size and not_np.size:
                            wdist_grid[y_idx, x_idx] = float(wasserstein_distance(same_np, not_np))

            out_path = run_output_root / "cross_2d" / f"wasserstein_2d_{pair_name}_{set_name}.png"
            best = plot_cross_pair_wasserstein_2d(
                wdist_grid=wdist_grid,
                scale_values_x=cross2d_scale_values,
                scale_values_y=cross2d_scale_values,
                out_path=out_path,
                subsystem_x=subsystem_a,
                subsystem_y=subsystem_b,
                set_name=set_name,
            )
            written_paths.append(out_path)
            plot_entries.append(
                {
                    "path": out_path,
                    "category": "cross2d",
                    "group_name": pair_name,
                    "set_name": set_name,
                    "title_prefix": f"Cross 2D scale scan ({pair_name}) [{run_label}]",
                    "num_events": len(pair_events),
                    "hit_summary": "cross-system same vs non-same",
                }
            )

            if best is None:
                best_rows.append(
                    {
                        "pair": pair_name,
                        "set": set_name,
                        "scale_x": "n/a",
                        "scale_y": "n/a",
                        "wdist": "n/a",
                    }
                )
            else:
                best_rows.append(
                    {
                        "pair": pair_name,
                        "set": set_name,
                        "scale_x": f"{best['best_scale_x']:.6g}",
                        "scale_y": f"{best['best_scale_y']:.6g}",
                        "wdist": f"{best['best_wdist']:.6g}",
                    }
                )

    for path in written_paths:
        print(f"[{run_label}] Wrote {path}")

    return best_rows, written_paths, plot_entries


def run_curve_sweep(
    events,
    args,
    cross_pairs,
    field_sets,
    curve_scale_values,
    rff_base_matrices,
    run_output_root,
    run_label,
    coord_scale_by_subsystem=None,
    cross2d_scale_values=None,
    cross2d_reference_best=None,
):
    rng = np.random.default_rng(args.seed)
    run_output_root.mkdir(parents=True, exist_ok=True)
    written_paths = []
    plot_entries = []
    curve_results = {set_name: {} for set_name in field_sets}
    cross_event_data = {}

    num_curve_scales = len(curve_scale_values)
    curve_scale_tensor = torch.as_tensor(curve_scale_values, dtype=torch.float32)

    def init_curve_buffers():
        buffers = {}
        for set_name in field_sets:
            buffers[set_name] = [[] for _ in range(num_curve_scales)]
        return buffers

    print("")
    print(f"=== Sweep Run: {run_label} ===")

    # Intra-subsystem comparisons
    for subsystem in args.subsystems:
        per_curve_scores_same = init_curve_buffers()
        per_curve_scores_not_same = init_curve_buffers()
        used_events = 0
        skipped_events = 0

        for inputs, targets in events:
            subsystem_data, _reason = prepare_subsystem_data(
                inputs=inputs,
                targets=targets,
                subsystem=subsystem,
                max_same_pairs=args.max_same_pairs,
                rng=rng,
                coord_scale_by_subsystem=coord_scale_by_subsystem,
            )
            if subsystem_data is None:
                skipped_events += 1
                continue

            used_events += 1

            for set_name, set_cfg in field_sets.items():
                coord_name = list(set_cfg["fields"].values())[0]
                embeddings_multi = random_fourier_pos_enc_multi(
                    xs=subsystem_data["coords"][coord_name],
                    base_matrix=rff_base_matrices[coord_name],
                    scales=curve_scale_tensor,
                    dim_per_field=args.dim_per_field,
                )
                same_multi = pair_dot_products_multi(embeddings_multi, subsystem_data["same_pairs"])
                not_same_multi = pair_dot_products_multi(embeddings_multi, subsystem_data["not_same_pairs"])

                for scale_idx in range(num_curve_scales):
                    per_curve_scores_same[set_name][scale_idx].append(same_multi[scale_idx])
                    per_curve_scores_not_same[set_name][scale_idx].append(not_same_multi[scale_idx])

        if used_events == 0:
            print(f"[{run_label}] Skipping {subsystem}: no usable events found")
            continue

        for set_name in field_sets:
            wdist_curve = np.full(len(curve_scale_values), np.nan, dtype=np.float64)
            for scale_idx in range(num_curve_scales):
                chunks_same = per_curve_scores_same[set_name][scale_idx]
                chunks_not = per_curve_scores_not_same[set_name][scale_idx]
                if chunks_same and chunks_not:
                    same = torch.cat(chunks_same, dim=0).detach().cpu().numpy()
                    not_same = torch.cat(chunks_not, dim=0).detach().cpu().numpy()
                    if same.size and not_same.size:
                        wdist_curve[scale_idx] = float(wasserstein_distance(same, not_same))

            curve_results[set_name][subsystem] = {
                "scales": curve_scale_values.copy(),
                "wdist": wdist_curve,
                "kind": "intra",
            }

        print(f"[{run_label}] {subsystem}: used {used_events}/{len(events)} events, skipped {skipped_events}")

    # Cross-subsystem comparisons
    for subsystem_a, subsystem_b in cross_pairs:
        pair_name = f"{subsystem_a}-{subsystem_b}"
        cross_event_data[pair_name] = []
        per_curve_scores_same = init_curve_buffers()
        per_curve_scores_not_same = init_curve_buffers()
        used_events = 0
        skipped_events = 0

        for inputs, targets in events:
            cross_data, _reason = prepare_cross_subsystem_data(
                inputs=inputs,
                targets=targets,
                subsystem_a=subsystem_a,
                subsystem_b=subsystem_b,
                max_same_pairs=args.max_same_pairs,
                rng=rng,
                coord_scale_by_subsystem=coord_scale_by_subsystem,
            )
            if cross_data is None:
                skipped_events += 1
                continue

            used_events += 1
            cross_data_for_2d = cross_data
            if (cross2d_reference_best is not None) and (coord_scale_by_subsystem is not None):
                coords_a_unscaled = undo_coord_scaling_for_subsystem(
                    coords=cross_data["coords_a"],
                    subsystem=subsystem_a,
                    coord_scale_by_subsystem=coord_scale_by_subsystem,
                )
                coords_b_unscaled = undo_coord_scaling_for_subsystem(
                    coords=cross_data["coords_b"],
                    subsystem=subsystem_b,
                    coord_scale_by_subsystem=coord_scale_by_subsystem,
                )
                cross_data_for_2d = {
                    "num_hits_a": cross_data["num_hits_a"],
                    "num_hits_b": cross_data["num_hits_b"],
                    "coords_a": coords_a_unscaled,
                    "coords_b": coords_b_unscaled,
                    "same_pairs": cross_data["same_pairs"],
                    "not_same_pairs": cross_data["not_same_pairs"],
                }

            cross_event_data[pair_name].append(cross_data_for_2d)

            for set_name, set_cfg in field_sets.items():
                coord_name = list(set_cfg["fields"].values())[0]
                embeddings_a_multi = random_fourier_pos_enc_multi(
                    xs=cross_data["coords_a"][coord_name],
                    base_matrix=rff_base_matrices[coord_name],
                    scales=curve_scale_tensor,
                    dim_per_field=args.dim_per_field,
                )
                embeddings_b_multi = random_fourier_pos_enc_multi(
                    xs=cross_data["coords_b"][coord_name],
                    base_matrix=rff_base_matrices[coord_name],
                    scales=curve_scale_tensor,
                    dim_per_field=args.dim_per_field,
                )

                same_multi = pair_dot_products_multi(
                    embeddings_left=embeddings_a_multi,
                    pairs=cross_data["same_pairs"],
                    embeddings_right=embeddings_b_multi,
                )
                not_same_multi = pair_dot_products_multi(
                    embeddings_left=embeddings_a_multi,
                    pairs=cross_data["not_same_pairs"],
                    embeddings_right=embeddings_b_multi,
                )

                for scale_idx in range(num_curve_scales):
                    per_curve_scores_same[set_name][scale_idx].append(same_multi[scale_idx])
                    per_curve_scores_not_same[set_name][scale_idx].append(not_same_multi[scale_idx])

        if used_events == 0:
            print(f"[{run_label}] Skipping cross pair {pair_name}: no usable events found")
            continue

        for set_name in field_sets:
            wdist_curve = np.full(len(curve_scale_values), np.nan, dtype=np.float64)
            for scale_idx in range(num_curve_scales):
                chunks_same = per_curve_scores_same[set_name][scale_idx]
                chunks_not = per_curve_scores_not_same[set_name][scale_idx]
                if chunks_same and chunks_not:
                    same = torch.cat(chunks_same, dim=0).detach().cpu().numpy()
                    not_same = torch.cat(chunks_not, dim=0).detach().cpu().numpy()
                    if same.size and not_same.size:
                        wdist_curve[scale_idx] = float(wasserstein_distance(same, not_same))

            curve_results[set_name][pair_name] = {
                "scales": curve_scale_values.copy(),
                "wdist": wdist_curve,
                "kind": "cross",
            }

        print(f"[{run_label}] {pair_name}: used {used_events}/{len(events)} events, skipped {skipped_events}")

    for set_name, set_curve_data in curve_results.items():
        if not set_curve_data:
            continue

        out_path = run_output_root / "summary" / f"wasserstein_vs_scale_{set_name}.png"
        plot_wasserstein_vs_scale(curve_data=set_curve_data, out_path=out_path, set_name=set_name)
        written_paths.append(out_path)
        plot_entries.append(
            {
                "path": out_path,
                "category": "curve",
                "group_name": "all-pairs",
                "set_name": set_name,
                "title_prefix": f"Wasserstein-vs-scale summary ({set_name}) [{run_label}]",
                "num_events": args.num_events,
                "hit_summary": "all pair types",
            }
        )

    for path in written_paths:
        print(f"[{run_label}] Wrote {path}")

    cross2d_rows, cross2d_written_paths, cross2d_plot_entries = run_cross_pair_2d_scan(
        cross_event_data=cross_event_data,
        cross_pairs=cross_pairs,
        field_sets=field_sets,
        cross2d_scale_values=cross2d_scale_values,
        rff_base_matrices=rff_base_matrices,
        dim_per_field=args.dim_per_field,
        run_output_root=run_output_root,
        run_label=run_label,
        cross2d_reference_best=cross2d_reference_best,
        cross2d_max_pairs=args.cross2d_max_pairs,
        seed=args.seed,
    )
    written_paths.extend(cross2d_written_paths)
    plot_entries.extend(cross2d_plot_entries)

    if cross2d_scale_values is not None:
        cross2d_summary_text = format_cross2d_best_summary(
            rows=cross2d_rows,
            title=f"Best cross-pair 2D scan points [run={run_label}]:",
        )
        print("")
        print(cross2d_summary_text)
        cross2d_summary_path = run_output_root / "tables" / "cross2d_optimal_scales.txt"
        write_text_report(cross2d_summary_path, cross2d_summary_text)
        print(f"[{run_label}] Wrote {cross2d_summary_path}")
        write_cross2d_pair_tables(rows=cross2d_rows, run_output_root=run_output_root, run_label=run_label)

    if not args.skip_pdf:
        pdf_name = args.pdf_path.name if args.pdf_path is not None else "summary.pdf"
        pdf_path = run_output_root / pdf_name
        written_pdf = write_summary_pdf(
            plot_entries=plot_entries,
            pdf_path=pdf_path,
            args=args,
            cross_pairs=cross_pairs,
            summary_title=f"RFF Scale Sweep Summary ({run_label})",
        )
        if written_pdf is not None:
            print(f"[{run_label}] Wrote {written_pdf}")

    return {
        "curve_results": curve_results,
        "cross2d_rows": cross2d_rows,
    }


def main():
    args = parse_args()
    if args.num_events <= 0:
        raise ValueError("--num-events must be > 0")
    if args.rff_scale <= 0:
        raise ValueError("--rff-scale must be > 0")
    if args.curve_num_scales <= 0:
        raise ValueError("--curve-num-scales must be > 0")
    if args.curve_scale_min_mult <= 0 or args.curve_scale_max_mult <= 0:
        raise ValueError("--curve-scale-min-mult and --curve-scale-max-mult must be > 0")
    if args.curve_scale_min_mult > args.curve_scale_max_mult:
        raise ValueError("--curve-scale-min-mult must be <= --curve-scale-max-mult")
    if not args.skip_cross2d:
        if args.cross2d_num_scales <= 0:
            raise ValueError("--cross2d-num-scales must be > 0")
        if args.cross2d_scale_min_mult <= 0 or args.cross2d_scale_max_mult <= 0:
            raise ValueError("--cross2d-scale-min-mult and --cross2d-scale-max-mult must be > 0")
        if args.cross2d_scale_min_mult > args.cross2d_scale_max_mult:
            raise ValueError("--cross2d-scale-min-mult must be <= --cross2d-scale-max-mult")

    events = load_events(args.config, args.event_index, args.num_events, args.num_workers)
    cross_pairs = parse_cross_subsystem_pairs(args.cross_subsystem_pairs)

    field_sets = build_field_sets()

    curve_scale_multipliers = np.logspace(
        np.log10(args.curve_scale_min_mult),
        np.log10(args.curve_scale_max_mult),
        args.curve_num_scales,
    )
    curve_scale_values = args.rff_scale * curve_scale_multipliers
    cross2d_scale_values = None
    if not args.skip_cross2d:
        cross2d_scale_multipliers = np.logspace(
            np.log10(args.cross2d_scale_min_mult),
            np.log10(args.cross2d_scale_max_mult),
            args.cross2d_num_scales,
        )
        cross2d_scale_values = args.rff_scale * cross2d_scale_multipliers

    rff_base_matrices = init_rff_base_matrices(BASE_COORD_NAMES, args.dim_per_field, args.seed)

    output_root = args.out_dir / args.out_prefix
    output_root.mkdir(parents=True, exist_ok=True)

    original_output_root = output_root / "original"
    scaled_output_root = output_root / "scaled"

    original_run_outputs = run_curve_sweep(
        events=events,
        args=args,
        cross_pairs=cross_pairs,
        field_sets=field_sets,
        curve_scale_values=curve_scale_values,
        rff_base_matrices=rff_base_matrices,
        run_output_root=original_output_root,
        run_label="original",
        coord_scale_by_subsystem=None,
        cross2d_scale_values=cross2d_scale_values,
        cross2d_reference_best=None,
    )
    curve_results_original = original_run_outputs["curve_results"]
    original_cross2d_rows = original_run_outputs["cross2d_rows"]
    cross2d_reference_best = build_cross2d_best_lookup(original_cross2d_rows)

    original_summary_text = format_optimal_scale_summary(
        curve_results=curve_results_original,
        title="Optimal RFF scales by variable and pair type (argmax Wasserstein distance) [original coords]:",
    )
    print("")
    print(original_summary_text)
    original_summary_path = original_output_root / "tables" / "optimal_scales.txt"
    write_text_report(original_summary_path, original_summary_text)
    print(f"[original] Wrote {original_summary_path}")

    best_intra_scales, source_tags = derive_best_intra_scales_by_subsystem(
        curve_results=curve_results_original,
        subsystems=args.subsystems,
    )
    normalization_text = format_coord_normalization_summary(
        best_scales=best_intra_scales,
        source_tags=source_tags,
        title=(
            "Coordinate normalization scales used for scaled run "
            "(coord_scaled_subsystem = coord / intra_best_scale(subsystem, coord)):"
        ),
    )
    print("")
    print(normalization_text)
    normalization_path = scaled_output_root / "tables" / "coord_normalization_scales.txt"
    write_text_report(normalization_path, normalization_text)
    print(f"[scaled] Wrote {normalization_path}")

    scaled_run_outputs = run_curve_sweep(
        events=events,
        args=args,
        cross_pairs=cross_pairs,
        field_sets=field_sets,
        curve_scale_values=curve_scale_values,
        rff_base_matrices=rff_base_matrices,
        run_output_root=scaled_output_root,
        run_label="scaled",
        coord_scale_by_subsystem=best_intra_scales,
        cross2d_scale_values=cross2d_scale_values,
        cross2d_reference_best=cross2d_reference_best,
    )
    curve_results_scaled = scaled_run_outputs["curve_results"]

    scaled_summary_text = format_optimal_scale_summary(
        curve_results=curve_results_scaled,
        title="Optimal RFF scales by variable and pair type (argmax Wasserstein distance) [scaled coords]:",
    )
    print("")
    print(scaled_summary_text)
    scaled_summary_path = scaled_output_root / "tables" / "optimal_scales.txt"
    write_text_report(scaled_summary_path, scaled_summary_text)
    print(f"[scaled] Wrote {scaled_summary_path}")

    print(f"Event index: {args.event_index}, num_events: {args.num_events}")
    print(f"Requested subsystems: {', '.join(args.subsystems)}")
    print(f"Requested cross pairs: {', '.join([f'{a}-{b}' for a, b in cross_pairs])}")
    print(
        "Curve scales: "
        f"{args.curve_num_scales} log-spaced in "
        f"[{args.curve_scale_min_mult}, {args.curve_scale_max_mult}] x base scale"
    )
    if args.skip_cross2d:
        print("Cross 2D scan: skipped")
    else:
        print(
            "Cross 2D scales: "
            f"{args.cross2d_num_scales} log-spaced in "
            f"[{args.cross2d_scale_min_mult}, {args.cross2d_scale_max_mult}] x base scale"
        )
        if args.cross2d_max_pairs > 0:
            print(f"Cross 2D pair cap/event: {args.cross2d_max_pairs}")
        else:
            print("Cross 2D pair cap/event: all")
    print(f"Output roots: {original_output_root} (original), {scaled_output_root} (scaled)")


if __name__ == "__main__":
    main()
