#!/usr/bin/env python3

import argparse
import re
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

DEFAULT_DIM_VALUES = [2**k for k in range(1, 10)]  # 2..512


def build_field_sets():
    return {coord_name: {"coord": coord_name} for coord_name in BASE_COORD_NAMES}


def build_set_order():
    return {name: idx for idx, name in enumerate(BASE_COORD_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scale coordinates with precomputed intra optimal RFF scales, then scan Wasserstein distance "
            "between same-particle and non-same-particle hit-pair normalized dot products versus embedding dimension."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("src/hepattn/experiments/cld/configs/unified.yaml"))
    parser.add_argument("--scales-table", type=Path, default=Path("src/hepattn/experiments/cld/plots/data/posenc_rff_scale_sweep/original/tables/optimal_scales.txt"))
    parser.add_argument("--event-index", type=int, default=0, help="Zero-based start index in the test dataloader.")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events to aggregate for statistics.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rff-scale", type=float, default=1.0, help="Fixed RFF scale used during dimension scan.")
    parser.add_argument(
        "--normalize-dot-by-dim",
        action="store_true",
        default=True,
        help="Normalize hit-pair dot products by embedding dimension (enabled by default).",
    )
    parser.add_argument(
        "--no-normalize-dot-by-dim",
        dest="normalize_dot_by_dim",
        action="store_false",
        help="Disable dot-product normalization by embedding dimension.",
    )
    parser.add_argument(
        "--dim-values",
        type=int,
        nargs="+",
        default=DEFAULT_DIM_VALUES,
        help="Embedding dimensions to scan (must be even), e.g. 2 4 8 ... 512.",
    )
    parser.add_argument("--max-same-pairs", type=int, default=25000, help="Subsample same-particle pairs to this cap per event.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("src/hepattn/experiments/cld/plots/data"))
    parser.add_argument("--out-prefix", type=str, default="posenc_rff_dim_sweep")
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
    parser.add_argument(
        "--include-cross",
        action="store_true",
        help="If set, include cross-subsystem pair curves. By default only intra-subsystem curves are produced.",
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


def load_intra_scales_table(scales_table_path, subsystems):
    if not scales_table_path.exists():
        raise FileNotFoundError(f"Scales table not found: {scales_table_path}")

    scales = {subsystem: {coord_name: 1.0 for coord_name in BASE_COORD_NAMES} for subsystem in subsystems}
    subsystems_set = set(subsystems)
    loaded = 0

    for line in scales_table_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("optimal"):
            continue
        if line.startswith("set"):
            continue
        if re.fullmatch(r"-+", line.replace(" ", "")):
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue

        set_name, pair_name, kind, best_scale = parts[0], parts[1], parts[2], parts[3]
        if kind != "intra":
            continue
        if pair_name not in subsystems_set:
            continue
        if set_name not in BASE_COORD_NAMES:
            continue

        try:
            scale_value = float(best_scale)
        except ValueError:
            continue
        if (not np.isfinite(scale_value)) or scale_value <= 0.0:
            continue

        scales[pair_name][set_name] = scale_value
        loaded += 1

    if loaded == 0:
        raise RuntimeError(f"No intra scale rows parsed from {scales_table_path}")

    return scales


def format_loaded_scales_summary(scales, title):
    rows = []
    for subsystem in sorted(scales):
        for coord_name in BASE_COORD_NAMES:
            rows.append(
                {
                    "subsystem": subsystem,
                    "coord": coord_name,
                    "scale": f"{float(scales[subsystem][coord_name]):.6g}",
                }
            )

    widths = {
        "subsystem": max(len("subsystem"), max(len(row["subsystem"]) for row in rows)),
        "coord": max(len("coord"), max(len(row["coord"]) for row in rows)),
        "scale": max(len("intra_best_scale"), max(len(row["scale"]) for row in rows)),
    }

    lines = [title]
    lines.append("  " + "  ".join(["subsystem".ljust(widths["subsystem"]), "coord".ljust(widths["coord"]), "intra_best_scale".ljust(widths["scale"])]))
    lines.append("  " + "  ".join(["-" * widths["subsystem"], "-" * widths["coord"], "-" * widths["scale"]]))
    for row in rows:
        lines.append("  " + "  ".join([row["subsystem"].ljust(widths["subsystem"]), row["coord"].ljust(widths["coord"]), row["scale"].ljust(widths["scale"])]))
    return "\n".join(lines)


def apply_coord_scaling_for_subsystem(coords, subsystem, coord_scale_by_subsystem):
    subsystem_scales = coord_scale_by_subsystem.get(subsystem, {})
    scaled_coords = {}
    for coord_name, values in coords.items():
        scale_value = float(subsystem_scales.get(coord_name, 1.0))
        if (not np.isfinite(scale_value)) or (scale_value <= 0.0):
            scale_value = 1.0
        scaled_coords[coord_name] = values / scale_value
    return scaled_coords


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


def extract_subsystem_view(inputs, targets, subsystem, coord_scale_by_subsystem):
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
    coords = apply_coord_scaling_for_subsystem(coords=coords, subsystem=subsystem, coord_scale_by_subsystem=coord_scale_by_subsystem)
    return {
        "num_hits": num_hits,
        "coords": coords,
        "particle_hit_valid": particle_hit_valid,
    }, None


def prepare_subsystem_data(inputs, targets, subsystem, max_same_pairs, rng, coord_scale_by_subsystem):
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
    return torch.unique(torch.cat(sampled_pairs, dim=0), dim=0)


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


def prepare_cross_subsystem_data(inputs, targets, subsystem_a, subsystem_b, max_same_pairs, rng, coord_scale_by_subsystem):
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
    not_same_pairs = sample_not_same_cross_pairs(hit_to_particle_a, hit_to_particle_b, same_pairs.shape[0], rng)
    if not_same_pairs.shape[0] == 0:
        return None, "failed to sample non-same-particle cross-system hit pairs"

    return {
        "coords_a": subsystem_a_view["coords"],
        "coords_b": subsystem_b_view["coords"],
        "same_pairs": same_pairs,
        "not_same_pairs": not_same_pairs,
    }, None


def init_rff_base_matrices(coord_names, max_half_dim, seed):
    if max_half_dim <= 0:
        raise ValueError("Maximum half-dimension must be > 0")
    gen = torch.Generator()
    gen.manual_seed(seed)
    matrices = {}
    for coord_name in sorted(coord_names):
        matrices[coord_name] = torch.randn((1, max_half_dim), generator=gen)
    return matrices


def random_fourier_pos_enc_trig(xs, base_matrix, scale, max_half_dim):
    if max_half_dim <= 0:
        raise ValueError(f"Maximum half-dimension must be > 0, got {max_half_dim}")

    matrix = ((1.0 / scale) * base_matrix[:, :max_half_dim]).to(device=xs.device, dtype=xs.dtype)
    proj = (2.0 * torch.pi * xs.unsqueeze(-1)) * matrix
    return torch.sin(proj), torch.cos(proj)


def pair_dot_products_multi_from_trig(
    sin_left,
    cos_left,
    pairs,
    half_dim_indices,
    dim_values=None,
    normalize_by_dim=True,
    sin_right=None,
    cos_right=None,
):
    num_dims = int(half_dim_indices.shape[0])
    if pairs.numel() == 0:
        return torch.empty((num_dims, 0), dtype=sin_left.dtype, device=sin_left.device)

    if sin_right is None:
        sin_right = sin_left
    if cos_right is None:
        cos_right = cos_left

    pairs = pairs.to(device=sin_left.device)
    dim_idx = half_dim_indices.to(device=sin_left.device)

    left_sin = sin_left[pairs[:, 0]]
    right_sin = sin_right[pairs[:, 1]]
    left_cos = cos_left[pairs[:, 0]]
    right_cos = cos_right[pairs[:, 1]]

    per_frequency_dot = (left_sin * right_sin) + (left_cos * right_cos)
    cumulative_dot = torch.cumsum(per_frequency_dot, dim=-1)
    selected = cumulative_dot[:, dim_idx].T.contiguous()

    if normalize_by_dim:
        if dim_values is None:
            raise ValueError("dim_values must be provided when normalize_by_dim=True")
        dim_values_t = torch.as_tensor(dim_values, dtype=selected.dtype, device=selected.device)
        selected = selected / dim_values_t[:, None]

    return selected


def plot_wasserstein_vs_dim(curve_data, out_path, set_name):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    dims_ref = None

    def line_sort_key(item):
        pair_name, payload = item
        return (0 if payload["kind"] == "intra" else 1, pair_name)

    for pair_name, payload in sorted(curve_data.items(), key=line_sort_key):
        dims = payload["dims"]
        wdist = payload["wdist"]
        dims_ref = dims
        finite = np.isfinite(wdist)
        if not np.any(finite):
            continue
        max_idx = int(np.nanargmax(wdist))
        best_dim = int(dims[max_idx])
        linestyle = "-" if payload["kind"] == "intra" else "--"
        marker = "o" if payload["kind"] == "intra" else "s"
        ax.plot(
            dims,
            wdist,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.4,
            markersize=3.5,
            label=f"{pair_name} (max@{best_dim})",
        )

    ax.set_xscale("log", base=2)
    if dims_ref is not None:
        ax.set_xticks(dims_ref)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("RFF embedding dimension")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title(f"Wasserstein vs RFF Dimension ({BASE_COORD_LABELS.get(set_name, set_name)})")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def format_optimal_dim_summary(curve_results, title):
    set_order = build_set_order()
    rows = []

    for set_name in sorted(curve_results, key=lambda s: set_order.get(s, 999)):
        for pair_name, payload in sorted(curve_results[set_name].items(), key=lambda item: (0 if item[1]["kind"] == "intra" else 1, item[0])):
            dims = payload["dims"]
            wdist = payload["wdist"]
            finite = np.isfinite(wdist)
            if np.any(finite):
                max_idx = int(np.nanargmax(wdist))
                best_dim = str(int(dims[max_idx]))
                best_wdist = f"{float(wdist[max_idx]):.6g}"
            else:
                best_dim = "n/a"
                best_wdist = "n/a"
            rows.append(
                {
                    "set": set_name,
                    "pair": pair_name,
                    "kind": payload["kind"],
                    "dim": best_dim,
                    "wdist": best_wdist,
                }
            )

    if not rows:
        return title + "\n  (no valid curves)"

    widths = {
        "set": max(len("set"), max(len(r["set"]) for r in rows)),
        "pair": max(len("pair"), max(len(r["pair"]) for r in rows)),
        "kind": max(len("kind"), max(len(r["kind"]) for r in rows)),
        "dim": max(len("best_dim"), max(len(r["dim"]) for r in rows)),
        "wdist": max(len("max_wdist"), max(len(r["wdist"]) for r in rows)),
    }

    lines = [title]
    lines.append("  " + "  ".join(["set".ljust(widths["set"]), "pair".ljust(widths["pair"]), "kind".ljust(widths["kind"]), "best_dim".ljust(widths["dim"]), "max_wdist".ljust(widths["wdist"])]))
    lines.append("  " + "  ".join(["-" * widths["set"], "-" * widths["pair"], "-" * widths["kind"], "-" * widths["dim"], "-" * widths["wdist"]]))
    for row in rows:
        lines.append("  " + "  ".join([row["set"].ljust(widths["set"]), row["pair"].ljust(widths["pair"]), row["kind"].ljust(widths["kind"]), row["dim"].ljust(widths["dim"]), row["wdist"].ljust(widths["wdist"])]))
    return "\n".join(lines)


def write_text_report(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def _add_text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.05, 0.92, title, fontsize=18, fontweight="bold", va="top", ha="left")
    if lines:
        ax.text(0.05, 0.84, "\n".join(lines), fontsize=12, va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def write_summary_pdf(plot_entries, pdf_path, args, cross_pairs):
    if not plot_entries:
        return None

    set_order = build_set_order()
    sorted_entries = sorted(plot_entries, key=lambda entry: (entry["group_name"], set_order.get(entry["set_name"], 999)))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _add_text_page(
            pdf,
            "RFF Dimension Sweep Summary",
            [
                f"Event start index: {args.event_index}",
                f"Number of events: {args.num_events}",
                f"Fixed RFF scale: {args.rff_scale}",
                f"Dimensions: {', '.join(str(d) for d in args.dim_values)}",
                f"Intra subsystems: {', '.join(args.subsystems)}",
                (
                    f"Cross pairs: {', '.join([f'{a}-{b}' for a, b in cross_pairs])}"
                    if cross_pairs
                    else "Cross pairs: skipped"
                ),
                f"Dot normalization: {'enabled (divide by dim)' if args.normalize_dot_by_dim else 'disabled'}",
                f"Number of plot pages: {len(sorted_entries)}",
            ],
        )

        for entry in sorted_entries:
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


def run_dim_sweep(events, args, cross_pairs, field_sets, dims, rff_base_matrices, coord_scale_by_subsystem):
    rng = np.random.default_rng(args.seed)
    num_dims = len(dims)
    half_dims = (dims // 2).astype(np.int64)
    max_half_dim = int(half_dims.max())
    half_dim_indices = torch.as_tensor(half_dims - 1, dtype=torch.long)
    dim_values = np.asarray(dims, dtype=np.float32)
    curve_results = {set_name: {} for set_name in field_sets}

    def init_buffers():
        return {set_name: [[] for _ in range(num_dims)] for set_name in field_sets}

    # Intra
    for subsystem in args.subsystems:
        scores_same = init_buffers()
        scores_not_same = init_buffers()
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
                coord_name = set_cfg["coord"]
                sin_vals, cos_vals = random_fourier_pos_enc_trig(
                    xs=subsystem_data["coords"][coord_name],
                    base_matrix=rff_base_matrices[coord_name],
                    scale=args.rff_scale,
                    max_half_dim=max_half_dim,
                )
                same_multi = pair_dot_products_multi_from_trig(
                    sin_left=sin_vals,
                    cos_left=cos_vals,
                    pairs=subsystem_data["same_pairs"],
                    half_dim_indices=half_dim_indices,
                    dim_values=dim_values,
                    normalize_by_dim=args.normalize_dot_by_dim,
                )
                not_same_multi = pair_dot_products_multi_from_trig(
                    sin_left=sin_vals,
                    cos_left=cos_vals,
                    pairs=subsystem_data["not_same_pairs"],
                    half_dim_indices=half_dim_indices,
                    dim_values=dim_values,
                    normalize_by_dim=args.normalize_dot_by_dim,
                )
                for dim_idx in range(num_dims):
                    scores_same[set_name][dim_idx].append(same_multi[dim_idx])
                    scores_not_same[set_name][dim_idx].append(not_same_multi[dim_idx])

        if used_events == 0:
            print(f"Skipping {subsystem}: no usable events found")
            continue

        for set_name in field_sets:
            wdist_curve = np.full(num_dims, np.nan, dtype=np.float64)
            for dim_idx in range(num_dims):
                same_chunks = scores_same[set_name][dim_idx]
                not_chunks = scores_not_same[set_name][dim_idx]
                if same_chunks and not_chunks:
                    same = torch.cat(same_chunks, dim=0).detach().cpu().numpy()
                    not_same = torch.cat(not_chunks, dim=0).detach().cpu().numpy()
                    if same.size and not_same.size:
                        wdist_curve[dim_idx] = float(wasserstein_distance(same, not_same))

            curve_results[set_name][subsystem] = {
                "dims": dims.copy(),
                "wdist": wdist_curve,
                "kind": "intra",
            }

        print(f"{subsystem}: used {used_events}/{len(events)} events, skipped {skipped_events}")

    # Cross
    for subsystem_a, subsystem_b in cross_pairs:
        pair_name = f"{subsystem_a}-{subsystem_b}"
        scores_same = init_buffers()
        scores_not_same = init_buffers()
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

            for set_name, set_cfg in field_sets.items():
                coord_name = set_cfg["coord"]
                sin_a, cos_a = random_fourier_pos_enc_trig(
                    xs=cross_data["coords_a"][coord_name],
                    base_matrix=rff_base_matrices[coord_name],
                    scale=args.rff_scale,
                    max_half_dim=max_half_dim,
                )
                sin_b, cos_b = random_fourier_pos_enc_trig(
                    xs=cross_data["coords_b"][coord_name],
                    base_matrix=rff_base_matrices[coord_name],
                    scale=args.rff_scale,
                    max_half_dim=max_half_dim,
                )

                same_multi = pair_dot_products_multi_from_trig(
                    sin_left=sin_a,
                    cos_left=cos_a,
                    pairs=cross_data["same_pairs"],
                    half_dim_indices=half_dim_indices,
                    dim_values=dim_values,
                    normalize_by_dim=args.normalize_dot_by_dim,
                    sin_right=sin_b,
                    cos_right=cos_b,
                )
                not_same_multi = pair_dot_products_multi_from_trig(
                    sin_left=sin_a,
                    cos_left=cos_a,
                    pairs=cross_data["not_same_pairs"],
                    half_dim_indices=half_dim_indices,
                    dim_values=dim_values,
                    normalize_by_dim=args.normalize_dot_by_dim,
                    sin_right=sin_b,
                    cos_right=cos_b,
                )
                for dim_idx in range(num_dims):
                    scores_same[set_name][dim_idx].append(same_multi[dim_idx])
                    scores_not_same[set_name][dim_idx].append(not_same_multi[dim_idx])

        if used_events == 0:
            print(f"Skipping cross pair {pair_name}: no usable events found")
            continue

        for set_name in field_sets:
            wdist_curve = np.full(num_dims, np.nan, dtype=np.float64)
            for dim_idx in range(num_dims):
                same_chunks = scores_same[set_name][dim_idx]
                not_chunks = scores_not_same[set_name][dim_idx]
                if same_chunks and not_chunks:
                    same = torch.cat(same_chunks, dim=0).detach().cpu().numpy()
                    not_same = torch.cat(not_chunks, dim=0).detach().cpu().numpy()
                    if same.size and not_same.size:
                        wdist_curve[dim_idx] = float(wasserstein_distance(same, not_same))

            curve_results[set_name][pair_name] = {
                "dims": dims.copy(),
                "wdist": wdist_curve,
                "kind": "cross",
            }

        print(f"{pair_name}: used {used_events}/{len(events)} events, skipped {skipped_events}")

    return curve_results


def main():
    args = parse_args()
    if args.num_events <= 0:
        raise ValueError("--num-events must be > 0")
    if args.rff_scale <= 0:
        raise ValueError("--rff-scale must be > 0")
    if not args.dim_values:
        raise ValueError("--dim-values must not be empty")

    dims = np.asarray(sorted(set(args.dim_values)), dtype=np.int64)
    for dim in dims:
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"All --dim-values must be even and >= 2, got {dim}")

    events = load_events(args.config, args.event_index, args.num_events, args.num_workers)
    cross_pairs = parse_cross_subsystem_pairs(args.cross_subsystem_pairs) if args.include_cross else []
    field_sets = build_field_sets()
    coord_scale_by_subsystem = load_intra_scales_table(args.scales_table, args.subsystems)

    output_root = args.out_dir / args.out_prefix
    output_root.mkdir(parents=True, exist_ok=True)

    scale_summary_text = format_loaded_scales_summary(
        coord_scale_by_subsystem,
        title=f"Loaded intra optimal scales from {args.scales_table}:",
    )
    print(scale_summary_text)
    scale_summary_path = output_root / "tables" / "loaded_intra_scales.txt"
    write_text_report(scale_summary_path, scale_summary_text)
    print(f"Wrote {scale_summary_path}")

    max_half_dim = int(dims.max() // 2)
    rff_base_matrices = init_rff_base_matrices(BASE_COORD_NAMES, max_half_dim=max_half_dim, seed=args.seed)

    curve_results = run_dim_sweep(
        events=events,
        args=args,
        cross_pairs=cross_pairs,
        field_sets=field_sets,
        dims=dims,
        rff_base_matrices=rff_base_matrices,
        coord_scale_by_subsystem=coord_scale_by_subsystem,
    )

    written_paths = []
    plot_entries = []
    for set_name, set_curve_data in curve_results.items():
        if not set_curve_data:
            continue
        out_path = output_root / "summary" / f"wasserstein_vs_dim_{set_name}.png"
        plot_wasserstein_vs_dim(curve_data=set_curve_data, out_path=out_path, set_name=set_name)
        written_paths.append(out_path)
        plot_entries.append(
            {
                "path": out_path,
                "group_name": "all-pairs",
                "set_name": set_name,
                "title_prefix": f"Wasserstein-vs-dimension summary ({set_name})",
            }
        )

    for path in written_paths:
        print(f"Wrote {path}")

    optimal_dim_text = format_optimal_dim_summary(
        curve_results=curve_results,
        title="Optimal RFF dimensions by variable and pair type (argmax Wasserstein distance):",
    )
    print("")
    print(optimal_dim_text)
    optimal_dim_path = output_root / "tables" / "optimal_dims.txt"
    write_text_report(optimal_dim_path, optimal_dim_text)
    print(f"Wrote {optimal_dim_path}")

    if not args.skip_pdf:
        pdf_path = args.pdf_path if args.pdf_path is not None else (output_root / "summary.pdf")
        written_pdf = write_summary_pdf(plot_entries=plot_entries, pdf_path=pdf_path, args=args, cross_pairs=cross_pairs)
        if written_pdf is not None:
            print(f"Wrote {written_pdf}")

    print(f"Event index: {args.event_index}, num_events: {args.num_events}")
    print(f"Requested subsystems: {', '.join(args.subsystems)}")
    if cross_pairs:
        print(f"Requested cross pairs: {', '.join([f'{a}-{b}' for a, b in cross_pairs])}")
    else:
        print("Requested cross pairs: skipped")
    print(f"Fixed RFF scale: {args.rff_scale}")
    print(f"Dot-product normalization: {'enabled (divide by dim)' if args.normalize_dot_by_dim else 'disabled'}")
    print(f"Scanned dimensions: {', '.join(str(int(d)) for d in dims)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
