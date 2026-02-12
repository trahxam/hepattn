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
from hepattn.models.posenc import pos_enc


def parse_args():
    parser = argparse.ArgumentParser(description="Compare hit positional-encoding dot products for same-particle vs non-same-particle hit pairs across subsystems.")
    parser.add_argument("--config", type=Path, default=Path("src/hepattn/experiments/cld/configs/unified.yaml"))
    parser.add_argument("--event-index", type=int, default=0, help="Zero-based start index in the test dataloader.")
    parser.add_argument("--num-events", type=int, default=10, help="Number of events to aggregate for statistics.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim-per-field", type=int, default=32, help="Positional-encoding dimension used per field (r, eta, phi).")
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--base", type=float, default=100.0)
    parser.add_argument("--rff-scale", type=float, default=1.0, help="Scale of random Fourier feature projection matrix.")
    parser.add_argument("--max-same-pairs", type=int, default=25000, help="Subsample same-particle pairs to this cap before plotting.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("src/hepattn/experiments/cld/plots/data"))
    parser.add_argument("--out-prefix", type=str, default="posenc_pair_dots")
    parser.add_argument("--pdf-path", type=Path, default=None, help="Optional path to write a consolidated summary PDF.")
    parser.add_argument("--skip-pdf", action="store_true", help="If set, skip writing the consolidated summary PDF.")
    parser.add_argument(
        "--subsystems",
        nargs="+",
        default=["vtxd", "trkr", "ecal", "hcal"],
        help="Subsystem inputs to plot (e.g. vtxd trkr ecal hcal)",
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
                f"Invalid --cross-subsystem-pairs value {pair_spec!r}. Expected format A-B, e.g. vtxd-trkr."
            )
        pair = (split[0], split[1])
        if pair in seen:
            continue
        parsed_pairs.append(pair)
        seen.add(pair)
    return parsed_pairs


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


def extract_subsystem_view(inputs, targets, subsystem):
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
    coords = {
        "x": inputs[f"{subsystem}_pos.x"][0][hit_valid].float(),
        "y": inputs[f"{subsystem}_pos.y"][0][hit_valid].float(),
        "z": inputs[f"{subsystem}_pos.z"][0][hit_valid].float(),
        "r": inputs[f"{subsystem}_pos.r"][0][hit_valid].float(),
        "eta": inputs[f"{subsystem}_pos.eta"][0][hit_valid].float(),
        "phi": inputs[f"{subsystem}_pos.phi"][0][hit_valid].float(),
    }
    return {"num_hits": num_hits, "coords": coords, "particle_hit_valid": particle_hit_valid}, None


def prepare_subsystem_data(inputs, targets, subsystem, max_same_pairs, rng):
    subsystem_view, reason = extract_subsystem_view(inputs=inputs, targets=targets, subsystem=subsystem)
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
    not_same_pairs = sample_not_same_pairs(
        hit_to_particle=hit_to_particle,
        num_pairs=same_pairs.shape[0],
        rng=rng,
    )
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


def prepare_cross_subsystem_data(inputs, targets, subsystem_a, subsystem_b, max_same_pairs, rng):
    subsystem_a_view, reason_a = extract_subsystem_view(inputs=inputs, targets=targets, subsystem=subsystem_a)
    if subsystem_a_view is None:
        return None, f"{subsystem_a}: {reason_a}"

    subsystem_b_view, reason_b = extract_subsystem_view(inputs=inputs, targets=targets, subsystem=subsystem_b)
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


def init_rff_matrices(coord_names, dim_per_field, scale, seed):
    half_dim = dim_per_field // 2
    if half_dim == 0:
        raise ValueError("--dim-per-field must be >= 2 for random Fourier features.")
    if scale <= 0:
        raise ValueError("--rff-scale must be > 0.")

    gen = torch.Generator()
    gen.manual_seed(seed)
    matrices = {}
    for coord_name in sorted(coord_names):
        matrices[coord_name] = scale * torch.randn((1, half_dim), generator=gen)
    return matrices


def random_fourier_pos_enc(xs, matrix, dim_per_field):
    matrix = matrix.to(device=xs.device, dtype=xs.dtype)
    proj = (2.0 * torch.pi * xs.unsqueeze(-1)) * matrix
    enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    # Keep output width consistent with --dim-per-field, even for odd dims.
    if enc.shape[-1] < dim_per_field:
        enc = torch.cat([enc, torch.zeros_like(enc[..., : dim_per_field - enc.shape[-1]])], dim=-1)
    return enc


def encode_field_set(coords, field_map, encoding_name, dim_per_field, alpha, base, rff_matrices):
    encodings = {}
    per_field_encodings = []
    for field_name, coord_name in field_map.items():
        values = coords[coord_name]
        if encoding_name == "sinusoidal":
            field_encoding = pos_enc(values, dim=dim_per_field, alpha=alpha, base=base)
        else:
            field_encoding = random_fourier_pos_enc(values, matrix=rff_matrices[coord_name], dim_per_field=dim_per_field)
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
        ax.set_xlabel("Dot product")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3, linestyle="--")

    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{title_prefix} PE dot-product comparison | events={num_events} | valid hits={hit_summary}")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _add_text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.05, 0.92, title, fontsize=18, fontweight="bold", va="top", ha="left")
    if lines:
        ax.text(0.05, 0.84, "\n".join(lines), fontsize=12, va="top", ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def write_summary_pdf(plot_entries, pdf_path, event_index, num_events, subsystems, cross_subsystem_pairs):
    if not plot_entries:
        return None

    encoding_order = {"sinusoidal": 0, "random_fourier": 1}
    category_order = {"intra": 0, "cross": 1}
    set_order = {"retaphi": 0, "xyz": 1}

    sorted_entries = sorted(
        plot_entries,
        key=lambda entry: (
            encoding_order.get(entry["encoding"], 99),
            category_order.get(entry["category"], 99),
            entry["group_name"],
            set_order.get(entry["set_name"], 99),
        ),
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _add_text_page(
            pdf,
            "Positional Encoding Dot-Product Summary",
            [
                f"Event start index: {event_index}",
                f"Number of events: {num_events}",
                f"Intra subsystems: {', '.join(subsystems)}",
                f"Cross pairs: {', '.join([f'{a}-{b}' for a, b in cross_subsystem_pairs])}",
                f"Number of plot pages: {len(sorted_entries)}",
            ],
        )

        current_encoding = None
        current_category = None
        current_group = None

        for entry in sorted_entries:
            if entry["encoding"] != current_encoding:
                current_encoding = entry["encoding"]
                current_category = None
                current_group = None
                _add_text_page(pdf, f"Encoding: {entry['encoding_label']}", [])

            if entry["category"] != current_category:
                current_category = entry["category"]
                current_group = None
                title = "Intra-Subsystem Comparisons" if current_category == "intra" else "Cross-Subsystem Comparisons"
                _add_text_page(pdf, title, [])

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


def main():
    args = parse_args()
    if args.num_events <= 0:
        raise ValueError("--num-events must be > 0")
    rng = np.random.default_rng(args.seed)

    events = load_events(args.config, args.event_index, args.num_events, args.num_workers)
    cross_subsystem_pairs = parse_cross_subsystem_pairs(args.cross_subsystem_pairs)

    encoding_methods = {
        "sinusoidal": "Sinusoidal",
        "random_fourier": "Random Fourier Features",
    }
    coord_names = ["x", "y", "z", "r", "eta", "phi"]
    rff_matrices = init_rff_matrices(coord_names, args.dim_per_field, args.rff_scale, args.seed)

    field_sets = {
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

    output_root = args.out_dir / args.out_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    written_paths = []
    plot_entries = []

    def init_score_buffers():
        score_buffers = {}
        for encoding_name in encoding_methods:
            score_buffers[encoding_name] = {}
            for set_name, set_cfg in field_sets.items():
                score_fields = ["combined"] + list(set_cfg["fields"].keys())
                score_buffers[encoding_name][set_name] = {field: [] for field in score_fields}
        return score_buffers

    # Intra-subsystem comparisons
    for subsystem in args.subsystems:
        per_encoding_set_scores_same = init_score_buffers()
        per_encoding_set_scores_random = init_score_buffers()
        used_events = 0
        skipped_events = 0
        total_hits = 0

        for inputs, targets in events:
            subsystem_data, _reason = prepare_subsystem_data(
                inputs=inputs,
                targets=targets,
                subsystem=subsystem,
                max_same_pairs=args.max_same_pairs,
                rng=rng,
            )
            if subsystem_data is None:
                skipped_events += 1
                continue

            used_events += 1
            total_hits += subsystem_data["num_hits"]

            # Compute dot-product scores for this event only, then aggregate
            # across events after all event-local pairings are done.
            for encoding_name in encoding_methods:
                for set_name, set_cfg in field_sets.items():
                    encodings = encode_field_set(
                        coords=subsystem_data["coords"],
                        field_map=set_cfg["fields"],
                        encoding_name=encoding_name,
                        dim_per_field=args.dim_per_field,
                        alpha=args.alpha,
                        base=args.base,
                        rff_matrices=rff_matrices,
                    )
                    scores_same = {name: pair_dot_products(emb, subsystem_data["same_pairs"]) for name, emb in encodings.items()}
                    scores_random = {name: pair_dot_products(emb, subsystem_data["not_same_pairs"]) for name, emb in encodings.items()}

                    for field_name in scores_same:
                        per_encoding_set_scores_same[encoding_name][set_name][field_name].append(scores_same[field_name])
                        per_encoding_set_scores_random[encoding_name][set_name][field_name].append(scores_random[field_name])

        if used_events == 0:
            print(f"Skipping {subsystem}: no usable events found")
            continue

        for encoding_name, encoding_label in encoding_methods.items():
            for set_name, set_cfg in field_sets.items():
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

                out_path = output_root / encoding_name / subsystem / f"{set_name}.png"
                plot_dot_products(
                    scores_same=scores_same,
                    scores_random=scores_random,
                    out_path=out_path,
                    field_order=list(set_cfg["fields"].keys()),
                    title_prefix=f"{subsystem.upper()} {set_cfg['title_suffix']} [{encoding_label}]",
                    num_events=used_events,
                    hit_summary=str(total_hits),
                )
                written_paths.append(out_path)
                plot_entries.append(
                    {
                        "path": out_path,
                        "encoding": encoding_name,
                        "encoding_label": encoding_label,
                        "category": "intra",
                        "group_name": subsystem,
                        "set_name": set_name,
                        "title_prefix": f"{subsystem.upper()} {set_cfg['title_suffix']} [{encoding_label}]",
                        "num_events": used_events,
                        "hit_summary": str(total_hits),
                    }
                )
        print(f"{subsystem}: used {used_events}/{len(events)} events, skipped {skipped_events}")

    # Cross-subsystem comparisons
    for subsystem_a, subsystem_b in cross_subsystem_pairs:
        pair_name = f"{subsystem_a}-{subsystem_b}"
        per_encoding_set_scores_same = init_score_buffers()
        per_encoding_set_scores_random = init_score_buffers()
        used_events = 0
        skipped_events = 0
        total_hits_a = 0
        total_hits_b = 0

        for inputs, targets in events:
            cross_data, _reason = prepare_cross_subsystem_data(
                inputs=inputs,
                targets=targets,
                subsystem_a=subsystem_a,
                subsystem_b=subsystem_b,
                max_same_pairs=args.max_same_pairs,
                rng=rng,
            )
            if cross_data is None:
                skipped_events += 1
                continue

            used_events += 1
            total_hits_a += cross_data["num_hits_a"]
            total_hits_b += cross_data["num_hits_b"]

            for encoding_name in encoding_methods:
                for set_name, set_cfg in field_sets.items():
                    encodings_a = encode_field_set(
                        coords=cross_data["coords_a"],
                        field_map=set_cfg["fields"],
                        encoding_name=encoding_name,
                        dim_per_field=args.dim_per_field,
                        alpha=args.alpha,
                        base=args.base,
                        rff_matrices=rff_matrices,
                    )
                    encodings_b = encode_field_set(
                        coords=cross_data["coords_b"],
                        field_map=set_cfg["fields"],
                        encoding_name=encoding_name,
                        dim_per_field=args.dim_per_field,
                        alpha=args.alpha,
                        base=args.base,
                        rff_matrices=rff_matrices,
                    )

                    scores_same = {
                        name: pair_dot_products(encodings_a[name], cross_data["same_pairs"], embeddings_right=encodings_b[name])
                        for name in encodings_a
                    }
                    scores_random = {
                        name: pair_dot_products(encodings_a[name], cross_data["not_same_pairs"], embeddings_right=encodings_b[name])
                        for name in encodings_a
                    }

                    for field_name in scores_same:
                        per_encoding_set_scores_same[encoding_name][set_name][field_name].append(scores_same[field_name])
                        per_encoding_set_scores_random[encoding_name][set_name][field_name].append(scores_random[field_name])

        if used_events == 0:
            print(f"Skipping cross pair {pair_name}: no usable events found")
            continue

        for encoding_name, encoding_label in encoding_methods.items():
            for set_name, set_cfg in field_sets.items():
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

                out_path = output_root / encoding_name / "cross" / pair_name / f"{set_name}.png"
                plot_dot_products(
                    scores_same=scores_same,
                    scores_random=scores_random,
                    out_path=out_path,
                    field_order=list(set_cfg["fields"].keys()),
                    title_prefix=f"{subsystem_a.upper()}-{subsystem_b.upper()} {set_cfg['title_suffix']} [{encoding_label}]",
                    num_events=used_events,
                    hit_summary=f"{total_hits_a}+{total_hits_b}",
                )
                written_paths.append(out_path)
                plot_entries.append(
                    {
                        "path": out_path,
                        "encoding": encoding_name,
                        "encoding_label": encoding_label,
                        "category": "cross",
                        "group_name": pair_name,
                        "set_name": set_name,
                        "title_prefix": f"{subsystem_a.upper()}-{subsystem_b.upper()} {set_cfg['title_suffix']} [{encoding_label}]",
                        "num_events": used_events,
                        "hit_summary": f"{total_hits_a}+{total_hits_b}",
                    }
                )
        print(f"{pair_name}: used {used_events}/{len(events)} events, skipped {skipped_events}")

    for path in written_paths:
        print(f"Wrote {path}")

    if not args.skip_pdf:
        pdf_path = args.pdf_path if args.pdf_path is not None else (output_root / "summary.pdf")
        written_pdf = write_summary_pdf(
            plot_entries=plot_entries,
            pdf_path=pdf_path,
            event_index=args.event_index,
            num_events=args.num_events,
            subsystems=args.subsystems,
            cross_subsystem_pairs=cross_subsystem_pairs,
        )
        if written_pdf is not None:
            print(f"Wrote {written_pdf}")

    print(f"Event index: {args.event_index}, num_events: {args.num_events}")
    print(f"Requested subsystems: {', '.join(args.subsystems)}")
    print(f"Requested cross pairs: {', '.join([f'{a}-{b}' for a, b in cross_subsystem_pairs])}")


if __name__ == "__main__":
    main()
