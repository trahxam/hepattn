"""Plot eta/phi spreads and index-window sizes for TrackML hits per particle.

Example:
  python src/hepattn/experiments/trackml/eval/plot_hit_spread.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hepattn.experiments.trackml.data import TrackMLDataset


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"
DEFAULT_OUT = Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/trackml/eval/plots/trackml_hit_spread.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eta/phi spreads and index spans for TrackML hits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to TrackML config YAML.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Dataset split.")
    parser.add_argument(
        "--num-events",
        type=int,
        default=1,
        help="Number of events to process (-1 for all).",
    )
    parser.add_argument(
        "--no-hit-eval",
        action="store_true",
        help="Disable hit-eval filtering even if present in config.",
    )
    parser.add_argument(
        "--require-hit-eval",
        action="store_true",
        help="Fail if the hit-eval file is missing instead of falling back.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output image path.")
    parser.add_argument("--dpi", type=int, default=400, help="DPI for saved figure.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str, num_events: int, use_hit_eval: bool, require_hit_eval: bool) -> TrackMLDataset:
    data = cfg["data"]
    dir_key = f"{split}_dir"
    num_key = f"num_{split}"
    hit_eval_key = f"hit_eval_{split}"

    dirpath = data[dir_key]
    cfg_num_events = data.get(num_key, -1)
    if num_events is None:
        num_events = cfg_num_events
    elif num_events < 0:
        num_events = cfg_num_events if cfg_num_events >= 0 else -1

    hit_eval_path = data.get(hit_eval_key)
    if not use_hit_eval:
        hit_eval_path = None

    if use_hit_eval and hit_eval_path:
        hit_eval_path = str(hit_eval_path)
        if not Path(hit_eval_path).is_file():
            if require_hit_eval:
                raise FileNotFoundError(
                    f"Hit-eval file not found: {hit_eval_path}. "
                    "Use --no-hit-eval to disable filtering."
                )
            print(f"Hit-eval file not found: {hit_eval_path}. Continuing without filtering.")
            hit_eval_path = None

    return TrackMLDataset(
        dirpath=dirpath,
        inputs=data["inputs"],
        targets=data["targets"],
        num_events=num_events,
        hit_volume_ids=data.get("hit_volume_ids"),
        feature_volume_ids=data.get("feature_volume_ids"),
        particle_min_pt=data.get("particle_min_pt", 1.0),
        particle_max_abs_eta=data.get("particle_max_abs_eta", 2.5),
        particle_min_num_hits=data.get("particle_min_num_hits", 3),
        event_max_num_particles=data.get("event_max_num_particles", 1000),
        strict_max_objects=data.get("strict_max_objects", False),
        hit_eval_path=hit_eval_path,
    )


def circular_span(angles: np.ndarray) -> float:
    if angles.size <= 1:
        return 0.0
    phi = np.sort(angles)
    gaps = np.diff(phi, append=phi[0] + 2 * np.pi)
    max_gap = gaps.max()
    return float(2 * np.pi - max_gap)


def index_spans(values: np.ndarray, particle_ids: np.ndarray) -> dict[int, int]:
    order = np.argsort(values, kind="mergesort")
    ordered_pids = particle_ids[order]

    min_idx = {}
    max_idx = {}
    for idx, pid in enumerate(ordered_pids):
        if pid not in min_idx:
            min_idx[pid] = idx
            max_idx[pid] = idx
        else:
            max_idx[pid] = idx

    return {pid: max_idx[pid] - min_idx[pid] + 1 for pid in min_idx}


def collect_spreads(dataset: TrackMLDataset) -> dict[str, dict[str, list[float]]]:
    selections = {
        "all": None,
        "pt>1": 1.0,
        "pt>2.5": 2.5,
    }
    metrics = {
        label: {"eta_ranges": [], "phi_ranges": [], "eta_spans": [], "phi_spans": []}
        for label in selections
    }

    for idx in range(len(dataset)):
        hits, _particles = dataset.load_event(idx)
        signal_hits = hits[hits["on_valid_particle"]]
        if len(signal_hits) == 0:
            continue

        particles = _particles.set_index("particle_id")
        pid_to_pt = particles["pt"].to_dict()

        grouped = signal_hits.groupby("particle_id")
        for pid, group in grouped:
            pt = pid_to_pt.get(pid)
            if pt is None:
                continue
            eta_range = float(group["eta"].max() - group["eta"].min())
            phi_range = float(circular_span(group["phi"].to_numpy()))
            for label, threshold in selections.items():
                if threshold is None or pt > threshold:
                    metrics[label]["eta_ranges"].append(eta_range)
                    metrics[label]["phi_ranges"].append(phi_range)

        pid = signal_hits["particle_id"].to_numpy()
        eta_span_map = index_spans(signal_hits["eta"].to_numpy(), pid)
        phi_span_map = index_spans(signal_hits["phi"].to_numpy(), pid)
        for pid, eta_span in eta_span_map.items():
            pt = pid_to_pt.get(pid)
            if pt is None:
                continue
            for label, threshold in selections.items():
                if threshold is None or pt > threshold:
                    metrics[label]["eta_spans"].append(float(eta_span))
                    metrics[label]["phi_spans"].append(float(phi_span_map[pid]))

    return metrics


def plot_distributions(metrics: dict[str, dict[str, list[float]]]):
    label_order = ["all", "pt>1", "pt>2.5"]
    colors = {
        "all": "black",
        "pt>1": "tab:blue",
        "pt>2.5": "tab:orange",
    }

    eta_all = np.asarray(metrics["all"]["eta_ranges"], dtype=float)
    phi_all = np.asarray(metrics["all"]["phi_ranges"], dtype=float)
    eta_spans_all = np.asarray(metrics["all"]["eta_spans"], dtype=float)
    phi_spans_all = np.asarray(metrics["all"]["phi_spans"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # Range histograms
    if eta_all.size:
        eta_max = np.percentile(eta_all, 99.5)
        bins = np.linspace(0, max(eta_max, 1e-3), 60)
        for label in label_order:
            data = np.asarray(metrics[label]["eta_ranges"], dtype=float)
            if data.size:
                axes[0, 0].hist(
                    data,
                    bins=bins,
                    histtype="step",
                    linewidth=1.5,
                    color=colors[label],
                    label=label,
                )
    axes[0, 0].set_title("Per-particle eta range")
    axes[0, 0].set_xlabel("max(eta) - min(eta)")
    axes[0, 0].set_ylabel("count")

    if phi_all.size:
        phi_max = min(np.percentile(phi_all, 99.5), 2 * np.pi)
        bins = np.linspace(0, max(phi_max, 1e-3), 60)
        for label in label_order:
            data = np.asarray(metrics[label]["phi_ranges"], dtype=float)
            if data.size:
                axes[0, 1].hist(
                    data,
                    bins=bins,
                    histtype="step",
                    linewidth=1.5,
                    color=colors[label],
                    label=label,
                )
    axes[0, 1].set_title("Per-particle phi range (circular)")
    axes[0, 1].set_xlabel("min arc covering hits [rad]")
    axes[0, 1].set_ylabel("count")

    # Index span CDFs
    for spans_all, ax, title, key in [
        (eta_spans_all, axes[1, 0], "Index span in eta-sorted hits", "eta_spans"),
        (phi_spans_all, axes[1, 1], "Index span in phi-sorted hits", "phi_spans"),
    ]:
        for label in label_order:
            spans = np.asarray(metrics[label][key], dtype=float)
            if spans.size:
                xs = np.sort(spans)
                ys = np.linspace(0, 1, xs.size, endpoint=True)
                ax.plot(xs, ys, color=colors[label], label=label)
        if spans_all.size:
            for q in [0.5, 0.9, 0.95, 0.99]:
                qx = np.percentile(spans_all, q * 100)
                ax.axvline(qx, color="tab:blue", alpha=0.3, linestyle="--")
                ax.text(qx, q, f"{int(q*100)}%={int(qx)}", rotation=0, va="bottom", fontsize=8)
            xmax = np.percentile(spans_all, 99.0)
            ax.set_xlim(0, max(1.0, xmax))
        ax.set_title(title)
        ax.set_xlabel("index span (max-min+1)")
        ax.set_ylabel("fraction <= span")
        ax.set_ylim(0, 1.0)

    for ax in axes.ravel():
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

    return fig


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset = build_dataset(
        cfg,
        args.split,
        num_events=args.num_events,
        use_hit_eval=not args.no_hit_eval,
        require_hit_eval=args.require_hit_eval,
    )

    metrics = collect_spreads(dataset)

    fig = plot_distributions(metrics)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved plot to {out_path}")

    for label_key in ["all", "pt>1", "pt>2.5"]:
        spans = np.asarray(metrics[label_key]["eta_spans"], dtype=float)
        spans_phi = np.asarray(metrics[label_key]["phi_spans"], dtype=float)
        if spans.size and spans_phi.size:
            q50, q90, q95 = np.percentile(spans, [50, 90, 95])
            q50p, q90p, q95p = np.percentile(spans_phi, [50, 90, 95])
            print(
                f"{label_key} eta index spans: 50%={q50:.0f}, 90%={q90:.0f}, 95%={q95:.0f} | "
                f"phi: 50%={q50p:.0f}, 90%={q90p:.0f}, 95%={q95p:.0f}"
            )


if __name__ == "__main__":
    main()
