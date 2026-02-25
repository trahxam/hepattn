"""Plot TrackML event hits in eta/phi with per-particle colors.

Example:
  python src/hepattn/experiments/trackml/eval/plot_event_eta_phi.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hepattn.experiments.trackml.data import TrackMLDataset


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tracking.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TrackML event hits in eta/phi space.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a TrackML tracking config YAML.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--event-idx",
        type=int,
        default=0,
        help="Event index within the chosen split.",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        default=None,
        help="Optional sample_id to select instead of event-idx.",
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/trackml/eval/plots/trackml_event_eta_phi.png"),
        help="Output image path. Set to empty string to skip saving.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively (in addition to saving if --out is set).",
    )
    parser.add_argument("--dpi", type=int, default=400, help="DPI for saved figure.")
    parser.add_argument("--noise-size", type=float, default=1.5, help="Marker size for noise hits.")
    parser.add_argument("--signal-size", type=float, default=3.0, help="Marker size for signal hits.")
    parser.add_argument("--noise-alpha", type=float, default=0.25, help="Alpha for noise hits.")
    parser.add_argument("--signal-alpha", type=float, default=0.8, help="Alpha for signal hits.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for color assignment.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, split: str, use_hit_eval: bool, require_hit_eval: bool) -> TrackMLDataset:
    data = cfg["data"]
    dir_key = f"{split}_dir"
    num_key = f"num_{split}"
    hit_eval_key = f"hit_eval_{split}"

    dirpath = data[dir_key]
    num_events = data.get(num_key, -1)
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


def resolve_event_index(dataset: TrackMLDataset, event_idx: int, sample_id: int | None) -> int:
    if sample_id is None:
        return event_idx
    try:
        return dataset.sample_ids.index(sample_id)
    except ValueError as exc:
        raise ValueError(f"sample_id {sample_id} not found in dataset") from exc


def plot_event(hits, event_name: str, args: argparse.Namespace):
    signal_hits = hits[hits["on_valid_particle"]]
    noise_hits = hits[~hits["on_valid_particle"]]

    unique_ids = signal_hits["particle_id"].unique()
    rng = np.random.default_rng(args.seed)
    color_values = rng.permutation(np.linspace(0.0, 1.0, max(len(unique_ids), 1)))
    cmap = plt.get_cmap("turbo")
    pid_to_color = {pid: cmap(color_values[i]) for i, pid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=(6, 5))

    if len(noise_hits) > 0:
        ax.scatter(
            noise_hits["eta"],
            noise_hits["phi"],
            s=args.noise_size,
            color="lightgray",
            alpha=args.noise_alpha,
            edgecolors="black",
            linewidths=0.2,
            label="noise",
        )

    for pid, group in signal_hits.groupby("particle_id"):
        ax.scatter(
            group["eta"],
            group["phi"],
            s=args.signal_size,
            color=pid_to_color.get(pid, "tab:blue"),
            alpha=args.signal_alpha,
            edgecolors="black",
            linewidths=0.2,
        )

    ax.set_xlabel("eta")
    ax.set_ylabel("phi")
    ax.set_title(f"{event_name}: hits in eta/phi (signal={len(signal_hits)}, noise={len(noise_hits)})")
    ax.grid(alpha=0.25, linestyle="--")

    return fig


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset = build_dataset(
        cfg,
        args.split,
        use_hit_eval=not args.no_hit_eval,
        require_hit_eval=args.require_hit_eval,
    )
    idx = resolve_event_index(dataset, args.event_idx, args.sample_id)
    event_name = dataset.event_names[idx]

    hits, _particles = dataset.load_event(idx)
    fig = plot_event(hits, event_name, args)

    if args.out and str(args.out).strip():
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi)
        print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
