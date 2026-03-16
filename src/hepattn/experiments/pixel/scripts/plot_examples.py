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

from hepattn.experiments.pixel.utils.cluster_plot import (
    load_data_config,
    load_first_batch,
    plot_charge_examples,
    plot_charge_examples_3d,
    set_seed,
)


def parse_args():
    parser = ArgumentParser(description="Plot pixel-cluster examples.")
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
        default=1000,
        help="Batch size to draw for plotting.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to show in each output figure. Defaults to filling all non-key panels.",
    )
    parser.add_argument(
        "--example-rows",
        type=int,
        default=4,
        help="Number of rows in the examples grid (one panel is reserved for the key/colorbar).",
    )
    parser.add_argument(
        "--example-cols",
        type=int,
        default=5,
        help="Number of columns in the examples grid (one panel is reserved for the key/colorbar).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plots",
        help="Directory where `examples.pdf` and `examples_3d.pdf` will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving.",
    )
    parser.add_argument(
        "--no-pitch-y-labels",
        action="store_true",
        dest="show_pitch_y_labels",
        help="Disable per-row y-pitch labels on the 2D examples plot.",
    )
    parser.add_argument(
        "--no-cluster-pitch-vector-title",
        action="store_true",
        dest="show_cluster_pitch_vector_title",
        help="Disable cluster_pitch_vector titles on the example plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic batch loading and example selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    data_cfg = load_data_config(args.config, batch_size=args.batch_size, num_workers=args.num_workers)
    data_cfg["seed"] = int(args.seed)
    inputs, targets = load_first_batch(data_cfg, split=args.split)
    num_examples = args.num_examples if args.num_examples is not None else max(1, args.example_rows * args.example_cols - 1)

    plot_charge_examples(
        inputs,
        targets,
        num_examples,
        output_dir=args.output_dir,
        show=args.show,
        nrows=args.example_rows,
        ncols=args.example_cols,
        show_pitch_y_labels=args.show_pitch_y_labels,
        show_cluster_pitch_vector_title=args.show_cluster_pitch_vector_title,
    )
    plot_charge_examples_3d(
        inputs,
        targets,
        num_examples,
        output_dir=args.output_dir,
        show=args.show,
        nrows=args.example_rows,
        ncols=args.example_cols,
        show_cluster_pitch_vector_title=args.show_cluster_pitch_vector_title,
    )


if __name__ == "__main__":
    main()
