import sys

import matplotlib.pyplot as plt
import torch

import hepattn.experiments.pixel.utils.cluster_plot as cluster_plot
from hepattn.experiments.pixel.scripts.plot_examples import parse_args
from hepattn.experiments.pixel.utils.cluster_plot import ensure_plot_fields


def test_ensure_plot_fields_requests_pitch_inputs_for_sparse_and_dense_views():
    data_cfg = {
        "inputs": {
            "pixel": ["x"],
        },
        "targets": {},
    }

    ensure_plot_fields(data_cfg)

    assert data_cfg["inputs"]["cluster"] == ["pitch_vector"]
    assert data_cfg["inputs"]["pixel"] == ["x", "y", "charge", "pitch_y"]


def test_plot_examples_cli_defaults_to_four_rows_and_auto_fills_example_slots(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plot_examples.py"])

    args = parse_args()

    assert args.example_rows == 4
    assert args.num_examples is None
    assert args.show_cluster_pitch_vector_title is True


def test_plot_examples_cli_can_disable_cluster_pitch_vector_titles(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plot_examples.py", "--no-cluster-pitch-vector-title"])

    args = parse_args()

    assert args.show_cluster_pitch_vector_title is False


def test_plot_charge_examples_sets_cluster_pitch_vector_title_when_enabled(monkeypatch, tmp_path):
    captured = {}

    def fake_finalize(fig, output_path, show, plt_mod, tight=True):
        captured["fig"] = fig

    monkeypatch.setattr(cluster_plot, "_finalize", fake_finalize)

    inputs = {
        "pixel_x": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "pixel_y": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        "pixel_charge": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "pixel_valid": torch.tensor([[True, True]]),
        "cluster_pitch_vector": torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
    }
    targets = {
        "particle_valid": torch.tensor([[True]]),
        "particle_x": torch.tensor([[0.0]], dtype=torch.float32),
        "particle_y": torch.tensor([[0.0]], dtype=torch.float32),
        "particle_phi": torch.tensor([[1.0]], dtype=torch.float32),
        "particle_theta": torch.tensor([[0.0]], dtype=torch.float32),
        "particle_primary": torch.tensor([[True]]),
        "particle_secondary": torch.tensor([[False]]),
        "particle_notruth": torch.tensor([[False]]),
    }

    cluster_plot.plot_charge_examples(
        inputs,
        targets,
        num_examples=1,
        output_dir=tmp_path,
        show=False,
        nrows=1,
        ncols=2,
        show_cluster_pitch_vector_title=True,
    )

    fig = captured["fig"]
    assert fig.axes[0].get_title() == "cluster_pitch_vector\n[10, 20, 30]"
    plt.close(fig)
