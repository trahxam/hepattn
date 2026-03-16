from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plotting import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300

# ----------------------------
# Particle class definitions
# ----------------------------
PARTICLE_CLASSES = {
    "is_charged_hadron": "Charged Hadrons",
    "is_neutral_hadron": "Neutral Hadrons",
    "is_electron": "Electrons",
    "is_photon": "Photons",
    "is_muon": "Muons",
}

PARTICLE_COLOURS = {
    "is_charged_hadron": "tab:green",
    "is_neutral_hadron": "tab:red",
    "is_electron": "tab:purple",
    "is_photon": "tab:brown",
    "is_muon": "tab:pink",
}

# Bins for per-event particle counts
PARTICLE_COUNT_BINS = {
    "is_charged_hadron": np.arange(0, 128) + 0.5,
    "is_neutral_hadron": np.arange(0, 60) + 0.5,
    "is_electron": np.arange(0, 20) + 0.5,
    "is_photon": np.arange(0, 128) + 0.5,
    "is_muon": np.arange(0, 10) + 0.5,
}

# ----------------------------
# Hit type definitions
# ----------------------------
HIT_TYPES = {
    "vtxd": "VTXD",
    "trkr": "Tracker",
    "ecal": "ECAL",
    "hcal": "HCAL",
    "muon": "Muon",
}

HIT_COLOURS = {
    "vtxd": "tab:blue",
    "trkr": "tab:orange",
    "ecal": "tab:green",
    "hcal": "tab:red",
    "muon": "tab:purple",
}

# Bins for per-event hit counts
HIT_COUNT_BINS = {
    "vtxd": np.arange(0, 80) + 0.5,
    "trkr": np.arange(0, 80) + 0.5,
    "ecal": np.geomspace(1, 5000, 40),
    "hcal": np.geomspace(1, 1000, 40),
    "muon": np.arange(0, 60) + 0.5,
}


def get_test_dataloader(config_path: Path, num_workers=10, batch_size=10, num_test=10_000):
    config = yaml.safe_load(config_path.read_text())["data"]
    config.update({
        "num_workers": num_workers,
        "batch_size": batch_size,
        "num_test": num_test,
    })
    dm = CLDDataModule(**config)
    dm.setup(stage="test")
    return dm.test_dataloader()


def fill_histograms(particle_hists, hit_hists, particle_totals, dataloader, steps=25):
    it = iter(dataloader)
    for _ in tqdm(range(steps)):
        try:
            inputs, targets = next(it)
        except StopIteration:
            break

        # Per-event particle class counts
        for cls_key, hist in particle_hists.items():
            # mask shape: (batch, max_particles), sum over particle dim -> (batch,)
            counts = targets[f"particle_{cls_key}"].bool().sum(dim=-1).float()
            hist.fill(counts)
            particle_totals[cls_key] += counts.sum().item()

        # Per-event hit counts
        for hit_key, hist in hit_hists.items():
            counts = inputs[f"{hit_key}_valid"].bool().sum(dim=-1).float()
            hist.fill(counts)


def plot_particle_count_hists(particle_hists, out_dir: Path):
    keys = list(PARTICLE_CLASSES)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))

    for ax, cls_key in zip(axes, keys):
        hist = particle_hists[cls_key]
        plot_hist_to_ax(
            ax,
            hist.counts,
            hist.bins,
            label=PARTICLE_CLASSES[cls_key],
            color=PARTICLE_COLOURS[cls_key],
            vertical_lines=True,
        )
        ax.set_xlabel(f"Num. {PARTICLE_CLASSES[cls_key]} per Event")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "event_particle_counts.png")
    plt.close(fig)


def plot_particle_pie(particle_totals: dict, out_dir: Path):
    labels = [PARTICLE_CLASSES[k] for k in particle_totals]
    colours = [PARTICLE_COLOURS[k] for k in particle_totals]
    values = np.array([particle_totals[k] for k in particle_totals], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        colors=colours,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.8,
    )
    ax.set_title("Average Particle Composition per Event")
    fig.tight_layout()
    fig.savefig(out_dir / "event_particle_composition_pie.png")
    plt.close(fig)


def plot_hit_count_hists(hit_hists, out_dir: Path):
    keys = list(HIT_TYPES)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))

    for ax, hit_key in zip(axes, keys):
        hist = hit_hists[hit_key]
        scale = "log" if hit_key in ("ecal", "hcal") else "linear"
        plot_hist_to_ax(
            ax,
            hist.counts,
            hist.bins,
            label=HIT_TYPES[hit_key],
            color=HIT_COLOURS[hit_key],
            vertical_lines=True,
        )
        ax.set_xlabel(f"Num. {HIT_TYPES[hit_key]} Hits per Event")
        ax.set_ylabel("Count")
        ax.set_xscale(scale)
        ax.set_yscale("log")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "event_hit_counts.png")
    plt.close(fig)


def main():
    config_path = Path("src/hepattn/experiments/cld/configs/combined.yaml")
    out_dir = Path("src/hepattn/experiments/cld/plots/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_test_dataloader(config_path)

    particle_hists = {cls_key: CountingHistogram(PARTICLE_COUNT_BINS[cls_key]) for cls_key in PARTICLE_CLASSES}
    hit_hists = {hit_key: CountingHistogram(HIT_COUNT_BINS[hit_key]) for hit_key in HIT_TYPES}
    particle_totals = {cls_key: 0.0 for cls_key in PARTICLE_CLASSES}

    fill_histograms(particle_hists, hit_hists, particle_totals, dataloader, steps=250)

    plot_particle_count_hists(particle_hists, out_dir)
    plot_particle_pie(particle_totals, out_dir)
    plot_hit_count_hists(hit_hists, out_dir)


if __name__ == "__main__":
    main()
