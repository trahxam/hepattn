import math
from dataclasses import dataclass
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
# Data model
# ----------------------------
@dataclass(frozen=True)
class Field:
    name: str  # underlying tensor name, e.g., "mom.r"
    label: str  # axis label
    scale: str  # "linear" | "log"
    bins: np.ndarray  # bin edges


# ----------------------------
# Field definitions
# ----------------------------
# Keys are "field ids" (what you refer to in PLOT_GROUPS).
# Multiple ids may point to the SAME underlying tensor by sharing Field.name.
FIELDS: dict[str, Field] = {
    "calib_energy_ecal": Field(
        name="calib_energy_ecal",
        label="Total Calibrated ECAL Energy [GeV]",
        scale="log",
        bins=np.logspace(-3, 2, 32),
    ),
    "calib_energy_hcal": Field(
        name="calib_energy_hcal",
        label="Total Calibrated HCAL Energy [GeV]",
        scale="log",
        bins=np.logspace(-3, 2, 32),
    ),
    "mom.r": Field(
        name="mom.r",
        label=r"$p_T$",
        scale="log",
        bins=np.geomspace(0.01, 365, 32),
    ),
    "mom.eta": Field(
        name="mom.eta",
        label=r"$\eta$",
        scale="linear",
        bins=np.linspace(-3, 3, 32),
    ),
    "mom.phi": Field(
        name="mom.phi",
        label=r"$\phi$",
        scale="linear",
        bins=np.linspace(-np.pi, np.pi, 32),
    ),
    "vtx.r": Field(
        name="vtx.r",
        label=r"Vertex $r$",
        scale="linear",
        bins=np.linspace(0, 500, 32),
    ),
    "isolation": Field(
        name="isolation",
        label=r"Particle Angular Isolation",
        scale="log",
        bins=np.geomspace(1e-4, math.pi, 32),
    ),
    "num_vtxd": Field(
        name="num_vtxd",
        label="Num. VTXD Hits",
        scale="linear",
        bins=np.arange(-1, 12) + 0.5,
    ),
    "num_trkr": Field(
        name="num_trkr",
        label="Num. Tracker Hits",
        scale="linear",
        bins=np.arange(-1, 12) + 0.5,
    ),
    "num_sihit": Field(
        name="num_sihit",
        label="Num. Si Hits",
        scale="linear",
        bins=np.arange(-1, 24) + 0.5,
    ),
    "num_ecal": Field(
        name="num_ecal",
        label="Num. ECAL Hits",
        scale="log",
        bins=np.geomspace(1, 1000, 32),
    ),
    "num_hcal": Field(
        name="num_hcal",
        label="Num. HCAL Hits",
        scale="log",
        bins=np.geomspace(1, 500, 32),
    ),
    "num_muon": Field(
        name="num_muon",
        label="Num. Muon Hits",
        scale="linear",
        bins=np.arange(-1, 14) + 0.5,
    ),
    "mass": Field(
        name="mass",
        label="Mass [GeV]",
        scale="log",
        bins=np.logspace(-4, 3, 32),
    ),
    "energy": Field(
        name="energy",
        label="Energy [GeV]",
        scale="log",
        bins=np.logspace(-3, 3, 32),
    ),
    "mom.rinv": Field(
        name="mom.rinv",
        label=r"$1/p_T$ [1/GeV]",
        scale="linear",
        bins=np.linspace(0, 25, 32),
    ),
    "mom.qopt": Field(
        name="mom.qopt",
        label=r"$q/p_T$ [1/GeV]",
        scale="linear",
        bins=np.linspace(-10, 10, 32),
    ),
    # --- Same underlying tensor, different binning ---
    "mom.qopt_fine": Field(
        name="mom.qopt",  # <- same source tensor
        label=r"$q/p_T$ [1/GeV]",
        scale="linear",
        bins=np.linspace(-1, 1, 32),
    ),
    "mom.sinphi": Field(
        name="mom.sinphi",
        label=r"$\sin \phi$",
        scale="linear",
        bins=np.linspace(-1, 1, 32),
    ),
    "mom.cosphi": Field(
        name="mom.cosphi",
        label=r"$\cos \phi$",
        scale="linear",
        bins=np.linspace(-1, 1, 32),
    ),
}

# ----------------------------
# Selections & styles
# ----------------------------
SELECTION_ALIASES: dict[str, str] = {
    "valid": "All",
    "is_charged": "Charged",
    "is_charged_hadron": "Charged Hadrons",
    "is_neutral_hadron": "Neutral Hadrons",
    "is_electron": "Electrons",
    "is_photon": "Photons",
    "is_muon": "Muons",
}

SELECTION_COLOURS: dict[str, str] = {
    "valid": "tab:blue",
    "is_charged": "tab:orange",
    "is_charged_hadron": "tab:green",
    "is_neutral_hadron": "tab:red",
    "is_electron": "tab:purple",
    "is_photon": "tab:brown",
    "is_muon": "tab:pink",
}

# Plot groups: {output_filename: [field ids]}
PLOT_GROUPS: dict[str, list[str]] = {
    "particle_pt_eta_phi": ["mom.r", "mom.eta", "mom.phi"],
    "particle_mass_energy": ["mass", "energy"],
    "particle_calo_energy": ["calib_energy_ecal", "calib_energy_hcal"],
    "particle_iso_d0": ["vtx.r", "isolation"],
    "particle_num_sihits": ["num_vtxd", "num_trkr", "num_sihit"],
    "particle_hits": ["num_ecal", "num_hcal", "num_muon"],
    "particle_qopt": ["mom.rinv", "mom.qopt", "mom.qopt_fine"],
    "particle_sincosphi": ["mom.sinphi", "mom.cosphi"],
}

OBJECT_NAME = "particle"


def build_histograms() -> dict[str, dict[str, CountingHistogram]]:
    """Create histograms for each field-id/selection combination.

    NOTE: Key by field *ids* (the dict keys), not Field.name. This allows
    multiple field ids to share the same underlying tensor but have different bins.
    """
    return {field_id: {sel: CountingHistogram(FIELDS[field_id].bins) for sel in SELECTION_ALIASES} for field_id in FIELDS}


def get_test_dataloader(config_path: Path, num_workers=10, batch_size=10, num_test=10_000):
    """Load CLD test dataloader with a few overrides."""
    config = yaml.safe_load(config_path.read_text())["data"]
    config.update({
        "num_workers": num_workers,
        "batch_size": batch_size,
        "num_test": num_test,
        # "test_dir": "/share/rcif2/maxhart/data/cld/test/prepped/",
    })
    dm = CLDDataModule(**config)
    dm.setup(stage="test")
    return dm.test_dataloader()


def fill_histograms(hists: dict[str, dict[str, CountingHistogram]], dataloader, steps: int = 25) -> None:
    """Iterate a few batches and fill histograms."""
    it = iter(dataloader)
    for _ in tqdm(range(steps)):
        try:
            _inputs, targets = next(it)
        except StopIteration:
            break

        for field_id, sel_map in hists.items():
            field = FIELDS[field_id]
            # Read from the underlying tensor name.
            values_all = targets[f"{OBJECT_NAME}_{field.name}"]
            for selection_name, hist in sel_map.items():
                mask = targets[f"{OBJECT_NAME}_{selection_name}"].bool()
                hist.fill(values_all[mask])


def plot_groups(
    hists: dict[str, dict[str, CountingHistogram]],
    out_dir: Path,
    groups: dict[str, list[str]] = PLOT_GROUPS,
) -> None:
    """Render grouped plots to PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for plot_name, field_ids in groups.items():
        fig, axes = plt.subplots(1, len(field_ids), figsize=(12, 3))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax_idx, field_id in enumerate(field_ids):
            field = FIELDS[field_id]
            ax = axes[ax_idx]

            for selection_name, hist in hists[field_id].items():
                plot_hist_to_ax(
                    ax,
                    hist.counts,
                    hist.bins,
                    label=SELECTION_ALIASES[selection_name],
                    color=SELECTION_COLOURS[selection_name],
                    vertical_lines=True,
                )

            ax.set_yscale("log")
            ax.set_xscale(field.scale)
            ax.set_xlabel(f"{OBJECT_NAME.capitalize()} {field.label}")
            ax.set_ylabel("Count")
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

        axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / f"{plot_name}.png")
        plt.close(fig)


def main():
    hists = build_histograms()
    dataloader = get_test_dataloader(Path("src/hepattn/experiments/cld/configs/base.yaml"))
    fill_histograms(hists, dataloader, steps=5)
    plot_groups(hists, Path("src/hepattn/experiments/cld/plots/data"))


if __name__ == "__main__":
    main()
