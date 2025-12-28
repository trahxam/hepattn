# src/hepattn/experiments/top/plots/make_feature_hists.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from hepattn.experiments.top.data import TopEventDataModule
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plotting import plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300


# --------------------------
# Labels / scales / bins
# --------------------------
aliases = {
    # jets (all provided by dataloader)
    "jet_pt": r"Jet $p_T$ [GeV]",
    "jet_pt_tev": r"Jet $p_T$ [TeV]",
    "jet_log_pt": r"Jet $\ln(p_T/\mathrm{GeV})$",
    "jet_eta": r"Jet $\eta$",
    "jet_phi": r"Jet $\phi$",
    "jet_e": r"Jet $E$ [GeV]",
    "jet_log_e": r"Jet $\ln(E/\mathrm{GeV})$",
    "jet_m": r"Jet $m$ [GeV]",
    "jet_log_m": r"Jet $\ln(m/\mathrm{GeV})$",
    "jet_bTag": r"Jet bTag",
    "jet_px": r"Jet $p_x$ [GeV]",
    "jet_py": r"Jet $p_y$ [GeV]",
    "jet_pz": r"Jet $p_z$ [GeV]",

    # jet pairs (provided by dataloader)
    "jet_jet_mjj": r"Jet-pair $m_{jj}$ [GeV]",
    "jet_jet_dphi": r"Jet-pair $\Delta\phi$",
    "jet_jet_deta": r"Jet-pair $\Delta\eta$",
    "jet_jet_dr": r"Jet-pair $\Delta R$",

    # particles (from targets; px/py computed in collate)
    "particle_pt": r"Particle $p_T$ [GeV]",
    "particle_eta": r"Particle $\eta$",
    "particle_phi": r"Particle $\phi$",
    "particle_m": r"Particle $m$ [GeV]",
    "particle_px": r"Particle $p_x$ [GeV]",
    "particle_py": r"Particle $p_y$ [GeV]",
    "particle_abs_pid": r"Particle $|pid|$",
}

scales = {
    # jets
    "jet_pt": "log",
    "jet_pt_tev": "log",
    "jet_log_pt": "linear",
    "jet_eta": "linear",
    "jet_phi": "linear",
    "jet_e": "log",
    "jet_log_e": "linear",
    "jet_m": "log",
    "jet_log_m": "linear",
    "jet_bTag": "linear",
    "jet_px": "linear",
    "jet_py": "linear",
    "jet_pz": "linear",

    # jet pairs
    "jet_jet_mjj": "log",
    "jet_jet_dphi": "linear",
    "jet_jet_deta": "linear",
    "jet_jet_dr": "linear",

    # particles
    "particle_pt": "log",
    "particle_eta": "linear",
    "particle_phi": "linear",
    "particle_m": "log",
    "particle_px": "linear",
    "particle_py": "linear",
    "particle_abs_pid": "log",
}

bins = {
    # jets
    "jet_pt": np.logspace(1, 3, 60),          # 10 .. 1000 GeV
    "jet_pt_tev": np.logspace(-2, 0, 60),     # 0.01 .. 1 TeV
    "jet_log_pt": np.linspace(0.0, 8.0, 80),
    "jet_eta": np.linspace(-5, 5, 60),
    "jet_phi": np.linspace(-np.pi, np.pi, 64),
    "jet_e": np.logspace(1, 3.5, 60),         # 10 .. ~3162 GeV
    "jet_log_e": np.linspace(0.0, 9.0, 80),
    "jet_m": np.logspace(-2, 2.5, 60),        # 0.01 .. ~316 GeV
    "jet_log_m": np.linspace(-10.0, 8.0, 100),
    "jet_bTag": np.array([-0.5, 0.5, 1.5]),   # 0/1
    "jet_px": np.linspace(-1000, 1000, 80),
    "jet_py": np.linspace(-1000, 1000, 80),
    "jet_pz": np.linspace(-2000, 2000, 80),

    # jet pairs
    "jet_jet_mjj": np.logspace(0, 4, 80),          # 1 .. 10000 GeV
    "jet_jet_dphi": np.linspace(-np.pi, np.pi, 64),
    "jet_jet_deta": np.linspace(-10, 10, 80),
    "jet_jet_dr": np.linspace(0, 10, 80),

    # particles
    "particle_pt": np.logspace(1, 3, 60),
    "particle_eta": np.linspace(-6, 6, 80),
    "particle_phi": np.linspace(-np.pi, np.pi, 64),
    "particle_m": np.logspace(-3, 2.5, 60),
    "particle_px": np.linspace(-1000, 1000, 80),
    "particle_py": np.linspace(-1000, 1000, 80),
    "particle_abs_pid": np.logspace(0, 4, 80),
}


# --------------------------
# Jet / particle groups
# --------------------------
jet_groups = {
    "all": "All jets",
    "btag": "b-tagged",
    "top1": "Top 1",
    "top2": "Top 2",
}

jet_colours = {
    "all": "tab:blue",
    "btag": "tab:orange",
    "top1": "tab:green",
    "top2": "tab:red",
}

particle_groups = {"all": "All particles"}
particle_colours = {"all": "tab:purple"}


# --------------------------
# Fields (assume all provided by dataloader/targets)
# --------------------------
jet_fields = [
    "jet_pt",
    "jet_pt_tev",
    "jet_log_pt",
    "jet_eta",
    "jet_phi",
    "jet_e",
    "jet_log_e",
    "jet_m",
    "jet_log_m",
    "jet_bTag",
    "jet_px",
    "jet_py",
    "jet_pz",
]

jet_pair_fields = [
    "jet_jet_mjj",
    "jet_jet_dphi",
    "jet_jet_deta",
    "jet_jet_dr",
]

particle_fields = [
    "particle_pt",
    "particle_eta",
    "particle_phi",
    "particle_m",
    "particle_px",
    "particle_py",
    "particle_abs_pid",
]


jet_hists = {f: {g: CountingHistogram(bins[f]) for g in jet_groups} for f in jet_fields}
jet_pair_hists = {f: {g: CountingHistogram(bins[f]) for g in jet_groups} for f in jet_pair_fields}
particle_hists = {f: {g: CountingHistogram(bins[f]) for g in particle_groups} for f in particle_fields}


# --------------------------
# Data
# --------------------------
config_path = Path("src/hepattn/experiments/top/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]

config["num_workers"] = 10
config["batch_size"] = 32
config["num_train"] = 1_000_000  # will clamp to available

datamodule = TopEventDataModule(**config)
datamodule.setup(stage="fit")
dataloader = datamodule.train_dataloader()
it = iter(dataloader)

num_batches = 200

for _ in tqdm(range(num_batches)):
    inputs, targets = next(it)

    jet_valid = inputs["jet_valid"].bool()
    jet_btag = (inputs["jet_bTag"] > 0.5) & jet_valid
    jet_top1 = targets["top_jet_valid"][:, 0, :].bool() & jet_valid
    jet_top2 = targets["top_jet_valid"][:, 1, :].bool() & jet_valid

    jet_masks = {
        "all": jet_valid,
        "btag": jet_btag,
        "top1": jet_top1,
        "top2": jet_top2,
    }

    # ---- jet-level ----
    for f in jet_fields:
        x = inputs[f]
        for g, msk in jet_masks.items():
            jet_hists[f][g].fill(x[msk].reshape(-1))

    # ---- jet-pair-level ----
    pair_valid = inputs["jet_jet_valid"].bool()  # (B, J, J)
    J = pair_valid.shape[-1]
    tri = torch.triu(torch.ones((J, J), dtype=torch.bool, device=pair_valid.device), diagonal=1)[None, :, :]
    pair_valid = pair_valid & tri  # count i<j only

    pair_masks = {g: pair_valid & (msk[:, :, None] & msk[:, None, :]) for g, msk in jet_masks.items()}

    for f in jet_pair_fields:
        x = inputs[f]  # (B, J, J)
        for g, pmask in pair_masks.items():
            jet_pair_hists[f][g].fill(x[pmask].reshape(-1))

    # ---- particles ----
    par_valid = targets["particle_valid"].bool()

    for f in particle_fields:
        if f == "particle_abs_pid":
            x = targets["particle_pid"].abs()
            particle_hists[f]["all"].fill(x[par_valid].reshape(-1))
        else:
            particle_hists[f]["all"].fill(targets[f][par_valid].reshape(-1))


# --------------------------
# Plot
# --------------------------
plots = {
    "jets_kinematics": ["jet_pt", "jet_eta", "jet_phi"],
    "jets_pt_tev": ["jet_pt_tev"],
    "jets_log": ["jet_log_pt", "jet_log_m", "jet_log_e"],
    "jets_mass_energy_btag": ["jet_m", "jet_e", "jet_bTag"],
    "jets_momentum": ["jet_px", "jet_py", "jet_pz"],

    "jet_pairs_mass": ["jet_jet_mjj"],
    "jet_pairs_geometry": ["jet_jet_dr", "jet_jet_dphi", "jet_jet_deta"],

    "particles_kinematics": ["particle_pt", "particle_eta", "particle_phi"],
    "particles_mass_momentum": ["particle_m", "particle_px", "particle_py"],
    "particles_pid": ["particle_abs_pid"],
}

out_dir = Path("src/hepattn/experiments/top/plots/data")
out_dir.mkdir(parents=True, exist_ok=True)

for plot_name, fields in plots.items():
    fig, ax = plt.subplots(1, len(fields))
    if len(fields) == 1:
        ax = [ax]
    fig.set_size_inches(4 * len(fields), 3)

    for ax_idx, f in enumerate(fields):
        if f in jet_hists:
            for g in jet_groups:
                hist = jet_hists[f][g]
                plot_hist_to_ax(
                    ax[ax_idx],
                    hist.counts,
                    hist.bins,
                    label=jet_groups[g],
                    color=jet_colours[g],
                    vertical_lines=True,
                )
        elif f in jet_pair_hists:
            for g in jet_groups:
                hist = jet_pair_hists[f][g]
                plot_hist_to_ax(
                    ax[ax_idx],
                    hist.counts,
                    hist.bins,
                    label=jet_groups[g],
                    color=jet_colours[g],
                    vertical_lines=True,
                )
        else:
            hist = particle_hists[f]["all"]
            plot_hist_to_ax(
                ax[ax_idx],
                hist.counts,
                hist.bins,
                label=particle_groups["all"],
                color=particle_colours["all"],
                vertical_lines=True,
            )

        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xscale(scales[f])
        ax[ax_idx].set_xlabel(aliases[f])
        ax[ax_idx].set_ylabel("Count")
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"{plot_name}.png")
    plt.close(fig)
