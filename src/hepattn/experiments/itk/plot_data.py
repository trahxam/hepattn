from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.itk.data import ITkDataModule
from hepattn.utils.histogram import CountingHistogram
from hepattn.utils.plot import plot_hist_to_ax


plt.rcParams["figure.dpi"] = 300


aliases = {
    "pixel_x": r"Pixel Position $x$ [m]",
    "pixel_y": r"Pixel Position $y$ [m]",
    "pixel_z": r"Pixel Position $z$ [m]",

    "pixel_r": r"Pixel Position $r$ [m]",
    "pixel_eta": r"Pixel Position $\eta$",
    "pixel_phi": r"Pixel Position $\phi$",

    "pixel_count": r"Pixel Count",
    "pixel_log_charge_count": r"Pixel Log Charge Count",

    "pixel_loc_eta": r"Pixel Local $\eta$",
    "pixel_loc_phi": r"Pixel Local $\phi$",

    "pixel_glob_eta": r"Pixel Global $\eta$",
    "pixel_glob_phi": r"Pixel Global $\phi$",

    "pixel_norm_x": r"Pixel Normal $x$",
    "pixel_norm_y": r"Pixel Normal $y$",
    "pixel_norm_z": r"Pixel Normal $z$",

    "pixel_localDir0": r"Pixel Local Dir. 0",
    "pixel_localDir1": r"Pixel Local Dir. 1",
    "pixel_localDir2": r"Pixel Local Dir. 2",

    "pixel_lengthDir0": r"Pixel Length Dir. 0",
    "pixel_lengthDir1": r"Pixel Length Dir. 1",
    "pixel_lengthDir2": r"Pixel Length Dir. 2",

    "pixel_eta_module": r"Pixel Module $\eta$",
    "pixel_phi_module": r"Pixel Module $\phi$",

    "particle_pt": r"Particle $p_T$ [GeV]",
    "particle_eta": r"Particle $\eta$",
    "particle_phi": r"Particle $\phi$",
}

scales = {
    "pixel_x": "linear",
    "pixel_y": "linear",
    "pixel_z": "linear",

    "pixel_r": "linear",
    "pixel_eta": "linear",
    "pixel_phi": "linear",

    "pixel_count": "linear",
    "pixel_log_charge_count": "linear",

    "particle_pt": "log",
    "particle_eta": "linear",
    "particle_phi": "linear",

    "pixel_loc_eta": "linear",
    "pixel_loc_phi": "linear",

    "pixel_glob_eta": "linear",
    "pixel_glob_phi": "linear",

    "pixel_norm_x": "linear",
    "pixel_norm_y": "linear",
    "pixel_norm_z": "linear",

    "pixel_localDir0": "linear",
    "pixel_localDir1": "linear",
    "pixel_localDir2": "linear",

    "pixel_lengthDir0": "linear",
    "pixel_lengthDir1": "linear",
    "pixel_lengthDir2": "linear",

    "pixel_eta_module": "linear",
    "pixel_phi_module": "linear",
}

bins = {
    "pixel_x": np.linspace(-4, 4, 32),
    "pixel_y": np.linspace(-4, 4, 32),
    "pixel_z": np.linspace(-8, 8, 32),

    "pixel_r": np.linspace(0, 4, 32),
    "pixel_eta": np.linspace(-4, 4, 32),
    "pixel_phi": np.linspace(-np.pi, np.pi, 32),

    "pixel_count": np.arange(32),
    "pixel_log_charge_count": np.linspace(0, 10, 32),
    
    "particle_pt": np.geomspace(0.5, 1000, 32),
    "particle_eta": np.linspace(-4, 4, 32),
    "particle_phi": np.linspace(-np.pi, np.pi, 32),

    "pixel_loc_eta": np.linspace(-0.1, 0.1, 32),
    "pixel_loc_phi": np.linspace(-0.1, 0.1, 32),

    "pixel_glob_eta": np.linspace(-4, 4, 32),
    "pixel_glob_phi": np.linspace(-np.pi, np.pi, 32),

    "pixel_norm_x": np.linspace(-1, 1, 32),
    "pixel_norm_y": np.linspace(-1, 1, 32),
    "pixel_norm_z": np.linspace(-1, 1, 32),

    "pixel_localDir0": np.linspace(-1, 1, 32),
    "pixel_localDir1": np.linspace(-1, 1, 32),
    "pixel_localDir2": np.linspace(-1, 1, 32),

    "pixel_lengthDir0": np.linspace(0, 0.1, 32),
    "pixel_lengthDir1": np.linspace(0, 0.1, 32),
    "pixel_lengthDir2": np.linspace(0, 0.1, 32),

    "pixel_eta_module": np.linspace(-4, 4, 32),
    "pixel_phi_module": np.linspace(-np.pi, np.pi, 32),

}

test_dirs = {
    "GNN4ITk": "/share/rcifdata/maxhart/data/itk/test/",
    "Benjamin": "/share/lustre/maxhart/data/itk/prepped/",
}

test_dir_colours = {
    "GNN4ITk": "cornflowerblue",
    "Benjamin": "firebrick",
}

hists = {test_dir_name: {field: CountingHistogram(bins) 
        for field, bins in bins.items()}
        for test_dir_name in test_dirs}

for test_dir_name, test_dir in test_dirs.items():
    # Setup the dataset
    config_path = Path("src/hepattn/experiments/itk/configs/filtering_pixel.yaml")
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 1
    config["num_test"] = -1
    config["test_dir"] = test_dir

    datamodule = ITkDataModule(**config)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    data_iterator = iter(dataloader)

    # Iterate through the dataset
    for _i in tqdm(range(100)):
        inputs, targets = next(data_iterator)
        data = targets | inputs

        for field, hist in hists[test_dir_name].items():
            values = data[field][data[f"{field.split("_")[0]}_valid"].bool()]
            hist.fill(values)

plots = {
    "hit_xyz": [
        "pixel_x",
        "pixel_y",
        "pixel_z",
    ],

    "hit_retaphi": [
        "pixel_r",
        "pixel_eta",
        "pixel_phi",
    ],

    "hit_charge": [
        "pixel_count",
        "pixel_log_charge_count",
    ],

    "hit_local": [
        "pixel_loc_eta",
        "pixel_loc_phi",
    ],

    "hit_global": [
        "pixel_glob_eta",
        "pixel_glob_phi",
    ],

    "hit_normals": [
        "pixel_norm_x",
        "pixel_norm_y",
        "pixel_norm_z",
    ],

    "hit_local_dirs": [
        "pixel_localDir0",
        "pixel_localDir1",
        "pixel_localDir2",
    ],

    "hit_length_dirs": [
        "pixel_lengthDir0",
        "pixel_lengthDir1",
        "pixel_lengthDir2",
    ],
    "hit_modules": [
        "pixel_eta_module",
        "pixel_phi_module",
    ],


    "particle_ptetaphi": [
        "particle_pt",
        "particle_eta",
        "particle_phi",
    ],
    
}

for plot_name, fields in plots.items():
    fig, ax = plt.subplots(1, len(fields))
    fig.set_size_inches(12, 3)

    for ax_idx, field in enumerate(fields):
        for test_dir_name in test_dirs:            
            plot_hist_to_ax(
                ax[ax_idx],
                hists[test_dir_name][field].counts,
                hists[test_dir_name][field].bins,
                label=test_dir_name,
                color=test_dir_colours[test_dir_name],
                vertical_lines=True,
            )

        ax[ax_idx].set_yscale("log")
        ax[ax_idx].set_xscale(scales[field])
        ax[ax_idx].set_xlabel(f" {aliases[field]}")
        ax[ax_idx].set_ylabel("Count")
        ax[ax_idx].grid(zorder=0, alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(Path(f"src/hepattn/experiments/itk/plots/data/{plot_name}.png"))