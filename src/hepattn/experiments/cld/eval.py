from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import binned_statistic
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataset
from hepattn.utils.eval_plots import bayesian_binomial_error, plot_hist_to_ax

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    config_path = Path("/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/cld/configs/sihit_ecal.yaml")
    eval_path = Path("/share/rcifdata/maxhart/hepattn-test/hepattn/logs/CLD_100mev_charged_tf_sihit_ecal_20250422-T221951/ckpts/epoch=024-val_loss=2.87862_test_eval.h5")

    # Now create the dataset
    config = yaml.safe_load(config_path.read_text())["data"]

    config["dirpath"] = Path("/share/rcifdata/maxhart/data/cld/prepped/test/")

    # Remve keys that are normally for the datamodule
    config_del_keys = [
        "train_dir",
        "test_dir",
        "val_dir",
        "num_train",
        "num_test",
        "num_val",
        "num_workers",
        "batch_size",
    ]

    for key in config_del_keys:
        config.pop(key)

    dataset = CLDDataset(**config)

    plot_specs = {
        "vtx.mom.r": ("Vertex $p_T$ [GeV]", np.geomspace(0.1, 100.0, 32), "log"),
        "vtx.mom.eta": (r"Vertex $\eta$", np.linspace(-4, 4, 32), "linear"),
        "vtx.mom.phi": (r"Vertex $\phi$", np.linspace(-np.pi, np.pi, 32), "linear"),
        "vtx.pos.r": ("Vertex $r_0$ [m]", np.linspace(0.0, 0.1, 32), "linear"),
        "vtx.pos.z": ("Vertex $z_0$ [m]", np.linspace(-0.1, 0.1, 32), "linear"),
        "num_sihit": ("Number of Silicon Hits", np.arange(6, 24) + 0.5, "linear"),
        "num_ecal": ("Number of ECAL Hits", np.geomspace(1, 10000, 32), "log"),
    }

    hits = ["sihit", "ecal"]

    particle_total_valid = {hit: {field: np.zeros(len(plot_specs[field][1]) - 1) for field in plot_specs} for hit in hits}
    particle_total_eff = {hit: {field: np.zeros(len(plot_specs[field][1]) - 1) for field in plot_specs} for hit in hits}

    flow_total_valid = {hit: {field: np.zeros(len(plot_specs[field][1]) - 1) for field in plot_specs} for hit in hits}
    flow_total_pur = {hit: {field: np.zeros(len(plot_specs[field][1]) - 1) for field in plot_specs} for hit in hits}

    for idx in tqdm(range(1000)):
        # Load the data from the event
        sample_id = dataset.sample_ids[idx]

        inputs, targets = dataset.load_event(sample_id)

        for hit in hits:
            hit_valid = targets[f"{hit}_valid"]

            # Loading a single event from the dataloader does not pad the particles, so we have to apply the
            # particle / object padding that was used for the model to both the particles and the masks
            particle_pad_size = dataset.event_max_num_particles - len(targets["particle_valid"])
            particle_valid = np.pad(targets["particle_valid"], ((0, particle_pad_size),), constant_values=False)

            particle_hit_valid = np.pad(targets[f"particle_{hit}_valid"], ((0, particle_pad_size), (0, 0)), constant_values=False)

            # Load the eval file
            with h5py.File(eval_path, "r") as eval_file:
                preds = eval_file[f"{sample_id}/preds/final/"]

                flow_valid = preds["flow_valid/flow_valid"][:]

                # The masks will have had the particle padding applied, but also the hit padding (since they are batched)
                flow_hit_valid = preds[f"flow_{hit}_assignment/flow_{hit}_valid"][:, :len(hit_valid)]

            particle_valid = particle_valid & (particle_hit_valid.sum(-1) > 0)

            hit_iou = (particle_hit_valid & flow_hit_valid).sum(-1) / (particle_hit_valid | flow_hit_valid).sum(-1)

            matched = particle_valid & flow_valid & (hit_iou >= 0.75)

            particle_eff = particle_valid & matched
            flow_eff = flow_valid & matched

            # Fill the particle histograms
            for field, (_, bins, _) in plot_specs.items():
                particle_field = np.pad(targets[f"particle_{field}"], ((0, particle_pad_size),), constant_values=0.0)

                # Do overflow binning
                particle_field = np.clip(particle_field, bins[0], bins[-1])

                num_valid, _, _ = binned_statistic(particle_field, particle_valid, statistic="sum", bins=bins)
                num_eff, _, _ = binned_statistic(particle_field, particle_eff, statistic="sum", bins=bins)

                particle_total_valid[hit][field] += num_valid
                particle_total_eff[hit][field] += num_eff

    plot_save_dir = Path(__file__).resolve().parent / Path("eval_plots")

    hit_aliases = {"sihit": "SiHit Assignment", "ecal": "ECAL Assignment"}

    # Now plot everything
    for hit in hits:
        for field, (alias, bins, scale) in plot_specs.items():
            total_valid = particle_total_valid[hit][field]
            total_eff = particle_total_eff[hit][field]

            # Total effieicny is the total number of effieint particles /
            # total number of valid (i.e. reconstructable) particles
            eff = total_eff / total_valid
            eff_errors = bayesian_binomial_error(total_eff, total_valid)

            # Plot the efficiency
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 3)

            plot_hist_to_ax(ax, eff, bins, eff_errors)

            ax.set_xlabel(f"Particle {alias}")
            ax.set_ylabel(f"Particle {hit_aliases[hit]} Efficiency")
            ax.set_xscale(scale)
            ax.legend()
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

            fig.savefig(plot_save_dir / Path(f"part_{hit}_eff_{field}.png"))

            # Plot distributions of truth quantities
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 3)

            plot_hist_to_ax(ax, total_valid, bins, vertical_lines=True)

            ax.set_xlabel(f"Particle {alias}")
            ax.set_ylabel("Count")
            ax.set_xscale(scale)
            ax.set_yscale("log")
            ax.legend()
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

            fig.savefig(plot_save_dir / Path(f"part_{hit}_{field}.png"))


if __name__ == "__main__":
    main()
