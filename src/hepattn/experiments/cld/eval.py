# ruff: noqa: E501

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import binned_statistic
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event
from hepattn.models.matcher import Matcher
from hepattn.utils.eval import apply_matching, calc_binary_reco_metrics, calc_cost, calculate_selections
from hepattn.utils.plot import plot_hist_to_ax
from hepattn.utils.stats import bayesian_binomial_error, combine_mean_std

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True

# Setup the dataset to load truth and pandora info from
config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]

# Need to overwrite whatever is in the config
config["num_workers"] = 0
config["batch_size"] = 1

# Load the entire test set, so that we can access any sample id
config["num_test"] = -1


# Get the dataset object
datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataset = datamodule.test_dataloader().dataset

# Define the test set eval we will look at
eval_file_path = Path("/share/rcifdata/maxhart/hepattn/logs/CLD_8_320_10MeV_neutrals_muon_20250809-T183715/ckpts/epoch=007-train_loss=2.60941_test_eval.h5")
# eval_file_path = Path("/share/rcifdata/maxhart/hepattn/logs/CLD_8_320_10MeV_tracking_20250808-T084158/ckpts/epoch=002-train_loss=2.24975_test_eval.h5")

eval_config_name = "eval_hadrons"

plot_save_dir = Path(f"src/hepattn/experiments/cld/eval_plots/{eval_config_name}/")
plot_save_dir.mkdir(parents=True, exist_ok=True)

event_display_save_dir = plot_save_dir / Path("event_displays")
event_display_save_dir.mkdir(parents=True, exist_ok=True)

histograms_save_dir = plot_save_dir / Path("histograms")
histograms_save_dir.mkdir(parents=True, exist_ok=True)

# Load eval config
eval_config_path = Path(f"src/hepattn/experiments/cld/{eval_config_name}.yaml")
eval_config = yaml.safe_load(eval_config_path.read_text())["eval"]

# Which hits sets will be considered in the eval
hits = ["vtxd", "trkr", "ecal", "hcal", "muon"]
# hits = ["vtxd", "trkr"]

pred_objects = ["particle", "pandora", "flow", "sitrack"]
pred_objects = ["particle", "pandora", "flow"]

# Spec for event displays
event_display_cfg = [
    {"x": "pos.x", "y": "pos.y", "px": "mom.x", "py": "mom.y", "input_names": hits, "xlabel": r"Global $x$", "ylabel": r"Global $y$"},
    {"x": "pos.z", "y": "pos.y", "px": "mom.z", "py": "mom.y", "input_names": hits, "xlabel": r"Global $z$", "ylabel": r"Global $y$"},
]


calo_hit_calibrations = {
    "ecal": 37.0,
    "hcal": 45.0,
}

# Setup the matcher - use the same machinery as the model as a consistency check
matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)

bin_types = {
    "linear": np.linspace,
    "log": np.geomspace,
}

bins = {name: bin_types[cfg["scale"]](cfg["min"], cfg["max"], cfg["num"]) for name, cfg in eval_config["bins"].items()}
histograms = {name: {i: np.zeros(len(bins[cfg["bins"]]) - 1) for i in ["n", "k"]} for name, cfg in eval_config["histograms"].items()}
bulk_metrics = {name: dict.fromkeys(["n", "k"], 0) for name in eval_config["bulk_metrics"]}

residual_metrics = {name: [] for name in eval_config["residual_metrics"]}

residual_histograms = {name: {i: np.zeros(len(bins[cfg["bins"]]) - 1) for i in ["mu", "sig", "n"]} for name, cfg in eval_config["residual_histograms"].items()}

largest_num_particles = 0

with h5py.File(eval_file_path, "r") as eval_file:
    for i, sample_id in tqdm(enumerate(eval_file.keys())):
        data = {}

        # Get the predictions just from the final layer
        final_preds = eval_file[f"{sample_id}/preds/final/"]
        final_outputs = eval_file[f"{sample_id}/outputs/final/"]

        # Load whether each slot was predicted as valid or not
        data["flow_valid"] = torch.from_numpy(final_preds["flow_valid/flow_valid"][:])
        data["flow_logit"] = torch.from_numpy(final_outputs["flow_valid/flow_logit"][:])

        data["flow_valid"] = data["flow_logit"].sigmoid() >= 0.5

        for hit in hits:
            # Make sure to drop any invalid hit slots from the mask
            data[f"flow_{hit}_valid"] = torch.from_numpy(final_preds[f"flow_{hit}_assignment/flow_{hit}_valid"][:])

        # Get the sample
        sample = dataset.load_sample(int(sample_id))

        # Prepare it by adding any necessary padding etc
        inputs, targets = dataset.prep_sample(sample)

        # Add the input and target data from the dataset
        data |= targets
        data |= inputs

        for pred_object in pred_objects:
            data[f"event_num_{pred_object}"] = data[f"{pred_object}_valid"].float().sum(-1)
            for hit in ["vtxd", "trkr", "ecal", "hcal"]:
                if f"{pred_object}_{hit}_valid" in data:
                    data[f"{pred_object}_{hit}_valid"] = data[f"{pred_object}_{hit}_valid"] & data[f"{pred_object}_valid"].unsqueeze(-1)

        for pred_object in pred_objects:
            for hit in ["ecal", "hcal"]:
                if f"{pred_object}_{hit}_valid" in data:
                    data[f"{pred_object}_{hit}_energy"] = data[f"{pred_object}_{hit}_valid"].float() * data[f"{hit}_energy"].unsqueeze(-2)
                    data[f"{pred_object}_energy_{hit}"] = data[f"{pred_object}_{hit}_energy"].sum(-1)

        for pred_object in pred_objects:
            data[f"{pred_object}_sihit_valid"] = torch.cat((data[f"{pred_object}_vtxd_valid"], data[f"{pred_object}_trkr_valid"]), dim=-1)

        for pred_object in pred_objects:
            costs = calc_cost(data, "particle", pred_object, eval_config["match_metrics"]["default"])

            data = apply_matching(data, "particle", pred_object, costs, matcher)

            eff_metrics = calc_binary_reco_metrics(data, "particle", pred_object, eval_config["binary_metrics"])
            pur_metrics = calc_binary_reco_metrics(data, pred_object, "particle", eval_config["binary_metrics"])

            data |= eff_metrics
            data |= pur_metrics

            for hit in hits:
                if f"{pred_object}_{hit}_valid" in data:
                    data[f"{pred_object}_num_{hit}"] = data[f"{pred_object}_{hit}_valid"].sum(-1)

            data[f"{pred_object}_num_sihit"] = data[f"{pred_object}_num_vtxd"] + data[f"{pred_object}_num_trkr"]

        for pred_object in pred_objects:
            if pred_object == "particle": continue
            data[f"{pred_object}_is_charged"] = data[f"{pred_object}_valid"] & (data[f"{pred_object}_num_sihit"] >= 4)
            data[f"{pred_object}_is_neutral"] = data[f"{pred_object}_valid"] & (data[f"{pred_object}_num_sihit"] == 0)

            if f"{pred_object}_energy_ecal" in data:
                data[f"{pred_object}_is_charged_hadron"] = data[f"{pred_object}_is_charged"] & (data[f"{pred_object}_energy_hcal"] >= 0.1) & (data[f"{pred_object}_energy_ecal"] >= 0.1)
                data[f"{pred_object}_is_neutral_hadron"] = data[f"{pred_object}_is_neutral"] & (data[f"{pred_object}_energy_hcal"] >= 0.1) & (data[f"{pred_object}_energy_ecal"] >= 0.1)

                data[f"{pred_object}_is_electron"] = data[f"{pred_object}_is_charged"] & (data[f"{pred_object}_num_hcal"] == 0) & (data[f"{pred_object}_energy_ecal"] >= 10)
                data[f"{pred_object}_is_photon"] = data[f"{pred_object}_is_neutral"] & (data[f"{pred_object}_num_hcal"] == 0) & (data[f"{pred_object}_energy_ecal"] >= 10)
                data[f"{pred_object}_is_muon"] = (data[f"{pred_object}_num_sihit"] >= 4) & (data[f"{pred_object}_num_ecal"] >= 10) & (data[f"{pred_object}_num_hcal"] >= 10) & (data[f"{pred_object}_num_muon"] >= 4)

        for object_name in ["particle"]:
            selections = calculate_selections(data, object_name, eval_config["selections"])
            data |= selections

        for name, cfg in eval_config["bulk_metrics"].items():
            selection = data[f"{cfg['object_name']}_{cfg['selection']}"].bool()
            n = data[f"{cfg['object_name']}_{cfg['denominator']}"][selection].float()
            k = data[f"{cfg['object_name']}_{cfg['numerator']}"][selection].float()

            bulk_metrics[name]["n"] += n.float().sum()
            bulk_metrics[name]["k"] += k.float().sum()

        for name, cfg in eval_config["residual_metrics"].items():
            selection = data[cfg["selection"]]
            true = data[cfg["true_object"]][selection]
            pred = data[cfg["pred_object"]][selection]

            residual_metrics[name].append(pred - true)

        for name, cfg in eval_config["residual_histograms"].items():
            selection = data[cfg["selection"]]
            true = data[cfg["true_field"]][selection]
            pred = data[cfg["pred_field"]][selection]

            field = data[f"{cfg['field']}"][selection].float()

            if selection.float().sum() == 0: continue

            res = pred - true

            mu_binned, _, _ = binned_statistic(field, res, statistic="mean", bins=bins[cfg["bins"]])
            sig_binned, _, _ = binned_statistic(field, res, statistic="std", bins=bins[cfg["bins"]])
            n_binned, _, _ = binned_statistic(field, res, statistic="count", bins=bins[cfg["bins"]])

            mu_combined, sig_combined, n_combined = combine_mean_std(
                residual_histograms[name]["mu"],
                residual_histograms[name]["sig"],
                residual_histograms[name]["n"],
                mu_binned,
                sig_binned,
                n_binned,
            )

            residual_histograms[name]["mu"] = mu_combined
            residual_histograms[name]["sig"] = sig_combined
            residual_histograms[name]["n"] = n_combined

        for name, cfg in eval_config["histograms"].items():
            selection = data[f"{cfg['object_name']}_{cfg['selection']}"]

            field = data[f"{cfg['object_name']}_{cfg['field']}"][selection].float()
            n = data[f"{cfg['object_name']}_{cfg['denominator']}"][selection].float()
            k = data[f"{cfg['object_name']}_{cfg['numerator']}"][selection].float()

            if n.sum() == 0:
                continue

            if k.sum() == 0:
                continue

            n_binned, _, _ = binned_statistic(field, n, statistic="sum", bins=bins[cfg["bins"]])
            k_binned, _, _ = binned_statistic(field, k, statistic="sum", bins=bins[cfg["bins"]])

            histograms[name]["n"] += n_binned
            histograms[name]["k"] += k_binned

        if data["event_num_particle"] > largest_num_particles:
            largest_num_particles = data["event_num_particle"]
            print(i, sample_id, largest_num_particles)

        if False:  # int(sample_id) == 1226276301090181:
            for object_name, criteria in [
                ("particle", "pandora_charged_reconstructed"),
                ("particle", "flow_charged_reconstructed"),
                # ("particle", "sitrack_charged_reconstructed_tight"),
                ("pandora", "particle_charged_reconstructed"),
                ("flow", "particle_charged_reconstructed"),
                # ("sitrack", "particle_charged_reconstructed_tight"),
                ]:
                fig = plot_cld_event(data, event_display_cfg, object_name)
                fig.savefig(event_display_save_dir / Path(f"{object_name}.png"))

        if i == 2500:
            break


for name, cfg in eval_config["bulk_metrics"].items():
    n = bulk_metrics[name]["n"]
    k = bulk_metrics[name]["k"]
    print(f"{name}: {k}/{n} ({(100 * k / n):.3f}%)")

for name, cfg in eval_config["histogram_plots"].items():
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    for item_name, item_cfg in cfg["items"].items():
        hist_cfg = eval_config["histograms"][item_cfg["histogram"]]

        n_binned = histograms[item_cfg["histogram"]]["n"]
        k_binned = histograms[item_cfg["histogram"]]["k"]
        p_binned = k_binned / n_binned
        p_binned_err = bayesian_binomial_error(k_binned, n_binned)

        plot_hist_to_ax(
            ax,
            p_binned,
            bins[hist_cfg["bins"]],
            p_binned_err,
            label=item_cfg.get("label"),
            color=item_cfg["color"],
            linestyle=item_cfg.get("linestyle"),
            )

    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xscale(cfg["scale"])

    ax.legend(fontsize=8)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    ymin, ymax = ax.get_ylim()
    ymin = max(ymin, 0.05)
    ymax = min(ymax, 1.01)
    ax.set_ylim(ymin, ymax)

    fig.savefig(histograms_save_dir / Path(f"{name}.png"))


for name, cfg in eval_config["residual_histogram_plots"].items():
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(6, 4)

    for item_name, item_cfg in cfg["items"].items():
        hist_cfg = eval_config["residual_histograms"][item_cfg["histogram"]]

        mu = residual_histograms[item_cfg["histogram"]]["mu"]
        sig = residual_histograms[item_cfg["histogram"]]["sig"]

        plot_hist_to_ax(
            ax[0],
            mu,
            bins[hist_cfg["bins"]],
            label=item_cfg.get("label"),
            color=item_cfg["color"],
            linestyle=item_cfg.get("linestyle"),
            )

        plot_hist_to_ax(
            ax[1],
            sig,
            bins[hist_cfg["bins"]],
            label=item_cfg.get("label"),
            color=item_cfg["color"],
            linestyle=item_cfg.get("linestyle"),
            )

    ax[1].set_xlabel(cfg["xlabel"])
    ax[0].set_ylabel("Mean " + cfg["ylabel"])
    ax[1].set_ylabel("S.D. " + cfg["ylabel"])

    # ax[0].set_ylim(-1.0, 1.0)
    # ax[1].set_ylim(-0.5, 0.5)

    ax[0].set_xscale(cfg["scale"])
    ax[1].set_xscale(cfg["scale"])
    ax[1].set_yscale("log")

    ax[0].legend(fontsize=8)
    ax[0].grid(zorder=0, alpha=0.25, linestyle="--")
    ax[1].grid(zorder=0, alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(histograms_save_dir / Path(f"{name}.png"))
