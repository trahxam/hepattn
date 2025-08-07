# ruff: noqa: E501

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import time
import h5py
from scipy.stats import binned_statistic
from tqdm import tqdm
from torch import Tensor

from hepattn.experiments.cld.data import CLDDataset, CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

from hepattn.utils.plot import bayesian_binomial_error, plot_hist_to_ax
from hepattn.utils.eval import apply_matching, calc_cost, calc_binary_reco_metrics, calculate_selections
from hepattn.utils.tensor_utils import merge_batches
from hepattn.models.matcher import Matcher


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
eval_file_path = Path("logs/CLD_5_320_10MeV_neutrals_batched_20250805-T152453/ckpts/epoch=001-train_loss=4.66640_reco_eval.h5")

plot_save_dir = Path("src/hepattn/experiments/cld/eval_plots/")

# Load eval config
eval_config_path = Path("src/hepattn/experiments/cld/eval_config.yaml")
eval_config = yaml.safe_load(eval_config_path.read_text())["eval"]

# Which hits sets will be considered in the eval
hits = ["vtxd", "trkr", "ecal", "hcal"]

# Spec for event displays
evet_display_cfg = [
    {"x": "pos.x", "y": "pos.y", "px": "mom.x", "py": "mom.y", "input_names": hits},
    {"x": "pos.z", "y": "pos.y", "px": "mom.z", "py": "mom.y", "input_names": hits},
]

# Setup the matcher - use the same machinery as the model as a consistency check
matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False,)

bin_types = {
    "linear": np.linspace,
    "log": np.geomspace,
}

bins = {name: bin_types[cfg["scale"]](cfg["min"], cfg["max"], cfg["num"]) for name, cfg in eval_config["bins"].items()}
histograms = {name: {i: np.zeros(len(bins[cfg["bins"]]) - 1) for i in ["n", "k"]} for name, cfg in eval_config["histograms"].items()}

with h5py.File(eval_file_path, "r") as eval_file:
    for i, sample_id in tqdm(enumerate(eval_file.keys())):
        data = {}

        # Get the predictions just from the final layer
        final_preds = eval_file[f"{sample_id}/preds/final/"]

        # Load whether each slot was predicted as valid or not
        data[f"flow_valid"] = torch.from_numpy(final_preds["flow_valid/flow_valid"][:])

        for hit in hits:
            # Make sure to drop any invalid hit slots from the mask
            data[f"flow_{hit}_valid"] = torch.from_numpy(final_preds[f"flow_{hit}_assignment/flow_{hit}_valid"][:])

        # Get the sample
        sample = dataset.load_sample(int(sample_id))

        # Prepare it by adding any necessary padding etc
        inputs, targets = dataset.prep_sample(sample)

        # Add the input and target data from the dataset
        data |= targets

        for pred_object in ["particle", "pandora", "flow"]:
            costs = calc_cost(data, "particle", pred_object, eval_config["match_metrics"]["default"])

            data = apply_matching(data, "particle", pred_object, costs, matcher)

            eff_metrics = calc_binary_reco_metrics(data, "particle", pred_object, eval_config["binary_metrics"])
            pur_metrics = calc_binary_reco_metrics(data, pred_object, "particle", eval_config["binary_metrics"])

            data |= eff_metrics
            data |= pur_metrics

            for hit in hits:
                data[f"{pred_object}_num_{hit}"] = data[f"{pred_object}_{hit}_valid"].sum(-1)
            
            data[f"{pred_object}_num_sihit"] = data[f"{pred_object}_num_vtxd"] + data[f"{pred_object}_num_trkr"]

        for object_name in ["particle"]:
            data[f"{object_name}_has_tracker_hits"] = (data[f"{object_name}_vtxd_valid"].sum(-1) >= 4) & (data[f"{object_name}_trkr_valid"].sum(-1) >= 4)
            data[f"{object_name}_has_ecal_hits"] = data[f"{object_name}_ecal_valid"].sum(-1) >= 16
            data[f"{object_name}_has_hcal_hits"] = data[f"{object_name}_ecal_valid"].sum(-1) >= 16
            data[f"{object_name}_has_calo_hits"] = data[f"{object_name}_has_ecal_hits"] | data[f"{object_name}_has_hcal_hits"]

            selections = calculate_selections(data, object_name, eval_config["selections"])
            data |= selections

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

        if i == 0:
            for object_name in ["particle", "pandora", "flow"]:
                fig = plot_cld_event_reconstruction(inputs, data, evet_display_cfg, object_name)
                fig.savefig(plot_save_dir / Path(f"event_displays/{object_name}.png"))
    
        if i == 100:
            break



for name, cfg in eval_config["histogram_plots"].items():
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 3)

    for item_name, item_cfg in cfg["items"].items():
        hist_cfg = eval_config["histograms"][item_cfg["histogram"]]

        n_binned = histograms[item_cfg["histogram"]]["n"]
        k_binned = histograms[item_cfg["histogram"]]["k"]
        p_binned = k_binned / n_binned
        p_binned_err = bayesian_binomial_error(k_binned, n_binned)
        


        plot_hist_to_ax(ax, p_binned, bins[hist_cfg["bins"]], p_binned_err, label=item_cfg["label"], color=item_cfg["color"])

    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xscale(cfg["scale"])

    ax.legend(fontsize=8)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path(f"histograms/{name}.png"))



