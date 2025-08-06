# ruff: noqa: E501

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import time
from scipy.stats import binned_statistic
from tqdm import tqdm
from torch import Tensor

from hepattn.experiments.cld.data import CLDDataset, CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

from hepattn.utils.plot import bayesian_binomial_error, plot_hist_to_ax
from hepattn.utils.eval import apply_matching, calc_cost, calc_binary_reco_metrics, calculate_selections
from hepattn.utils.tensor_utils import merge_batches


plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True

config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
config = yaml.safe_load(config_path.read_text())["data"]
config["num_workers"] = 5
config["batch_size"] = 5


datamodule = CLDDataModule(**config)
datamodule.setup(stage="test")
dataloader = datamodule.test_dataloader()
data_iterator = iter(dataloader)


inputs = []
targets = []

# Take a sample batch from the dataloader
for i in range(10):
    t0 = time.time()
    batch_inputs, batch_targets = next(data_iterator)
    inputs.append(batch_inputs)
    targets.append(batch_targets)
    
    print(time.time() - t0)
    print(batch_targets["sample_id"])
