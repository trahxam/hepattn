from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDParticleDataset


plt.rcParams["figure.dpi"] = 300


config = yaml.safe_load(config_path.read_text())["data"]
config.update({
    "num_workers": num_workers,
    "batch_size": batch_size,
    "num_test": num_test,
    # "test_dir": "/share/rcif2/maxhart/data/cld/test/prepped/",
})
dm = CLDDataModule(**config)
dm.setup(stage="test")