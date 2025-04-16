from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

from hepattn.experiments.cld.data import CLDDataset, collate_fn
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

plt.rcParams["figure.dpi"] = 300


class TestCLDEvent:
    @pytest.fixture
    def cld_dataset(self):
        config_path = Path("src/hepattn/experiments/cld/configs/base.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]

        dirpath = "/share/rcifdata/maxhart/data/cld/prepped/train"
        num_events = 10
        particle_min_pt = 0.1
        event_max_num_particles = 256

        merge_inputs = {
            "sihit": [
                "vtb",
                "vte",
                "itb",
                "ite",
                "otb",
                "ote",
            ],
            "ecal": ["ecb", "ece"],
            "hcal": [
                "hcb",
                "hce",
                "hco",
            ],
        }

        dataset = CLDDataset(
            dirpath=dirpath,
            inputs=config["inputs"],
            targets=config["targets"],
            num_events=num_events,
            particle_min_pt=particle_min_pt,
            event_max_num_particles=event_max_num_particles,
            merge_inputs=merge_inputs,
        )

        return dataset

    def test_cld_batching(self, cld_dataset):
        idxs = [1, 2, 4]

        batch = [cld_dataset[idx] for idx in idxs]

        batched_inputs, batched_targets = collate_fn(batch, cld_dataset.inputs, cld_dataset.targets)
        
        for k, v in batched_inputs.items():
            print(k, v.shape)
        
        for k, v in batched_targets.items():
            print(k, v.shape)

        


    