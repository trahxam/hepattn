from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.plot_event import plot_cld_event_reconstruction

plt.rcParams["figure.dpi"] = 300


class TestCLDEvent:
    @pytest.fixture
    def cld_datamodule(self):
        config_path = Path("src/hepattn/experiments/cld/configs/merged.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]

        datamodule = CLDDataModule(**config)
        datamodule.setup(stage="fit")

        return datamodule

    def test_cld_batching(self, cld_datamodule):
        train_dataloader = cld_datamodule.train_dataloader()

        data_iterator = iter(train_dataloader)

        for i in range(10):
            inputs, targets = next(data_iterator)

            for k, v in inputs.items():
                print(k, v.shape)
            

    