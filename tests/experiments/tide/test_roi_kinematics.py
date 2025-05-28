from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml

from hepattn.experiments.tide.data import ROIDataModule

plt.rcParams["figure.dpi"] = 300

torch.manual_seed(42)


class TestROIDataModule:
    @pytest.fixture
    def roi_datamodule(self):
        config_path = Path("src/hepattn/experiments/tide/configs/base.yaml")
        config = yaml.safe_load(config_path.read_text())["data"]
        config["num_workers"] = 0
        config["batch_size"] = 1000
        config["num_test"] = 10000

        datamodule = ROIDataModule(**config)
        datamodule.setup(stage="test")

        return datamodule

    def test_roi_data(self, roi_datamodule):
        dataloader = roi_datamodule.test_dataloader()
        dataset = dataloader.dataset
        data_iterator = iter(dataloader)
        
        output_dir = Path("tests/outputs/tide/")

        inputs, targets = next(data_iterator)

        for k, v in inputs.items():
            print(k, v.dtype, v.shape)
        
        for k, v in targets.items():
            print(k, v.dtype, v.shape)

        hits = ["pix", "sct"]
        global_fields = ["r", "theta", "eta", "phi"]
        local_fields = ["eta", "phi"]

        field_alias = {"r": r"$r$", "theta": r"$\theta$", "eta": r"$\eta$", "phi": r"$\phi$", }
        hit_alias = {"pix": "Pixel", "sct": "SCT"}
        
        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(8, 3)

        for i, hit in enumerate(hits):
            for j, field in enumerate(global_fields):
                ax[i,j].hist(inputs[f"{hit}_{field}"][inputs[f"{hit}_valid"]], bins=24, histtype="step")
                ax[i,j].set_xlabel(fr"{hit_alias[hit]} {field_alias[field]}")
                ax[i,j].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_global_coords.png"))

        fig, ax = plt.subplots(2, 4)
        fig.set_size_inches(8, 3)

        for i, hit in enumerate(hits):
            for j, field in enumerate(global_fields):
                ax[i,j].hist(inputs[f"{hit}_mod_{field}"][inputs[f"{hit}_valid"]], bins=24, histtype="step")
                ax[i,j].set_xlabel(fr"{hit_alias[hit]} Module {field_alias[field]}")
                ax[i,j].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_module_coords.png"))

        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(8, 3)

        for i, hit in enumerate(hits):
            for j, field in enumerate(local_fields):
                ax[i,j].hist(inputs[f"{hit}_d{field}"][inputs[f"{hit}_valid"]], bins=np.linspace(-0.1, 0.1, 24), histtype="step")
                ax[i,j].set_xlabel(fr"{hit_alias[hit]} ROI $\Delta${field_alias[field]}")
                ax[i,j].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_delta_coords.png"))

        tracks = ["sudo", "sisp", "reco"]
        track_fields = ["pt", "eta", "phi", "z0", "d0"]
        track_alias = {"sudo": "Pseudo Track", "sisp": "SiSp", "reco": "Reco"}
        field_alias = {"vx": r"$v_x$", "vy": r"$v_y$", "vz": r"$v_z$", "z0": r"$z_0$", "d0": r"$d_0$", "pt": r"$p_T$", "eta": r"$\eta$", "phi": r"$\phi$"}

        fig, ax = plt.subplots(len(tracks), len(track_fields))
        fig.set_size_inches(8, 4)

        for i, track in enumerate(tracks):
            for j, field in enumerate(track_fields):
                ax[i,j].hist(targets[f"{track}_{field}"][targets[f"{track}_valid"]], bins=24, histtype="step")
                ax[i,j].set_xlabel(fr"{track_alias[track]} {field_alias[field]}")
                ax[i,j].set_ylabel("Count")
                ax[i,j].set_yscale("log")

        fig.tight_layout()
        fig.savefig(output_dir / Path("tide_track_coords.png"))

