from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from hepattn.experiments.cld.data import CLDDataModule

plt.rcParams["figure.dpi"] = 300


@pytest.fixture
def cld_datamodule():
    config_path = Path("src/hepattn/experiments/cld/configs/merged.yaml")
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0
    config["batch_size"] = 250

    datamodule = CLDDataModule(**config)
    datamodule.setup(stage="fit")

    return datamodule


def test_plot_cld_hit_dr(cld_datamodule):
    dataloader = cld_datamodule.train_dataloader()
    data_iterator = iter(dataloader)

    for _i in range(1):
        inputs, targets = next(data_iterator)

        mask = targets["particle_sihit_valid"]
        hit_phi = inputs["sihit_pos.phi"]
        hit_eta = inputs["sihit_pos.eta"]
        hit_time = inputs["sihit_time"]

        phi = np.ma.masked_array(mask * hit_phi[..., None, :], mask=~mask)
        eta = np.ma.masked_array(mask * hit_eta[..., None, :], mask=~mask)
        time = np.ma.masked_array(mask * hit_time[..., None, :], mask=~mask)

        std_phi = np.ma.std(phi, axis=-1)
        std_eta = np.ma.std(eta, axis=-1)

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(8, 3)

        ax[0].hist(std_eta.flatten(), bins=32, histtype="step")
        ax[1].hist(std_phi.flatten(), bins=32, histtype="step")

        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

        fig.tight_layout()
        fig.savefig(Path("tests/outputs/cld/cld_hit_std_eta_phi.png"))

        idx = np.ma.argsort(time, axis=-1)

        phi = np.take_along_axis(phi, idx, axis=-1)
        eta = np.take_along_axis(eta, idx, axis=-1)
        time = np.take_along_axis(time, idx, axis=-1)

        dphi = np.ma.diff(phi, axis=-1)
        dphi[dphi < -np.pi] += 2 * np.pi
        dphi[dphi > np.pi] -= 2 * np.pi

        deta = np.ma.diff(eta, axis=-1)
        dr = np.sqrt(deta**2 + dphi**2)

        fig, ax = plt.subplots(3, 2)
        fig.set_size_inches(8, 6)

        ax[0, 0].hist(deta.flatten(), bins=64, histtype="step")
        ax[1, 0].hist(dphi.flatten(), bins=64, histtype="step")
        ax[2, 0].hist(dr.flatten(), bins=64, histtype="step")

        ax[0, 0].set_yscale("log")
        ax[1, 0].set_yscale("log")
        ax[2, 0].set_yscale("log")

        ax[0, 0].set_xlabel(r"Hit $\Delta \eta$")
        ax[1, 0].set_xlabel(r"Hit $\Delta \phi$")
        ax[2, 0].set_xlabel(r"Hit $\Delta R$")

        ax[0, 0].set_ylabel("Count")

        ax[0, 1].hist(deta.flatten(), bins=np.linspace(-0.05, 0.05, 64), histtype="step")
        ax[1, 1].hist(dphi.flatten(), bins=np.linspace(-0.05, 0.05, 64), histtype="step")
        ax[2, 1].hist(dr.flatten(), bins=np.linspace(0.0, 0.05, 64), histtype="step")

        ax[0, 1].set_yscale("log")
        ax[1, 1].set_yscale("log")
        ax[2, 1].set_yscale("log")

        ax[0, 1].set_xlabel(r"Hit $\Delta \eta$")
        ax[1, 1].set_xlabel(r"Hit $\Delta \phi$")
        ax[2, 1].set_xlabel(r"Hit $\Delta R$")

        ax[0, 0].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(Path("tests/outputs/cld/cld_hit_deta_dphi_dr.png"))
