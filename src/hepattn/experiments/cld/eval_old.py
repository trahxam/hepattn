from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.models.matcher import Matcher
from hepattn.utils.eval_utils import (
    apply_matching,
    calc_binary_reco_metrics,
    calc_cost,
    calculate_selections,
)
from hepattn.utils.histogram import GaussianHistogram, PoissonHistogram
from hepattn.utils.plotting import plot_hist_to_ax
from hepattn.utils.stats import bayesian_binomial_error

# ---------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def to_numpy_1d(x: Any) -> np.ndarray:
    """Torch/array-like -> flattened float32 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(np.float32).reshape(-1)


def to_bool_1d(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(bool).reshape(-1)


def scalar_sum(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().sum().item())
    if isinstance(x, np.ndarray):
        return float(x.sum().item())
    return float(np.asarray(x).sum())


# ---------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------
CONFIG_PATH = Path("src/hepattn/experiments/cld/configs/base.yaml")
EVAL_CONFIG_NAME = "eval_hadrons"
EVAL_FILE_PATH = Path(
    "/share/rcifdata/maxhart/hepattn/logs/"
    "CLD_8_320_10MeV_neutrals_20251025-T213514/ckpts/"
    "epoch=002-train_loss=4.08055_test_eval.h5"
)

HITS = ["vtxd", "trkr", "ecal", "hcal", "muon"]
PRED_OBJECTS = ["particle", "pandora", "flow"]

# Optional early-stop; set None for full run
MAX_EVENTS: int | None = 1000


def main() -> None:
    # -----------------------------------------------------------------
    # Load configs
    # -----------------------------------------------------------------
    data_cfg = yaml.safe_load(CONFIG_PATH.read_text())["data"]
    data_cfg["num_workers"] = 0
    data_cfg["batch_size"] = 1
    data_cfg["num_test"] = -1

    eval_cfg_path = Path(
        f"src/hepattn/experiments/cld/eval_configs/{EVAL_CONFIG_NAME}.yaml"
    )
    eval_cfg = yaml.safe_load(eval_cfg_path.read_text())["eval"]

    # -----------------------------------------------------------------
    # Output dirs
    # -----------------------------------------------------------------
    plot_root = Path(f"src/hepattn/experiments/cld/eval_plots/{EVAL_CONFIG_NAME}/")
    (plot_root / "event_displays").mkdir(parents=True, exist_ok=True)
    hists_dir = plot_root / "histograms"
    hists_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Data module / dataset
    # -----------------------------------------------------------------
    datamodule = CLDDataModule(**data_cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.test_dataloader().dataset  # type: ignore[assignment]

    # -----------------------------------------------------------------
    # Matcher and binning
    # -----------------------------------------------------------------
    matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)

    bin_types = {"linear": np.linspace, "log": np.geomspace}
    bins: dict[str, np.ndarray] = {
        name: bin_types[cfg["scale"]](cfg["min"], cfg["max"], cfg["num"])
        for name, cfg in eval_cfg["bins"].items()
    }

    # -----------------------------------------------------------------
    # Histogram objects
    # -----------------------------------------------------------------
    # Efficiency/purity histograms
    poisson_hists: dict[str, PoissonHistogram] = {}
    for name, cfg in eval_cfg["histograms"].items():
        field_key = f"{cfg['object_name']}_{cfg['field']}"
        sel_key = f"{cfg['object_name']}_{cfg['selection']}"
        num_key = f"{cfg['object_name']}_{cfg['numerator']}"
        den_key = f"{cfg['object_name']}_{cfg['denominator']}"
        poisson_hists[name] = PoissonHistogram(
            field=field_key,
            bins=bins[cfg["bins"]],
            selection=sel_key,
            numerator=num_key,
            denominator=den_key,
        )

    # Residual histograms (Gaussian summary of residuals)
    gauss_hists: dict[str, GaussianHistogram] = {}
    for name, cfg in eval_cfg["residual_histograms"].items():
        # We'll provide a temporary "residual" values array at fill-time
        gauss_hists[name] = GaussianHistogram(
            field=cfg["field"],
            bins=bins[cfg["bins"]],
            selection=cfg["selection"],
            values="residual",
        )

    # Bulk metrics accumulators
    bulk_metrics: dict[str, dict[str, float]] = {
        name: {"n": 0.0, "k": 0.0} for name in eval_cfg["bulk_metrics"]
    }

    # -----------------------------------------------------------------
    # Event loop
    # -----------------------------------------------------------------
    largest_num_particles = 0.0

    with h5py.File(EVAL_FILE_PATH, "r") as f:
        keys = list(f.keys())
        for i, sample_id in tqdm(enumerate(keys), total=len(keys)):
            # ---------------------------------------------
            # Load preds/outputs (final layer only)
            # ---------------------------------------------
            preds = f[f"{sample_id}/preds/final/"]
            outs = f[f"{sample_id}/outputs/final/"]

            data: dict[str, Any] = {}
            data["flow_logit"] = torch.from_numpy(outs["flow_valid/flow_logit"][:])
            data["flow_valid"] = data["flow_logit"].sigmoid() >= 0.5

            for hit in HITS:
                key = f"flow_{hit}_assignment/flow_{hit}_valid"
                if key in preds:
                    data[f"flow_{hit}_valid"] = torch.from_numpy(preds[key][:])

            # ---------------------------------------------
            # Load and prepare the sample
            # ---------------------------------------------
            sample = dataset.load_sample(int(sample_id))
            inputs, targets = dataset.prep_sample(sample)
            data |= targets
            data |= inputs

            # Align predicted hit slots to truth hit counts
            for hit in HITS:
                flow_key = f"flow_{hit}_valid"
                truth_key = f"{hit}_valid"
                if flow_key in data and truth_key in data:
                    n_truth = data[truth_key].shape[-1]
                    data[flow_key] = data[flow_key][:, :, :n_truth]

            # ---------------------------------------------
            # Per-object bookkeeping
            # ---------------------------------------------
            for obj in PRED_OBJECTS:
                data[f"event_num_{obj}"] = data[f"{obj}_valid"].float().sum(-1)
                for hit in ("vtxd", "trkr", "ecal", "hcal"):
                    key = f"{obj}_{hit}_valid"
                    if key in data:
                        data[key] = data[key] & data[f"{obj}_valid"].unsqueeze(-1)

            for obj in PRED_OBJECTS:
                for hit in ("ecal", "hcal"):
                    key = f"{obj}_{hit}_valid"
                    if key in data:
                        data[f"{obj}_{hit}_energy"] = (
                            data[key].float() * data[f"{hit}_energy"].unsqueeze(-2)
                        )
                        data[f"{obj}_energy_{hit}"] = data[f"{obj}_{hit}_energy"].sum(-1)

            for obj in PRED_OBJECTS:
                data[f"{obj}_sihit_valid"] = torch.cat(
                    (data[f"{obj}_vtxd_valid"], data[f"{obj}_trkr_valid"]), dim=-1
                )

            # ---------------------------------------------
            # Matching and binary metrics
            # ---------------------------------------------
            for obj in PRED_OBJECTS:
                costs = calc_cost(data, "particle", obj, eval_cfg["match_metrics"]["default"])
                data = apply_matching(data, "particle", obj, costs, matcher)

                eff_metrics = calc_binary_reco_metrics(
                    data, "particle", obj, eval_cfg["binary_metrics"]
                )
                pur_metrics = calc_binary_reco_metrics(
                    data, obj, "particle", eval_cfg["binary_metrics"]
                )
                data |= eff_metrics
                data |= pur_metrics

                for hit in HITS:
                    key = f"{obj}_{hit}_valid"
                    if key in data:
                        data[f"{obj}_num_{hit}"] = data[key].sum(-1)

                data[f"{obj}_num_sihit"] = data[f"{obj}_num_vtxd"] + data[f"{obj}_num_trkr"]

            # ---------------------------------------------
            # Loose PID-style flags for non-truth objects
            # ---------------------------------------------
            for obj in PRED_OBJECTS:
                if obj == "particle":
                    continue

                data[f"{obj}_is_charged"] = data[f"{obj}_valid"] & (data[f"{obj}_num_sihit"] >= 4)
                data[f"{obj}_is_neutral"] = data[f"{obj}_valid"] & (data[f"{obj}_num_sihit"] == 0)

                if f"{obj}_energy_ecal" in data and f"{obj}_energy_hcal" in data:
                    data[f"{obj}_is_charged_hadron"] = (
                        data[f"{obj}_is_charged"]
                        & (data[f"{obj}_energy_hcal"] >= 0.1)
                        & (data[f"{obj}_energy_ecal"] >= 0.1)
                    )
                    data[f"{obj}_is_neutral_hadron"] = (
                        data[f"{obj}_is_neutral"]
                        & (data[f"{obj}_energy_hcal"] >= 0.1)
                        & (data[f"{obj}_energy_ecal"] >= 0.1)
                    )
                    data[f"{obj}_is_electron"] = (
                        data[f"{obj}_is_charged"]
                        & (data[f"{obj}_num_hcal"] == 0)
                        & (data[f"{obj}_energy_ecal"] >= 10)
                    )
                    data[f"{obj}_is_photon"] = (
                        data[f"{obj}_is_neutral"]
                        & (data[f"{obj}_num_hcal"] == 0)
                        & (data[f"{obj}_energy_ecal"] >= 10)
                    )
                    data[f"{obj}_is_muon"] = (
                        (data[f"{obj}_num_sihit"] >= 4)
                        & (data[f"{obj}_num_ecal"] >= 10)
                        & (data[f"{obj}_num_hcal"] >= 10)
                        & (data[f"{obj}_num_muon"] >= 4)
                    )

            # ---------------------------------------------
            # Truth selections
            # ---------------------------------------------
            selections = calculate_selections(data, "particle", eval_cfg["selections"])
            data |= selections

            # ---------------------------------------------
            # Bulk metrics
            # ---------------------------------------------
            for name, cfg in eval_cfg["bulk_metrics"].items():
                sel_key = f"{cfg['object_name']}_{cfg['selection']}"
                den_key = f"{cfg['object_name']}_{cfg['denominator']}"
                num_key = f"{cfg['object_name']}_{cfg['numerator']}"
                sel = data[sel_key].bool()
                n = data[den_key][sel].float()
                k = data[num_key][sel].float()
                bulk_metrics[name]["n"] += scalar_sum(n)
                bulk_metrics[name]["k"] += scalar_sum(k)

            # ---------------------------------------------
            # Residual Gaussian histograms
            # ---------------------------------------------
            for name, rcfg in eval_cfg["residual_histograms"].items():
                sel_key = rcfg["selection"]
                field_key = rcfg["field"]
                true_key = rcfg["true_field"]
                pred_key = rcfg["pred_field"]

                # Build minimal data dict for the histogram class
                # NOTE: arrays are flattened to 1D for consistency
                data_for_hist = {
                    sel_key: to_bool_1d(data[sel_key]),
                    field_key: to_numpy_1d(data[field_key]),
                    "residual": to_numpy_1d((data[pred_key] - data[true_key]).float()),
                }
                gauss_hists[name].fill(data_for_hist)

            # ---------------------------------------------
            # Efficiency/purity Poisson histograms
            # ---------------------------------------------
            for name, hcfg in eval_cfg["histograms"].items():
                field_key = f"{hcfg['object_name']}_{hcfg['field']}"
                sel_key = f"{hcfg['object_name']}_{hcfg['selection']}"
                num_key = f"{hcfg['object_name']}_{hcfg['numerator']}"
                den_key = f"{hcfg['object_name']}_{hcfg['denominator']}"

                data_for_hist = {
                    field_key: to_numpy_1d(data[field_key]),
                    sel_key: to_bool_1d(data[sel_key]),
                    num_key: to_numpy_1d(data[num_key]),
                    den_key: to_numpy_1d(data[den_key]),
                }
                poisson_hists[name].fill(data_for_hist)

            # Track largest event occupancy
            num_particles = float(data["event_num_particle"].detach().cpu().item())
            if num_particles > largest_num_particles:
                largest_num_particles = num_particles
                print(i, sample_id, largest_num_particles)

            if MAX_EVENTS is not None and i + 1 >= MAX_EVENTS:
                break

    # -----------------------------------------------------------------
    # Print bulk metrics
    # -----------------------------------------------------------------
    for name, _cfg in eval_cfg["bulk_metrics"].items():
        n = bulk_metrics[name]["n"]
        k = bulk_metrics[name]["k"]
        pct = 100.0 * (k / n) if n > 0 else 0.0
        print(f"{name}: {k:.0f}/{n:.0f} ({pct:.3f}%)")

    # -----------------------------------------------------------------
    # Plots (eff/pur)
    # -----------------------------------------------------------------
    for name, cfg in eval_cfg["histogram_plots"].items():
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 4)

        for _item_name, item_cfg in cfg["items"].items():
            ph = poisson_hists[item_cfg["histogram"]]
            n_binned = ph.n
            k_binned = ph.k
            p = np.divide(k_binned, n_binned, out=np.zeros_like(k_binned), where=n_binned > 0)
            p_err = bayesian_binomial_error(k_binned, n_binned)

            # bins name comes from the referenced histogram
            hcfg = eval_cfg["histograms"][item_cfg["histogram"]]
            plot_hist_to_ax(
                ax,
                p,
                bins[hcfg["bins"]],
                p_err,
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
        ax.set_ylim(max(ymin, 0.05), min(ymax, 1.01))

        fig.savefig(hists_dir / Path(f"{name}.png"))

    # -----------------------------------------------------------------
    # Plots (residuals)
    # -----------------------------------------------------------------
    for name, cfg in eval_cfg["residual_histogram_plots"].items():
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(6, 4)

        for _item_name, item_cfg in cfg["items"].items():
            gh = gauss_hists[item_cfg["histogram"]]
            hcfg = eval_cfg["residual_histograms"][item_cfg["histogram"]]

            plot_hist_to_ax(
                axs[0],
                gh.mu,
                bins[hcfg["bins"]],
                label=item_cfg.get("label"),
                color=item_cfg["color"],
                linestyle=item_cfg.get("linestyle"),
            )
            plot_hist_to_ax(
                axs[1],
                gh.sigma,
                bins[hcfg["bins"]],
                label=item_cfg.get("label"),
                color=item_cfg["color"],
                linestyle=item_cfg.get("linestyle"),
            )

        axs[1].set_xlabel(cfg["xlabel"])
        axs[0].set_ylabel(f"Mean {cfg['ylabel']}")
        axs[1].set_ylabel(f"S.D. {cfg['ylabel']}")
        axs[0].set_xscale(cfg["scale"])
        axs[1].set_xscale(cfg["scale"])
        axs[1].set_yscale("log")
        axs[0].legend(fontsize=8)
        axs[0].grid(zorder=0, alpha=0.25, linestyle="--")
        axs[1].grid(zorder=0, alpha=0.25, linestyle="--")

        fig.tight_layout()
        fig.savefig(hists_dir / Path(f"{name}.png"))


if __name__ == "__main__":
    main()
