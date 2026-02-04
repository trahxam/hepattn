from __future__ import annotations

from collections import defaultdict
import json
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


def event_filename_to_event_id(event_filename: Path) -> int:
    id_parts = str(event_filename.stem.replace("_condor", "")).split("_")
    job_id = id_parts[-3]
    proc_id = id_parts[-2]
    event_id = id_parts[-1]
    return int(job_id + proc_id.zfill(4) + event_id.zfill(4))


def resolve_sample_id_to_file(test_dir: Path, sample_ids: list[int], cache_path: Path | None = None) -> dict[int, str]:
    sample_id_to_file: dict[int, str] = {}

    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            for sample_id in sample_ids:
                filename = cached.get(str(sample_id))
                if filename and Path(filename).is_file():
                    sample_id_to_file[sample_id] = filename
        except json.JSONDecodeError:
            pass

    unresolved_sample_ids = [sample_id for sample_id in sample_ids if sample_id not in sample_id_to_file]
    if not unresolved_sample_ids:
        return sample_id_to_file

    # Build a one-time index of first-level test directories keyed by (job_id, proc_id).
    dir_index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for subdir in test_dir.iterdir():
        if not subdir.is_dir():
            continue
        stem = subdir.name.replace("_condor", "")
        parts = stem.split("_")
        if len(parts) < 2:
            continue

        job_id = parts[-2]
        proc_id = parts[-1]
        dir_index[(job_id, proc_id)].append(subdir)

        # Also index non-zero-padded proc IDs for robust lookup.
        if proc_id.isdigit():
            dir_index[(job_id, str(int(proc_id)))].append(subdir)

    for sample_id in tqdm(unresolved_sample_ids, desc="Resolving sample files"):
        sample_id_str = str(sample_id)
        job_id = sample_id_str[:-8]
        proc_id_4 = sample_id_str[-8:-4]
        event_id_4 = sample_id_str[-4:]
        event_id = str(int(event_id_4))

        proc_candidates = [proc_id_4]
        if proc_id_4.isdigit():
            proc_candidates.append(str(int(proc_id_4)))

        matched = False
        for proc_id in proc_candidates:
            for subdir in dir_index.get((job_id, proc_id), []):
                candidate = subdir / f"{subdir.name}_{event_id}.npz"
                if candidate.is_file():
                    sample_id_to_file[sample_id] = str(candidate)
                    matched = True
                    break

                candidate_padded = subdir / f"{subdir.name}_{event_id_4}.npz"
                if candidate_padded.is_file():
                    sample_id_to_file[sample_id] = str(candidate_padded)
                    matched = True
                    break

                # Rare fallback for naming deviations.
                for filename in subdir.glob(f"*_{event_id}.npz"):
                    try:
                        if event_filename_to_event_id(filename) == sample_id:
                            sample_id_to_file[sample_id] = str(filename)
                            matched = True
                            break
                    except (ValueError, IndexError):
                        continue
                if matched:
                    break
            if matched:
                break

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {str(sample_id): filename for sample_id, filename in sample_id_to_file.items()}
        cache_path.write_text(json.dumps(cache_data, sort_keys=True))

    return sample_id_to_file


# ---------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------

EVAL_CONFIG_NAME = "eval_tracking"
EVAL_FILE_PATH = Path(
    #"/share/rcifdata/maxhart/hepattn/logs/CLD_5_320_10MeV_charged_tracking_20251127-T105254/ckpts/epoch=001-train_loss=1.04790_prepped_new_eval.h5"
    "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/All_20260202-T181849/ckpts/epoch=000-val_loss=17.18311_test_eval.h5"
)

CONFIG_PATH = EVAL_FILE_PATH.parent.parent / "config.yaml"

HITS = ["vtxd", "trkr", "ecal", "hcal", "muon"]
PRED_OBJECTS = ["particle", "pandora", "sitrack", "flow"]

# Optional early-stop; set None for full run
MAX_EVENTS: int | None = 1000


def main() -> None:
    # -----------------------------------------------------------------
    # Load configs
    # -----------------------------------------------------------------
    data_cfg = yaml.safe_load(CONFIG_PATH.read_text())["data"]
    data_cfg["num_workers"] = 0
    data_cfg["batch_size"] = 1

    with h5py.File(EVAL_FILE_PATH, "r") as f:
        sample_ids = [int(sample_id) for sample_id in list(f.keys())[:MAX_EVENTS]]

    if len(sample_ids) == 0:
        raise ValueError(f"No sample IDs found in eval file: {EVAL_FILE_PATH}")

    # Avoid recursively scanning the full test tree in setup(stage="test").
    data_cfg["fast_file_discovery"] = True
    data_cfg["num_test"] = 1

    eval_cfg_path = Path(f"src/hepattn/experiments/cld/eval_configs/{EVAL_CONFIG_NAME}.yaml")
    eval_cfg = yaml.safe_load(eval_cfg_path.read_text())["eval"]

    # -----------------------------------------------------------------
    # Output dirs
    # ----------------------------------------------------------------
    plot_root = CONFIG_PATH.parent / EVAL_CONFIG_NAME
    plot_root.mkdir(parents=True, exist_ok=True)

    hists_dir = plot_root / "histograms"
    hists_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Data module / dataset
    # -----------------------------------------------------------------
    print("Setting up data module...")
    datamodule = CLDDataModule(**data_cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.test_dataloader().dataset  # type: ignore[assignment]
    mapping_cache_path = plot_root / "sample_id_to_file.json"
    sample_id_to_file = resolve_sample_id_to_file(Path(data_cfg["test_dir"]), sample_ids, mapping_cache_path)
    if len(sample_id_to_file) != len(sample_ids):
        missing = sorted(set(sample_ids) - set(sample_id_to_file))
        raise FileNotFoundError(
            f"Failed to resolve {len(missing)} sample IDs from {data_cfg['test_dir']}. "
            f"First missing IDs: {missing[:5]}"
        )
    dataset.event_ids_to_event_filenames = sample_id_to_file
    print(f"Resolved {len(sample_id_to_file):,} eval sample files")

    # -----------------------------------------------------------------
    # Matcher and binning
    # -----------------------------------------------------------------
    matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)

    bin_types = {"linear": np.linspace, "log": np.geomspace}
    bins: dict[str, np.ndarray] = {name: bin_types[cfg["scale"]](cfg["min"], cfg["max"], cfg["num"]) for name, cfg in eval_cfg["bins"].items()}

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
    bulk_metrics: dict[str, dict[str, float]] = {name: {"n": 0.0, "k": 0.0} for name in eval_cfg["bulk_metrics"]}

    # -----------------------------------------------------------------
    # Event loop
    # -----------------------------------------------------------------
    largest_num_particles = 0.0

    with h5py.File(EVAL_FILE_PATH, "r") as f:
        for i, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids)):
            # ---------------------------------------------
            # Load preds/outputs (final layer only)
            # ---------------------------------------------
            preds = f[f"{sample_id}/preds/final/reco"]
            outs = f[f"{sample_id}/outputs/final/reco"]

            data: dict[str, Any] = {}
            data["flow_logit"] = torch.from_numpy(outs["flow_logit"][:])
            if "flow_valid" in preds:
                data["flow_valid"] = torch.from_numpy(preds["flow_valid"][:]).bool()
            else:
                # Fallback for older files where only logits are available.
                if data["flow_logit"].dim() == 3 and data["flow_logit"].shape[-1] > 1:
                    data["flow_valid"] = data["flow_logit"].argmax(-1) != 0
                else:
                    data["flow_valid"] = data["flow_logit"].sigmoid() >= 0.5

            for hit in HITS:
                key = f"flow_{hit}_valid"
                if key in preds:
                    data[f"flow_{hit}_valid"] = torch.from_numpy(preds[key][:])

            # ---------------------------------------------
            # Load and prepare the sample
            # ---------------------------------------------
            sample = dataset.load_sample(sample_id)
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
                        data[f"{obj}_{hit}_energy"] = data[key].float() * data[f"{hit}_energy"].unsqueeze(-2)
                        data[f"{obj}_energy_{hit}"] = data[f"{obj}_{hit}_energy"].sum(-1)

            for obj in PRED_OBJECTS:
                data[f"{obj}_sihit_valid"] = torch.cat((data[f"{obj}_vtxd_valid"], data[f"{obj}_trkr_valid"]), dim=-1)

            # ---------------------------------------------
            # Matching and binary metrics
            # ---------------------------------------------
            for obj in PRED_OBJECTS:
                costs = calc_cost(data, "particle", obj, eval_cfg["match_metrics"]["default"])
                data = apply_matching(data, "particle", obj, costs, matcher)

                eff_metrics = calc_binary_reco_metrics(data, "particle", obj, eval_cfg["binary_metrics"])
                pur_metrics = calc_binary_reco_metrics(data, obj, "particle", eval_cfg["binary_metrics"])
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
                        data[f"{obj}_is_charged"] & (data[f"{obj}_energy_hcal"] >= 0.1) & (data[f"{obj}_energy_ecal"] >= 0.1)
                    )
                    data[f"{obj}_is_neutral_hadron"] = (
                        data[f"{obj}_is_neutral"] & (data[f"{obj}_energy_hcal"] >= 0.1) & (data[f"{obj}_energy_ecal"] >= 0.1)
                    )
                    data[f"{obj}_is_electron"] = data[f"{obj}_is_charged"] & (data[f"{obj}_num_hcal"] == 0) & (data[f"{obj}_energy_ecal"] >= 10)
                    data[f"{obj}_is_photon"] = data[f"{obj}_is_neutral"] & (data[f"{obj}_num_hcal"] == 0) & (data[f"{obj}_energy_ecal"] >= 10)
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
            largest_num_particles = max(largest_num_particles, num_particles)

    # -----------------------------------------------------------------
    # Print bulk metrics
    # -----------------------------------------------------------------
    for name in eval_cfg["bulk_metrics"]:
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

        for item_cfg in cfg["items"].values():
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

        for item_cfg in cfg["items"].values():
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
