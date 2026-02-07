from __future__ import annotations

from collections import defaultdict, OrderedDict
from copy import deepcopy
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
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr.reshape(-1)


def to_bool_1d(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    return arr.reshape(-1)


def scalar_sum(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().sum().item())
    if isinstance(x, np.ndarray):
        return float(x.sum().item())
    return float(np.asarray(x).sum())


def accumulate_comet_style_metrics(
    data: dict[str, Any],
    obj: str,
    hit: str,
    wps: tuple[float, ...],
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
    eps: float = 1e-8,
) -> None:
    pred_key = f"{obj}_{hit}_valid"
    true_key = f"particle_{hit}_valid"
    if pred_key not in data or true_key not in data:
        return
    if f"{obj}_valid" not in data or "particle_valid" not in data:
        return

    pred_hit_masks = data[pred_key].bool()
    true_hit_masks = data[true_key].bool()
    pred_valid = data[f"{obj}_valid"].bool() & (pred_hit_masks.sum(-1) > 0)
    true_valid = data["particle_valid"].bool() & (true_hit_masks.sum(-1) > 0)

    pred_hit_masks = pred_hit_masks & pred_valid.unsqueeze(-1)
    true_hit_masks = true_hit_masks & true_valid.unsqueeze(-1)

    hit_tp = (pred_hit_masks & true_hit_masks).sum(-1).float()
    hit_p = pred_hit_masks.sum(-1).float()
    hit_t = true_hit_masks.sum(-1).float()
    both_valid = true_valid & pred_valid

    hit_eff = hit_tp / (hit_t + eps)
    hit_pur = hit_tp / (hit_p + eps)
    true_valid_count = true_valid.float().sum(-1) + eps
    pred_valid_count = pred_valid.float().sum(-1) + eps

    for wp in wps:
        effs = (hit_eff >= wp) & both_valid
        purs = (hit_pur >= wp) & both_valid

        eff = (effs.float().sum(-1) / true_valid_count).mean().item()
        pur = (purs.float().sum(-1) / pred_valid_count).mean().item()

        eff_key = f"p{wp}_{hit}_eff"
        pur_key = f"p{wp}_{hit}_pur"
        metric_sums[eff_key] += eff
        metric_sums[pur_key] += pur
        metric_counts[eff_key] += 1
        metric_counts[pur_key] += 1


def event_filename_to_event_id(event_filename: Path) -> int:
    id_parts = str(event_filename.stem.replace("_condor", "")).split("_")
    job_id = id_parts[-3]
    proc_id = id_parts[-2]
    event_id = id_parts[-1]
    return int(job_id + proc_id.zfill(4) + event_id.zfill(4))

_DIR_INDEX_CACHE: dict[Path, dict[tuple[str, str], list[Path]]] = {}


def resolve_sample_id_to_file(test_dir: Path, sample_ids: list[int], cache_path: Path | None = None) -> dict[int, str]:
    test_dir = test_dir.resolve()
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
    dir_index = _DIR_INDEX_CACHE.get(test_dir)
    if dir_index is None:
        dir_index = defaultdict(list)
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
        _DIR_INDEX_CACHE[test_dir] = dir_index

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

EVAL_CONFIG_NAME = "eval_tracking_final"
EVAL_RUNS = [
    {
        "path": Path(
            "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/All_20260202-T181849/ckpts/epoch=000-val_loss=17.18311_test_eval.h5"
        ),
        "label": "Combined",
        "color": "mediumvioletred",
    },
    {
        "path": Path(
            "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/TrackingFixed_20260205-T000653/ckpts/epoch=000-val_loss=7.40857_test_eval.h5"
        ),
        "label": "Tracking",
        "color": "dodgerblue",
    },
]

HITS = ["vtxd", "trkr", "ecal", "hcal", "muon"]
HITS_NO_MUON = ("vtxd", "trkr", "ecal", "hcal")
COMET_HITS = ("sihit", "vtxd", "trkr")
COMET_WORKING_POINTS = (0.5, 0.75, 1.0)
# Set this to ["flow"] to evaluate only model outputs and omit pandora/sitrack baselines.
PRED_OBJECTS = ["flow"]
# Matching between predicted objects and truth is driven by the eval config:
# `eval.match_metrics.default` (e.g. set it to use only `sihit` and `dice`).
#
# Binary metrics can either use the matched alignment ("matched") or
# skip matching and consider a metric satisfied if any reco object
# matches a truth object ("any").
BINARY_METRICS_MATCH_MODE = "matched"  # "matched" or "any"
#
# If true, use flow class logits to define flow_is_* flags used by purity metrics.
USE_FLOW_CLASS_LOGITS_FOR_PID = True

# Optional early-stop; set None for full run
MAX_EVENTS: int | None = 1000
# Cache prepared inputs/targets across runs to avoid reloading data.
# Set to 0 to disable. Defaults to MAX_EVENTS if set, otherwise disabled.
BASE_DATA_CACHE_MAX_EVENTS = MAX_EVENTS if MAX_EVENTS is not None else 0
CACHE_BASE_DATA = True
SAVE_COMPARISON_PLOTS = True
COMPARISON_PLOT_DIR = Path("src/hepattn/experiments/cld/plots/eval_tracking_compare")
MIN_BIN_COUNT = 16
TRIM_XRANGE_TO_VALID_BINS = True


def filter_eval_config(eval_cfg: dict[str, Any], pred_objects: list[str]) -> dict[str, Any]:
    known_pred_objects = {"flow", "sitrack", "pandora"}
    excluded_objects = known_pred_objects - set(pred_objects)

    def excluded(name: str) -> bool:
        return any(name.startswith(f"{obj}_") for obj in excluded_objects)

    filtered: dict[str, Any] = dict(eval_cfg)

    for section in ("bulk_metrics", "histograms", "residual_histograms"):
        entries = eval_cfg.get(section, {})
        filtered[section] = {name: cfg for name, cfg in entries.items() if not excluded(name)}

    filtered_histograms = filtered.get("histograms", {})
    filtered_residual_histograms = filtered.get("residual_histograms", {})

    histogram_plots: dict[str, Any] = {}
    for name, cfg in eval_cfg.get("histogram_plots", {}).items():
        items = {k: v for k, v in cfg.get("items", {}).items() if v.get("histogram") in filtered_histograms}
        if items:
            histogram_plots[name] = {**cfg, "items": items}
    filtered["histogram_plots"] = histogram_plots

    residual_histogram_plots: dict[str, Any] = {}
    for name, cfg in eval_cfg.get("residual_histogram_plots", {}).items():
        items = {k: v for k, v in cfg.get("items", {}).items() if v.get("histogram") in filtered_residual_histograms}
        if items:
            residual_histogram_plots[name] = {**cfg, "items": items}
    filtered["residual_histogram_plots"] = residual_histogram_plots

    return filtered


def style_eval_config(eval_cfg: dict[str, Any], run_label: str, run_color: str) -> dict[str, Any]:
    """Override flow plot styling for a specific training run."""
    styled = deepcopy(eval_cfg)
    for plots_key in ("histogram_plots", "residual_histogram_plots"):
        for _, cfg in styled.get(plots_key, {}).items():
            for _, item_cfg in cfg.get("items", {}).items():
                histogram_name = item_cfg.get("histogram", "")
                if histogram_name.startswith("flow_"):
                    item_cfg["color"] = run_color
                    item_cfg["label"] = run_label
    return styled


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_bulk_metrics_latex_table(
    run_summaries: list[dict[str, Any]],
    metric_names: list[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(run_summaries) == 0 or len(metric_names) == 0:
        out_path.write_text("% No bulk metrics available.\n")
        return

    num_runs = len(run_summaries)
    colspec = "l" + ("c" * num_runs)
    header_cols = " & ".join(latex_escape(str(run["label"])) for run in run_summaries)

    lines = [
        "% Auto-generated by eval_old.py",
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        rf"Metric & {header_cols} \\",
        r"\hline",
    ]

    for metric_name in metric_names:
        cells = [latex_escape(metric_name)]
        for run in run_summaries:
            metric = run["bulk_metrics"].get(metric_name)
            if metric is None or metric["n"] <= 0:
                cells.append("--")
            else:
                cells.append(f"{metric['pct']:.3f}\\%")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Bulk efficiency and purity metrics (\%).}",
            r"\label{tab:cld_bulk_metrics}",
            r"\end{table}",
            "",
        ]
    )

    out_path.write_text("\n".join(lines))


def main() -> None:
    eval_cfg_path = Path(f"src/hepattn/experiments/cld/eval_configs/{EVAL_CONFIG_NAME}.yaml")
    base_eval_cfg = yaml.safe_load(eval_cfg_path.read_text())["eval"]
    eval_cfg = filter_eval_config(base_eval_cfg, PRED_OBJECTS)
    run_summaries: list[dict[str, Any]] = []
    eval_objects = list(dict.fromkeys(["particle", *PRED_OBJECTS]))
    eval_matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)
    flow_class_id_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
    data_contexts: dict[str, dict[str, Any]] = {}
    cache_enabled = CACHE_BASE_DATA and len(EVAL_RUNS) > 1 and (
        BASE_DATA_CACHE_MAX_EVENTS is None or BASE_DATA_CACHE_MAX_EVENTS > 0
    )

    def get_flow_class_ids(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = flow_class_id_cache.get(key)
        if cached is None:
            charged = torch.tensor([2, 4, 5], device=device, dtype=dtype)
            neutral = torch.tensor([1, 3], device=device, dtype=dtype)
            cached = (charged, neutral)
            flow_class_id_cache[key] = cached
        return cached

    for run in EVAL_RUNS:
        eval_file_path = Path(run["path"])
        eval_cfg_run = style_eval_config(eval_cfg, str(run["label"]), str(run["color"]))
        config_path = eval_file_path.parent.parent / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found for eval file: {config_path}")

        run_cfg = yaml.safe_load(config_path.read_text())
        data_cfg = dict(run_cfg["data"])
        data_cfg["num_workers"] = 0
        data_cfg["batch_size"] = 1
        data_cfg["fast_file_discovery"] = True
        data_cfg["num_test"] = 1

        with h5py.File(eval_file_path, "r") as f:
            sample_ids = [int(sample_id) for sample_id in list(f.keys())[:MAX_EVENTS]]

        if len(sample_ids) == 0:
            print(f"Skipping empty eval file: {eval_file_path}")
            continue

        # -----------------------------------------------------------------
        # Output dirs
        # ----------------------------------------------------------------
        plot_root = config_path.parent / EVAL_CONFIG_NAME
        plot_root.mkdir(parents=True, exist_ok=True)

        hists_dir = plot_root / "histograms"
        hists_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------------
        # Data module / dataset
        # -----------------------------------------------------------------
        print(f"\nEvaluating: {eval_file_path}")
        data_key = json.dumps(data_cfg, sort_keys=True, default=str)
        data_context = data_contexts.get(data_key)
        if data_context is None:
            print("Setting up data module...")
            datamodule = CLDDataModule(**data_cfg)
            datamodule.setup(stage="test")
            dataset = datamodule.test_dataloader().dataset  # type: ignore[assignment]
            base_data_cache = OrderedDict() if cache_enabled else None
            data_context = {"dataset": dataset, "base_data_cache": base_data_cache}
            data_contexts[data_key] = data_context
        else:
            print("Reusing data module...")
            dataset = data_context["dataset"]
            base_data_cache = data_context["base_data_cache"]
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
        bin_types = {"linear": np.linspace, "log": np.geomspace}
        bins: dict[str, np.ndarray] = {
            name: bin_types[cfg["scale"]](cfg["min"], cfg["max"], cfg["num"]) for name, cfg in eval_cfg_run["bins"].items()
        }

        # -----------------------------------------------------------------
        # Histogram objects
        # -----------------------------------------------------------------
        poisson_hists: dict[str, PoissonHistogram] = {}
        for name, cfg in eval_cfg_run["histograms"].items():
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

        gauss_hists: dict[str, GaussianHistogram] = {}
        for name, cfg in eval_cfg_run["residual_histograms"].items():
            gauss_hists[name] = GaussianHistogram(
                field=cfg["field"],
                bins=bins[cfg["bins"]],
                selection=cfg["selection"],
                values="residual",
            )

        poisson_hist_defs: list[tuple[str, str, str, str, str]] = []
        for name, cfg in eval_cfg_run["histograms"].items():
            field_key = f"{cfg['object_name']}_{cfg['field']}"
            sel_key = f"{cfg['object_name']}_{cfg['selection']}"
            num_key = f"{cfg['object_name']}_{cfg['numerator']}"
            den_key = f"{cfg['object_name']}_{cfg['denominator']}"
            poisson_hist_defs.append((name, field_key, sel_key, num_key, den_key))

        gauss_hist_defs: list[tuple[str, str, str, str, str]] = []
        for name, cfg in eval_cfg_run["residual_histograms"].items():
            gauss_hist_defs.append((name, cfg["selection"], cfg["field"], cfg["true_field"], cfg["pred_field"]))

        bulk_metric_defs: list[tuple[str, str, str, str]] = []
        for name, cfg in eval_cfg_run["bulk_metrics"].items():
            sel_key = f"{cfg['object_name']}_{cfg['selection']}"
            den_key = f"{cfg['object_name']}_{cfg['denominator']}"
            num_key = f"{cfg['object_name']}_{cfg['numerator']}"
            bulk_metric_defs.append((name, sel_key, den_key, num_key))

        match_metrics = eval_cfg_run.get("match_metrics", {}).get("default", {})
        binary_metrics_cfg = eval_cfg_run["binary_metrics"]
        selections_cfg = eval_cfg_run["selections"]

        bulk_metrics: dict[str, dict[str, float]] = {name: {"n": 0.0, "k": 0.0} for name in eval_cfg_run["bulk_metrics"]}
        comet_metric_sums: dict[str, float] = defaultdict(float)
        comet_metric_counts: dict[str, int] = defaultdict(int)

        largest_num_particles = 0.0

        def load_base_data(sample_id: int) -> dict[str, Any]:
            sample = dataset.load_sample(sample_id)
            if sample is None:
                raise RuntimeError(f"Failed to load sample {sample_id} from dataset.")
            inputs, targets = dataset.prep_sample(sample)
            base_data: dict[str, Any] = {}
            base_data.update(targets)
            base_data.update(inputs)

            # Some datasets/configs may not explicitly include sihit as an input stream;
            # build it from vtxd+trkr so sihit-based matching works consistently.
            if "sihit_valid" not in base_data and "vtxd_valid" in base_data and "trkr_valid" in base_data:
                base_data["sihit_valid"] = torch.cat((base_data["vtxd_valid"], base_data["trkr_valid"]), dim=-1)
            return base_data

        with h5py.File(eval_file_path, "r") as f, torch.inference_mode():
            for i, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids)):
                # ---------------------------------------------
                # Load preds/outputs (final layer only)
                # ---------------------------------------------
                preds = f[f"{sample_id}/preds/final/reco"]
                outs = f[f"{sample_id}/outputs/final/reco"]

                data: dict[str, Any] = {}
                flow_logit = torch.from_numpy(outs["flow_logit"][:])
                data["flow_logit"] = flow_logit
                if "flow_valid" in preds:
                    data["flow_valid"] = torch.from_numpy(preds["flow_valid"][:]).bool()
                else:
                    # Fallback for older files where only logits are available.
                    if flow_logit.dim() == 3 and flow_logit.shape[-1] > 1:
                        data["flow_valid"] = flow_logit.argmax(-1) != 0
                    else:
                        data["flow_valid"] = flow_logit.sigmoid() >= 0.5

                if USE_FLOW_CLASS_LOGITS_FOR_PID:
                    flow_class_idx = flow_logit.argmax(-1)
                    data["flow_class_idx"] = flow_class_idx
                    data["flow_is_null"] = flow_class_idx == 0
                    data["flow_is_neutral_hadron"] = flow_class_idx == 1
                    data["flow_is_charged_hadron"] = flow_class_idx == 2
                    data["flow_is_photon"] = flow_class_idx == 3
                    data["flow_is_electron"] = flow_class_idx == 4
                    data["flow_is_muon"] = flow_class_idx == 5
                    charged_classes, neutral_classes = get_flow_class_ids(flow_class_idx.device, flow_class_idx.dtype)
                    data["flow_is_charged"] = torch.isin(flow_class_idx, charged_classes)
                    data["flow_is_neutral"] = torch.isin(flow_class_idx, neutral_classes)

                for hit in HITS:
                    key = f"flow_{hit}_valid"
                    if key in preds:
                        data[f"flow_{hit}_valid"] = torch.from_numpy(preds[key][:])

                # ---------------------------------------------
                # Load and prepare the sample
                # ---------------------------------------------
                if base_data_cache is None:
                    base_data = load_base_data(sample_id)
                else:
                    base_data = base_data_cache.get(sample_id)
                    if base_data is not None:
                        base_data_cache.move_to_end(sample_id)
                    else:
                        base_data = load_base_data(sample_id)
                        base_data_cache[sample_id] = base_data
                        base_data_cache.move_to_end(sample_id)
                        if BASE_DATA_CACHE_MAX_EVENTS is None or BASE_DATA_CACHE_MAX_EVENTS > 0:
                            while BASE_DATA_CACHE_MAX_EVENTS is not None and len(base_data_cache) > BASE_DATA_CACHE_MAX_EVENTS:
                                base_data_cache.popitem(last=False)
                data.update(base_data)

                # Align predicted hit slots to truth hit counts
                for hit in HITS:
                    flow_key = f"flow_{hit}_valid"
                    truth_key = f"{hit}_valid"
                    if flow_key in data and truth_key in data:
                        n_truth = data[truth_key].shape[-1]
                        data[flow_key] = data[flow_key][:, :, :n_truth]
                        flow_logit_key = f"flow_{hit}_logit"
                        if flow_logit_key in data:
                            data[flow_logit_key] = data[flow_logit_key][:, :, :n_truth]
                    elif truth_key in data and "flow_valid" in data:
                        # Some trainings (e.g. tracking-only) do not output all detector masks.
                        # Fill missing masks with all-false tensors so generic eval configs still run.
                        batch_size, num_queries = data["flow_valid"].shape
                        n_truth = data[truth_key].shape[-1]
                        data[flow_key] = torch.zeros(
                            (batch_size, num_queries, n_truth),
                            dtype=torch.bool,
                            device=data[truth_key].device,
                        )

                # ---------------------------------------------
                # Per-object bookkeeping
                # ---------------------------------------------
                for obj in eval_objects:
                    if f"{obj}_valid" not in data:
                        continue
                    data[f"event_num_{obj}"] = data[f"{obj}_valid"].float().sum(-1)
                    for hit in HITS_NO_MUON:
                        key = f"{obj}_{hit}_valid"
                        if key in data:
                            data[key] = data[key] & data[f"{obj}_valid"].unsqueeze(-1)

                for obj in eval_objects:
                    for hit in ("ecal", "hcal"):
                        key = f"{obj}_{hit}_valid"
                        if key in data:
                            data[f"{obj}_{hit}_energy"] = data[key].float() * data[f"{hit}_energy"].unsqueeze(-2)
                            data[f"{obj}_energy_{hit}"] = data[f"{obj}_{hit}_energy"].sum(-1)

                for obj in eval_objects:
                    if f"{obj}_vtxd_valid" in data and f"{obj}_trkr_valid" in data:
                        data[f"{obj}_sihit_valid"] = torch.cat((data[f"{obj}_vtxd_valid"], data[f"{obj}_trkr_valid"]), dim=-1)

                    for hit in HITS:
                        key = f"{obj}_{hit}_valid"
                        if key in data:
                            data[f"{obj}_num_{hit}"] = data[key].sum(-1)

                    if f"{obj}_num_vtxd" in data and f"{obj}_num_trkr" in data:
                        data[f"{obj}_num_sihit"] = data[f"{obj}_num_vtxd"] + data[f"{obj}_num_trkr"]

                # Comet-style tracking metrics (before extra eval-time rematching).
                for hit in COMET_HITS:
                    accumulate_comet_style_metrics(
                        data,
                        "flow",
                        hit,
                        COMET_WORKING_POINTS,
                        comet_metric_sums,
                        comet_metric_counts,
                    )

                # ---------------------------------------------
                # Matching and binary metrics
                # ---------------------------------------------
                for obj in PRED_OBJECTS:
                    if BINARY_METRICS_MATCH_MODE == "matched" and match_metrics:
                        costs = calc_cost(data, "particle", obj, match_metrics)
                        if costs is not None:
                            data = apply_matching(data, "particle", obj, costs, eval_matcher)

                    eff_metrics = calc_binary_reco_metrics(
                        data,
                        "particle",
                        obj,
                        binary_metrics_cfg,
                        match_mode=BINARY_METRICS_MATCH_MODE,
                    )
                    pur_metrics = calc_binary_reco_metrics(
                        data,
                        obj,
                        "particle",
                        binary_metrics_cfg,
                        match_mode=BINARY_METRICS_MATCH_MODE,
                    )
                    data |= eff_metrics
                    data |= pur_metrics

                # ---------------------------------------------
                # Truth selections
                # ---------------------------------------------
                selections = calculate_selections(data, "particle", selections_cfg)
                data |= selections

                np_cache: dict[str, np.ndarray] = {}
                bool_cache: dict[str, np.ndarray] = {}

                def get_np(key: str) -> np.ndarray:
                    cached = np_cache.get(key)
                    if cached is None:
                        cached = to_numpy_1d(data[key])
                        np_cache[key] = cached
                    return cached

                def get_bool(key: str) -> np.ndarray:
                    cached = bool_cache.get(key)
                    if cached is None:
                        cached = to_bool_1d(data[key])
                        bool_cache[key] = cached
                    return cached

                # ---------------------------------------------
                # Bulk metrics
                # ---------------------------------------------
                for name, sel_key, den_key, num_key in bulk_metric_defs:
                    sel = data[sel_key].bool()
                    n = data[den_key][sel].float()
                    k = data[num_key][sel].float()
                    bulk_metrics[name]["n"] += scalar_sum(n)
                    bulk_metrics[name]["k"] += scalar_sum(k)

                # ---------------------------------------------
                # Residual Gaussian histograms
                # ---------------------------------------------
                for name, sel_key, field_key, true_key, pred_key in gauss_hist_defs:
                    data_for_hist = {
                        sel_key: get_bool(sel_key),
                        field_key: get_np(field_key),
                        "residual": get_np(pred_key) - get_np(true_key),
                    }
                    gauss_hists[name].fill(data_for_hist)

                # ---------------------------------------------
                # Efficiency/purity Poisson histograms
                # ---------------------------------------------
                for name, field_key, sel_key, num_key, den_key in poisson_hist_defs:
                    data_for_hist = {
                        field_key: get_np(field_key),
                        sel_key: get_bool(sel_key),
                        num_key: get_np(num_key),
                        den_key: get_np(den_key),
                    }
                    poisson_hists[name].fill(data_for_hist)

                # Track largest event occupancy
                num_particles = float(data["particle_valid"].float().sum(-1).detach().cpu().item())
                largest_num_particles = max(largest_num_particles, num_particles)

        # -----------------------------------------------------------------
        # Print bulk metrics
        # -----------------------------------------------------------------
        print(f"Bulk metrics for {run['label']}:")
        run_bulk_metrics: dict[str, dict[str, float]] = {}
        for name in eval_cfg_run["bulk_metrics"]:
            n = bulk_metrics[name]["n"]
            k = bulk_metrics[name]["k"]
            pct = 100.0 * (k / n) if n > 0 else 0.0
            print(f"{name}: {k:.0f}/{n:.0f} ({pct:.3f}%)")
            run_bulk_metrics[name] = {"n": n, "k": k, "pct": pct}

        print(f"Comet-style metrics for {run['label']}:")
        for metric_name in sorted(comet_metric_sums):
            n = comet_metric_counts.get(metric_name, 0)
            if n == 0:
                continue
            print(f"{metric_name}: {comet_metric_sums[metric_name] / n:.4f}")

        # -----------------------------------------------------------------
        # Plots (eff/pur)
        # -----------------------------------------------------------------
        for name, cfg in eval_cfg_run["histogram_plots"].items():
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)
            valid_union: np.ndarray | None = None

            for item_cfg in cfg["items"].values():
                ph = poisson_hists[item_cfg["histogram"]]
                n_binned = ph.n
                k_binned = ph.k
                p = np.divide(k_binned, n_binned, out=np.zeros_like(k_binned), where=n_binned > 0)
                p_err = bayesian_binomial_error(k_binned, n_binned)
                valid = n_binned >= MIN_BIN_COUNT
                p = p.copy()
                p_err = p_err.copy()
                p[~valid] = np.nan
                p_err[~valid] = np.nan
                valid_union = valid if valid_union is None else (valid_union | valid)

                hcfg = eval_cfg_run["histograms"][item_cfg["histogram"]]
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
            item0 = next(iter(cfg["items"].values()))
            bins_key = eval_cfg_run["histograms"][item0["histogram"]]["bins"]
            plot_bins = bins[bins_key]
            ax.set_xscale(cfg["scale"])
            if cfg["scale"] == "log":
                positive_edges = plot_bins[np.isfinite(plot_bins) & (plot_bins > 0)]
                if positive_edges.size < 2:
                    plt.close(fig)
                    continue
                if TRIM_XRANGE_TO_VALID_BINS and valid_union is not None and valid_union.any():
                    first = int(np.argmax(valid_union))
                    last = int(len(valid_union) - 1 - np.argmax(valid_union[::-1]))
                    x0 = max(float(plot_bins[first]), float(positive_edges[0]))
                    x1 = float(plot_bins[last + 1])
                    if x1 <= x0:
                        x1 = float(positive_edges[-1])
                    ax.set_xlim(x0, x1)
                else:
                    ax.set_xlim(float(positive_edges[0]), float(positive_edges[-1]))
            elif TRIM_XRANGE_TO_VALID_BINS and valid_union is not None and valid_union.any():
                first = int(np.argmax(valid_union))
                last = int(len(valid_union) - 1 - np.argmax(valid_union[::-1]))
                ax.set_xlim(plot_bins[first], plot_bins[last + 1])
            ax.legend(fontsize=8)
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(max(ymin, 0.05), min(ymax, 1.01))

            fig.savefig(hists_dir / Path(f"{name}.png"))

        # -----------------------------------------------------------------
        # Plots (residuals)
        # -----------------------------------------------------------------
        for name, cfg in eval_cfg_run["residual_histogram_plots"].items():
            fig, axs = plt.subplots(2, 1)
            fig.set_size_inches(6, 4)

            for item_cfg in cfg["items"].values():
                gh = gauss_hists[item_cfg["histogram"]]
                hcfg = eval_cfg_run["residual_histograms"][item_cfg["histogram"]]

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

        run_summaries.append(
            {
                "label": str(run["label"]),
                "color": str(run["color"]),
                "bins": bins,
                "bulk_metrics": run_bulk_metrics,
                "poisson_hists": {name: {"n": hist.n.copy(), "k": hist.k.copy()} for name, hist in poisson_hists.items()},
            }
        )

    if len(run_summaries) > 0:
        bulk_table_dir = COMPARISON_PLOT_DIR if SAVE_COMPARISON_PLOTS else Path(
            Path(EVAL_RUNS[0]["path"]).parent.parent / EVAL_CONFIG_NAME
        )
        bulk_table_path = bulk_table_dir / "bulk_metrics_table.tex"
        write_bulk_metrics_latex_table(
            run_summaries=run_summaries,
            metric_names=list(eval_cfg.get("bulk_metrics", {}).keys()),
            out_path=bulk_table_path,
        )
        print(f"Wrote LaTeX bulk-metrics table to: {bulk_table_path}")

    if SAVE_COMPARISON_PLOTS and len(run_summaries) > 1:
        COMPARISON_PLOT_DIR.mkdir(parents=True, exist_ok=True)

        for name, cfg in eval_cfg["histogram_plots"].items():
            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)
            valid_union: np.ndarray | None = None

            for item_cfg in cfg["items"].values():
                hist_name = item_cfg["histogram"]
                hcfg = eval_cfg["histograms"][hist_name]

                for summary in run_summaries:
                    hist_data = summary["poisson_hists"].get(hist_name)
                    if hist_data is None:
                        continue

                    n_binned = hist_data["n"]
                    k_binned = hist_data["k"]
                    p = np.divide(k_binned, n_binned, out=np.zeros_like(k_binned), where=n_binned > 0)
                    p_err = bayesian_binomial_error(k_binned, n_binned)
                    valid = n_binned >= MIN_BIN_COUNT
                    p = p.copy()
                    p_err = p_err.copy()
                    p[~valid] = np.nan
                    p_err[~valid] = np.nan
                    valid_union = valid if valid_union is None else (valid_union | valid)
                    plot_hist_to_ax(
                        ax,
                        p,
                        summary["bins"][hcfg["bins"]],
                        p_err,
                        label=str(summary["label"]),
                        color=str(summary["color"]),
                        linestyle="-",
                    )

            ax.set_xlabel(cfg["xlabel"])
            ax.set_ylabel(cfg["ylabel"])
            item0 = next(iter(cfg["items"].values()))
            bins_key = eval_cfg["histograms"][item0["histogram"]]["bins"]
            plot_bins = run_summaries[0]["bins"][bins_key]
            ax.set_xscale(cfg["scale"])
            if cfg["scale"] == "log":
                positive_edges = plot_bins[np.isfinite(plot_bins) & (plot_bins > 0)]
                if positive_edges.size < 2:
                    plt.close(fig)
                    continue
                if TRIM_XRANGE_TO_VALID_BINS and valid_union is not None and valid_union.any():
                    first = int(np.argmax(valid_union))
                    last = int(len(valid_union) - 1 - np.argmax(valid_union[::-1]))
                    x0 = max(float(plot_bins[first]), float(positive_edges[0]))
                    x1 = float(plot_bins[last + 1])
                    if x1 <= x0:
                        x1 = float(positive_edges[-1])
                    ax.set_xlim(x0, x1)
                else:
                    ax.set_xlim(float(positive_edges[0]), float(positive_edges[-1]))
            elif TRIM_XRANGE_TO_VALID_BINS and valid_union is not None and valid_union.any():
                first = int(np.argmax(valid_union))
                last = int(len(valid_union) - 1 - np.argmax(valid_union[::-1]))
                ax.set_xlim(plot_bins[first], plot_bins[last + 1])
            ax.legend(fontsize=8)
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(max(ymin, 0.05), min(ymax, 1.01))

            fig.savefig(COMPARISON_PLOT_DIR / Path(f"{name}.png"))


if __name__ == "__main__":
    main()
