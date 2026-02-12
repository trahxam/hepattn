#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.task import CLDTask
from hepattn.models.loss import cost_fns
from hepattn.models.matcher import Matcher


# User settings (edit as needed)
eval_files = [
    {
        "label": "all",
        "path": Path(
            "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/All_20260202-T181849/ckpts/epoch=000-val_loss=17.18311_test_eval.h5"
        ),
    },
    {
        "label": "tracking",
        "path": Path(
            "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/TrackingFixed_20260205-T000653/ckpts/epoch=000-val_loss=7.40857_test_eval.h5"
        ),
    },
]
sample_id = None  # int or None to use first in file
pred_object = "reco"  # name of CLDTask output group in eval file (often "reco")
out_path = None  # set to Path(...) to override default output
bins = 32
default_output_dir = Path("src/hepattn/experiments/cld/plots/data/hit_costs_eval")
hit_cost_variants = [
    ("dice", "mask_dice"),
    ("bce", "mask_bce"),
    ("bce_scaled", "mask_bce_balanced"),
    ("iou", "mask_iou"),
]


def event_filename_to_event_id(event_filename: Path) -> int:
    id_parts = str(event_filename.stem.replace("_condor", "")).split("_")
    job_id = id_parts[-3]
    proc_id = id_parts[-2]
    event_id = id_parts[-1]
    return int(job_id + proc_id.zfill(4) + event_id.zfill(4))


def resolve_sample_id_to_file(test_dir: Path, sample_id: int, cache_path: Path | None = None) -> str:
    sample_id_to_file: dict[int, str] = {}

    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            filename = cached.get(str(sample_id))
            if filename and Path(filename).is_file():
                return filename
        except json.JSONDecodeError:
            pass

    sample_id_str = str(sample_id)
    job_id = sample_id_str[:-8]
    proc_id_4 = sample_id_str[-8:-4]
    event_id_4 = sample_id_str[-4:]
    event_id = str(int(event_id_4))

    # Build a one-time index of first-level test directories keyed by (job_id, proc_id).
    dir_index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for subdir in test_dir.iterdir():
        if not subdir.is_dir():
            continue
        stem = subdir.name.replace("_condor", "")
        parts = stem.split("_")
        if len(parts) < 2:
            continue

        job_id_dir = parts[-2]
        proc_id_dir = parts[-1]
        dir_index[(job_id_dir, proc_id_dir)].append(subdir)
        if proc_id_dir.isdigit():
            dir_index[(job_id_dir, str(int(proc_id_dir)))].append(subdir)

    proc_candidates = [proc_id_4]
    if proc_id_4.isdigit():
        proc_candidates.append(str(int(proc_id_4)))

    for proc_id in proc_candidates:
        for subdir in dir_index.get((job_id, proc_id), []):
            candidate = subdir / f"{subdir.name}_{event_id}.npz"
            if candidate.is_file():
                filename = str(candidate)
                break

            candidate_padded = subdir / f"{subdir.name}_{event_id_4}.npz"
            if candidate_padded.is_file():
                filename = str(candidate_padded)
                break

            filename = ""
            for alt in subdir.glob(f"*_{event_id}.npz"):
                try:
                    if event_filename_to_event_id(alt) == sample_id:
                        filename = str(alt)
                        break
                except (ValueError, IndexError):
                    continue
            if filename:
                break
        else:
            continue
        break
    else:
        raise FileNotFoundError(f"Failed to resolve sample_id {sample_id} from {test_dir}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({str(sample_id): filename}, sort_keys=True))

    return filename


def collect_datasets(group: h5py.Group, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, item in group.items():
        key = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:
            out.update(collect_datasets(item, prefix=f"{key}/"))
    return out


def find_dataset(datasets: dict[str, np.ndarray], candidates: list[str]) -> np.ndarray:
    for name in candidates:
        if name in datasets:
            return datasets[name]
    for name in candidates:
        matches = [k for k in datasets if k.endswith(name)]
        if matches:
            return datasets[matches[0]]
    raise KeyError(f"Missing dataset. Tried keys: {candidates}")


def ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(0)
    return x


def load_sample_ids(path: Path) -> list[int]:
    with h5py.File(path, "r") as f:
        return [int(sample_id) for sample_id in f.keys()]


def shared_hist_bins(hist_data, num_bins: int):
    if not hist_data:
        return num_bins

    lo = min(float(values.min()) for _, values in hist_data)
    hi = max(float(values.max()) for _, values in hist_data)
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-3)
        lo -= pad
        hi += pad

    step = (hi - lo) / num_bins
    return [lo + i * step for i in range(num_bins + 1)]


def compute_costs_for_variant(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    hits: list[str],
    hit_cost_weights: dict[str, float],
    hit_cost_name: str,
    dice_logit_scale: float,
) -> dict[str, torch.Tensor]:
    costs: dict[str, torch.Tensor] = {}

    flow_class_logit = outputs["flow_logit"].detach().to(torch.float32)
    logit_null = flow_class_logit[..., 0]
    logit_nonnull = torch.logsumexp(flow_class_logit[..., 1:], dim=-1)
    valid_logit = logit_nonnull - logit_null
    costs["object_bce"] = 1.0 + cost_fns["object_bce"](valid_logit, targets["particle_valid"].to(torch.float32))

    for hit in hits:
        hit_weight = float(hit_cost_weights.get(hit, 1.0))
        flow_hit_logit = outputs[f"flow_{hit}_logit"].detach().to(torch.float32)
        if hit_cost_name == "mask_dice":
            flow_hit_logit = flow_hit_logit * dice_logit_scale

        target_hit_mask = targets[f"particle_{hit}_valid"].to(torch.float32)
        hit_pad_mask = targets[f"{hit}_valid"]
        costs[f"{hit}_{hit_cost_name}"] = hit_weight * cost_fns[hit_cost_name](
            flow_hit_logit,
            target_hit_mask,
            input_pad_mask=hit_pad_mask,
        )

    return costs


if sample_id is None:
    common_ids: set[int] | None = None
    for run in eval_files:
        ids = set(load_sample_ids(Path(run["path"])))
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        raise RuntimeError("No common sample_id found across eval files.")
    shared_sample_id = sorted(common_ids)[0]
else:
    shared_sample_id = int(sample_id)


for run in eval_files:
    eval_file_path = Path(run["path"])
    label = str(run.get("label", eval_file_path.stem))

    config_path = eval_file_path.parent.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found for eval file: {config_path}")

    run_cfg = yaml.safe_load(config_path.read_text())
    data_cfg = dict(run_cfg["data"])
    data_cfg["num_workers"] = 1
    data_cfg["batch_size"] = 1
    data_cfg["fast_file_discovery"] = True
    data_cfg["num_test"] = 1
    test_dir_path = Path(data_cfg["test_dir"])
    if not test_dir_path.exists():
        alt_test_dir = Path(str(test_dir_path).replace("/share/rcif2/", "/share/rcifdata/"))
        if alt_test_dir.exists():
            data_cfg["test_dir"] = str(alt_test_dir)

    datamodule = CLDDataModule(**data_cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.test_dataloader().dataset  # type: ignore[assignment]

    run_sample_id = shared_sample_id
    with h5py.File(eval_file_path, "r") as f:
        sample_key = str(run_sample_id)
        if sample_key not in f:
            raise KeyError(f"Sample id {run_sample_id} not found in {eval_file_path}")

        sample_group = f[sample_key]
        output_group_candidates = [
            f"outputs/final/{pred_object}",
            "outputs/final",
            "outputs/final/reco",
            "outputs/final/flow",
        ]
        outputs_group = None
        for path in output_group_candidates:
            if path in sample_group:
                outputs_group = sample_group[path]
                break
        if outputs_group is None:
            raise KeyError(f"Could not find outputs group. Tried: {output_group_candidates}")

        output_datasets = collect_datasets(outputs_group)

        outputs: dict[str, torch.Tensor] = {}
        outputs["flow_logit"] = torch.from_numpy(
            find_dataset(output_datasets, ["flow_logit", "flow_valid/flow_logit"])
        )

        tasks_cfg = run_cfg["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
        cld_init_args = next(
            task["init_args"] for task in tasks_cfg if task["class_path"].endswith("CLDTask")
        )
        cld_task = CLDTask(**cld_init_args)
        hits = list(cld_task.hits_included)

        for hit in hits:
            outputs[f"flow_{hit}_logit"] = torch.from_numpy(
                find_dataset(
                    output_datasets,
                    [
                        f"flow_{hit}_logit",
                        f"flow_{hit}_assignment/flow_{hit}_logit",
                    ],
                )
            )

    sample_path = resolve_sample_id_to_file(Path(data_cfg["test_dir"]), run_sample_id)
    dataset.event_ids_to_event_filenames = {run_sample_id: sample_path}
    sample = dataset.load_sample(run_sample_id)
    _inputs, targets = dataset.prep_sample(sample)

    outputs["flow_logit"] = ensure_batch(outputs["flow_logit"]).to(torch.float32)
    for hit in hits:
        key = f"flow_{hit}_logit"
        outputs[key] = ensure_batch(outputs[key]).to(torch.float32)
        n_truth = targets[f"{hit}_valid"].shape[-1]
        if outputs[key].shape[-1] > n_truth:
            outputs[key] = outputs[key][..., :n_truth]

    num_queries = outputs["flow_logit"].shape[1]
    batch_size = targets["particle_valid"].shape[0]
    num_targets = targets["particle_valid"].shape[1]
    if num_targets < num_queries:
        pad = num_queries - num_targets
        pad_valid = torch.zeros(
            (batch_size, pad),
            dtype=targets["particle_valid"].dtype,
            device=targets["particle_valid"].device,
        )
        targets["particle_valid"] = torch.cat([targets["particle_valid"], pad_valid], dim=1)
        for hit in hits:
            key = f"particle_{hit}_valid"
            if key in targets:
                t = targets[key]
                pad_mask = torch.zeros((batch_size, pad, t.shape[2]), dtype=t.dtype, device=t.device)
                targets[key] = torch.cat([t, pad_mask], dim=1)

    matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)
    num_targets = targets["particle_valid"].shape[1]
    valid_truth = targets["particle_valid"].bool()[0]
    flow_logit = outputs["flow_logit"][0].to(torch.float32)
    logit_null = flow_logit[..., 0]
    logit_nonnull = torch.logsumexp(flow_logit[..., 1:], dim=-1)
    pred_valid = (logit_nonnull - logit_null).sigmoid() >= 0.5
    hit_cost_weights = {hit: float(cld_task.hit_cost_weights.get(hit, 1.0)) for hit in hits}

    for cost_label, hit_cost_name in hit_cost_variants:
        costs = compute_costs_for_variant(
            outputs=outputs,
            targets=targets,
            hits=hits,
            hit_cost_weights=hit_cost_weights,
            hit_cost_name=hit_cost_name,
            dice_logit_scale=cld_task.mask_dice_cost_logit_scale,
        )
        total_cost = None
        for cost in costs.values():
            total_cost = cost if total_cost is None else total_cost + cost
        if total_cost is None:
            continue

        pred_idx = matcher(total_cost, object_valid_mask=targets["particle_valid"].bool())
        matched_idx = pred_idx[0][:num_targets]
        matched_pred_valid = pred_valid[matched_idx]
        num_valid_truth = int(valid_truth.sum().item())
        num_matched_valid = int(matched_pred_valid[valid_truth].sum().item()) if num_valid_truth else 0
        print(
            f"Match stats ({label}, {cost_label}): "
            f"valid_truth={num_valid_truth}, matched_to_valid_pred={num_matched_valid}"
        )

        fig, ax_diag = plt.subplots(1, 1, figsize=(6, 4))
        fig_all, ax_all = plt.subplots(1, 1, figsize=(6, 4))
        fig_hit_vs_true = None

        diag_hist_data = []
        all_hist_data = []
        target_idx = torch.arange(num_targets, device=matched_idx.device)
        term_suffix = f"_{hit_cost_name}"

        for name, cost in costs.items():
            label_term = name[:-len(term_suffix)] if name.endswith(term_suffix) else name
            matched = cost[0, matched_idx, target_idx]
            arr = matched.detach().cpu().numpy()
            if arr.size:
                diag_hist_data.append((label_term, arr))

            arr_all = cost[0].detach().flatten().cpu().numpy()
            if arr_all.size:
                all_hist_data.append((label_term, arr_all))

        diag_bins = shared_hist_bins(diag_hist_data, bins)
        all_bins = shared_hist_bins(all_hist_data, bins)

        for label_term, arr in diag_hist_data:
            ax_diag.hist(arr, bins=diag_bins, alpha=0.9, label=label_term, histtype="step")

        for label_term, arr_all in all_hist_data:
            ax_all.hist(arr_all, bins=all_bins, alpha=0.9, label=label_term, histtype="step")

        hit_cost_points = []
        if valid_truth.any():
            for hit in hits:
                cost_key = f"{hit}_{hit_cost_name}"
                target_key = f"particle_{hit}_valid"
                if cost_key not in costs or target_key not in targets:
                    continue

                matched_hit_cost = costs[cost_key][0, matched_idx, target_idx]
                true_hit_count = targets[target_key][0].to(torch.float32).sum(dim=-1)

                x = true_hit_count[valid_truth].detach().cpu().numpy()
                y = matched_hit_cost[valid_truth].detach().cpu().numpy()
                if x.size and y.size:
                    hit_cost_points.append((hit, x, y))

        if hit_cost_points:
            n_plots = len(hit_cost_points)
            ncols = min(3, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
            fig_hit_vs_true, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.5 * ncols, 3.8 * nrows),
                squeeze=False,
            )

            for i, (hit, x, y) in enumerate(hit_cost_points):
                ax = axes[i // ncols][i % ncols]
                ax.scatter(x, y, s=14, alpha=0.7, linewidths=0)

                unique_counts = sorted({int(v) for v in x.tolist()})
                x_mean = []
                y_mean = []
                for count in unique_counts:
                    mask = x == count
                    if mask.any():
                        x_mean.append(count)
                        y_mean.append(float(y[mask].mean()))
                if x_mean:
                    ax.plot(x_mean, y_mean, color="black", linewidth=1.4, label="mean")
                    ax.legend(frameon=False, fontsize=8)

                ax.set_title(hit)
                ax.set_xlabel("# true hits on particle")
                ax.set_ylabel("matched cost")
                ax.grid(alpha=0.25)

            for i in range(n_plots, nrows * ncols):
                axes[i // ncols][i % ncols].axis("off")
            fig_hit_vs_true.suptitle(f"Matched cost vs true hit count ({label}, {cost_label})")
            fig_hit_vs_true.tight_layout()

        ax_diag.set_title(f"Matched cost (diag) by term ({label}, {cost_label})")
        ax_diag.set_xlabel("cost")
        ax_diag.set_ylabel("count")
        ax_diag.set_yscale("log")
        ax_diag.legend()
        fig.tight_layout()

        ax_all.set_title(f"All costs by term ({label}, {cost_label})")
        ax_all.set_xlabel("cost")
        ax_all.set_ylabel("count")
        ax_all.set_yscale("log")
        ax_all.legend()
        fig_all.tight_layout()

        fig_matrix = None
        matrix = total_cost[0].detach().cpu().numpy()
        nrows, ncols = matrix.shape
        base = 6.0
        scale = max(nrows, ncols)
        fig_w = base * (ncols / scale)
        fig_h = base * (nrows / scale)
        fig_matrix, ax_matrix = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        im = ax_matrix.imshow(matrix, origin="lower", interpolation="nearest", aspect="equal")
        ax_matrix.set_aspect("equal", adjustable="box")
        ax_matrix.set_title(f"Total cost (all preds x all truth slots) ({label}, {cost_label})")
        ax_matrix.set_xlabel("truth particle index")
        ax_matrix.set_ylabel("pred query index")
        cbar = fig_matrix.colorbar(im, ax=ax_matrix)
        cbar.set_label("total cost")
        fig_matrix.tight_layout()

        if out_path is None:
            out_dir = default_output_dir / f"event_{run_sample_id}" / cost_label / label
            out_path_run = out_dir / f"hit_costs_eval_{label}_{run_sample_id}.png"
        else:
            out_base = Path(out_path)
            if out_base.suffix:
                out_dir = out_base.parent / f"{out_base.stem}_event_{run_sample_id}" / cost_label / label
                out_path_run = out_dir / f"{out_base.stem}_{label}_{run_sample_id}.png"
            else:
                out_dir = out_base / f"event_{run_sample_id}" / cost_label / label
                out_path_run = out_dir / f"hit_costs_eval_{label}_{run_sample_id}.png"
        out_path_run.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(out_path_run, dpi=200)
        out_path_all = out_path_run.with_name(out_path_run.stem + "_all.png")
        fig_all.savefig(out_path_all, dpi=200)
        out_path_matrix = out_path_run.with_name(out_path_run.stem + "_matrix.png")
        fig_matrix.savefig(out_path_matrix, dpi=200)
        if fig_hit_vs_true is not None:
            out_path_hit_vs_true = out_path_run.with_name(out_path_run.stem + "_matched_vs_true_hits.png")
            fig_hit_vs_true.savefig(out_path_hit_vs_true, dpi=200)

        print(f"Sample id ({label}, {cost_label}): {run_sample_id}")
        print(f"Wrote {out_path_run}")
        print(f"Wrote {out_path_all}")
        print(f"Wrote {out_path_matrix}")
        if fig_hit_vs_true is not None:
            print(f"Wrote {out_path_hit_vs_true}")

        plt.close(fig)
        plt.close(fig_all)
        plt.close(fig_matrix)
        if fig_hit_vs_true is not None:
            plt.close(fig_hit_vs_true)
