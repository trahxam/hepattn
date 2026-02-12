#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.task import CLDTask
from hepattn.models.loss import cost_fns
from hepattn.models.matcher import Matcher

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# User settings (edit as needed)
eval_files = [
    {
        "label": "combined",
        "path": Path(
            "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/All_20260202-T181849/ckpts/epoch=000-val_loss=17.18311_test_eval.h5"
        ),
    },
    {
        "label": "tracking_only",
        "path": Path(
            "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/logs/TrackingFixed_20260205-T000653/ckpts/epoch=000-val_loss=7.40857_test_eval.h5"
        ),
    },
]
sample_id = None  # int or None to use first common sample_id across eval_files
pred_object = "reco"  # output group name in eval file
out_path = None  # set to Path(...) to override default output location
default_output_dir = Path("src/hepattn/experiments/cld/plots/data/eval_costs")
include_object_valid_cost = True  # match CLDTask.cost: 1 + object_bce(valid vs invalid)
mask_bce_scale_in_combined_cost = 0.1  # CLDTask.cost combines hit term as dice + 0.1 * bce


def event_filename_to_event_id(event_filename: Path) -> int:
    id_parts = str(event_filename.stem.replace("_condor", "")).split("_")
    job_id = id_parts[-3]
    proc_id = id_parts[-2]
    event_id = id_parts[-1]
    return int(job_id + proc_id.zfill(4) + event_id.zfill(4))


def resolve_sample_id_to_file(test_dir: Path, sample_id_value: int) -> str:
    sample_id_str = str(sample_id_value)
    job_id = sample_id_str[:-8]
    proc_id_4 = sample_id_str[-8:-4]
    event_id_4 = sample_id_str[-4:]
    event_id = str(int(event_id_4))

    # One-time index of first-level test directories keyed by (job_id, proc_id).
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
                return str(candidate)

            candidate_padded = subdir / f"{subdir.name}_{event_id_4}.npz"
            if candidate_padded.is_file():
                return str(candidate_padded)

            for alt in subdir.glob(f"*_{event_id}.npz"):
                try:
                    if event_filename_to_event_id(alt) == sample_id_value:
                        return str(alt)
                except (ValueError, IndexError):
                    continue

    raise FileNotFoundError(f"Failed to resolve sample_id {sample_id_value} from {test_dir}")


def collect_datasets(group: h5py.Group, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for name, item in group.items():
        key = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:
            out.update(collect_datasets(item, prefix=f"{key}/"))
    return out


def find_dataset(datasets: dict[str, object], candidates: list[str]) -> object:
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
        return [int(sample_id_key) for sample_id_key in f.keys()]


def get_common_sample_id() -> int:
    if sample_id is not None:
        return int(sample_id)

    common_ids: set[int] | None = None
    for run in eval_files:
        ids = set(load_sample_ids(Path(run["path"])))
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        raise RuntimeError("No common sample_id found across eval files.")
    return sorted(common_ids)[0]


def prepare_run_context(eval_file_path: Path) -> tuple[dict, CLDTask, list[str], object]:
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

    tasks_cfg = run_cfg["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
    cld_init_args = next(task["init_args"] for task in tasks_cfg if task["class_path"].endswith("CLDTask"))
    cld_task = CLDTask(**cld_init_args)
    hits = list(cld_task.hits_included)
    return data_cfg, cld_task, hits, dataset


def compute_combined_and_term_cost_matrices(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    cld_task: CLDTask,
    hits: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    combined_cost = None
    term_costs: dict[str, torch.Tensor] = {}

    if include_object_valid_cost:
        flow_class_logit = outputs["flow_logit"].detach().to(torch.float32)
        logit_null = flow_class_logit[..., 0]
        logit_nonnull = torch.logsumexp(flow_class_logit[..., 1:], dim=-1)
        valid_logit = logit_nonnull - logit_null
        object_cost = 1.0 + cost_fns["object_bce"](valid_logit, targets["particle_valid"].to(torch.float32))
        term_costs["object_ce_valid_invalid"] = object_cost
        combined_cost = object_cost

    for hit in hits:
        hit_weight = float(cld_task.hit_cost_weights.get(hit, 1.0))
        flow_hit_logit = outputs[f"flow_{hit}_logit"].detach().to(torch.float32)
        target_hit_mask = targets[f"particle_{hit}_valid"].to(torch.float32)
        hit_pad_mask = targets[f"{hit}_valid"]

        hit_mask_dice_cost = cost_fns["mask_dice"](
            flow_hit_logit * cld_task.mask_dice_cost_logit_scale,
            target_hit_mask,
            input_pad_mask=hit_pad_mask,
        )
        hit_mask_bce_cost = cost_fns["mask_bce"](
            flow_hit_logit,
            target_hit_mask,
            input_pad_mask=hit_pad_mask,
        )
        term_costs[f"{hit}_mask_dice"] = hit_mask_dice_cost

        combined_hit_cost = hit_weight * (hit_mask_dice_cost + mask_bce_scale_in_combined_cost * hit_mask_bce_cost)
        combined_cost = combined_hit_cost if combined_cost is None else combined_cost + combined_hit_cost

    if combined_cost is None:
        raise RuntimeError("No cost terms found while computing total cost.")
    return combined_cost, term_costs


def resolve_output_root(base_out_path: Path | None, run_label: str, sample_id_value: int) -> Path:
    if base_out_path is None:
        return default_output_dir / f"event_{sample_id_value}" / run_label

    out_base = Path(base_out_path)
    if out_base.suffix:
        return out_base.parent / f"{out_base.stem}_event_{sample_id_value}" / run_label

    return out_base / f"event_{sample_id_value}" / run_label


def plot_cost_matrix(
    matrix: torch.Tensor,
    *,
    title: str,
    x_label: str,
    y_label: str,
    out_file: Path,
) -> None:
    nrows, ncols = matrix.shape
    base = 6.0
    scale = max(nrows, ncols, 1)
    fig_w = max(4.0, base * (ncols / scale))
    fig_h = max(4.0, base * (nrows / scale))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    im = ax.imshow(matrix.numpy(), origin="lower", interpolation="nearest", aspect="equal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cost")
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


shared_sample_id = get_common_sample_id()
matcher = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False)

for run in eval_files:
    eval_file_path = Path(run["path"])
    label = str(run.get("label", eval_file_path.stem))

    data_cfg, cld_task, hits, dataset = prepare_run_context(eval_file_path)

    with h5py.File(eval_file_path, "r") as f:
        sample_key = str(shared_sample_id)
        if sample_key not in f:
            raise KeyError(f"Sample id {shared_sample_id} not found in {eval_file_path}")

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

    sample_path = resolve_sample_id_to_file(Path(data_cfg["test_dir"]), shared_sample_id)
    dataset.event_ids_to_event_filenames = {shared_sample_id: sample_path}
    sample = dataset.load_sample(shared_sample_id)
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

    combined_cost, term_costs = compute_combined_and_term_cost_matrices(
        outputs=outputs,
        targets=targets,
        cld_task=cld_task,
        hits=hits,
    )
    matched_pred_idx = matcher(combined_cost, object_valid_mask=targets["particle_valid"].bool())[0].to(torch.long)
    matched_pred_idx_cpu = matched_pred_idx.cpu()

    combined_matrix_all = combined_cost[0].detach().cpu().index_select(0, matched_pred_idx_cpu)

    pred_valid = outputs["flow_logit"][0].argmax(dim=-1) != 0
    truth_valid = targets["particle_valid"][0].bool()
    pred_valid_matched = pred_valid.index_select(0, matched_pred_idx)
    pred_idx = pred_valid_matched.nonzero(as_tuple=False).flatten().cpu()
    truth_idx = truth_valid.nonzero(as_tuple=False).flatten().cpu()

    using_filtered_particles = bool(pred_idx.numel() > 0 and truth_idx.numel() > 0)
    if using_filtered_particles:
        combined_matrix_valid_only = combined_matrix_all.index_select(0, pred_idx).index_select(1, truth_idx)
    else:
        combined_matrix_valid_only = combined_matrix_all

    out_root = resolve_output_root(out_path, label, shared_sample_id)

    combined_valid_path = out_root / "combined_cost" / "valid_only" / "total_cost.png"
    combined_all_path = out_root / "combined_cost" / "all" / "total_cost.png"

    plot_cost_matrix(
        combined_matrix_valid_only,
        title=(
            f"Combined cost ({label}, valid-only, matched rows)\n"
            "(object valid BCE + weighted (dice + 0.1*bce))"
            "\n"
            f"sample_id={shared_sample_id}, preds={combined_matrix_valid_only.shape[0]}, truth={combined_matrix_valid_only.shape[1]}"
        ),
        x_label="truth particle index (valid only)" if using_filtered_particles else "truth slot index (all)",
        y_label="pred particle index (matched)",
        out_file=combined_valid_path,
    )

    plot_cost_matrix(
        combined_matrix_all,
        title=(
            f"Combined cost ({label}, all, matched rows)\n"
            "(object valid BCE + weighted (dice + 0.1*bce))"
            "\n"
            f"sample_id={shared_sample_id}, preds={combined_matrix_all.shape[0]}, truth={combined_matrix_all.shape[1]}"
        ),
        x_label="truth slot index (all)",
        y_label="pred query index (matched)",
        out_file=combined_all_path,
    )

    term_labels = {
        "object_ce_valid_invalid": "Object CE (valid vs invalid)",
    }
    for hit in hits:
        term_labels[f"{hit}_mask_dice"] = f"{hit} mask Dice"

    written_term_paths: list[Path] = []
    for term_name, term_matrix in term_costs.items():
        term_matrix_all = term_matrix[0].detach().cpu().index_select(0, matched_pred_idx_cpu)
        if using_filtered_particles:
            term_matrix_valid = term_matrix_all.index_select(0, pred_idx).index_select(1, truth_idx)
        else:
            term_matrix_valid = term_matrix_all

        term_valid_path = out_root / "term_costs" / "valid_only" / f"{term_name}.png"
        term_all_path = out_root / "term_costs" / "all" / f"{term_name}.png"
        term_display = term_labels.get(term_name, term_name)

        plot_cost_matrix(
            term_matrix_valid,
            title=(
                f"{term_display} ({label}, valid-only, rows matched by combined cost)"
                "\n"
                f"sample_id={shared_sample_id}, preds={term_matrix_valid.shape[0]}, truth={term_matrix_valid.shape[1]}"
            ),
            x_label="truth particle index (valid only)" if using_filtered_particles else "truth slot index (all)",
            y_label="pred particle index (matched)",
            out_file=term_valid_path,
        )
        plot_cost_matrix(
            term_matrix_all,
            title=(
                f"{term_display} ({label}, all, rows matched by combined cost)"
                "\n"
                f"sample_id={shared_sample_id}, preds={term_matrix_all.shape[0]}, truth={term_matrix_all.shape[1]}"
            ),
            x_label="truth slot index (all)",
            y_label="pred query index (matched)",
            out_file=term_all_path,
        )
        written_term_paths.extend([term_valid_path, term_all_path])

    print(f"Sample id ({label}): {shared_sample_id}")
    print(f"Using filtered particles: {using_filtered_particles}")
    print(f"Wrote {combined_valid_path}")
    print(f"Wrote {combined_all_path}")
    for term_path in written_term_paths:
        print(f"Wrote {term_path}")
