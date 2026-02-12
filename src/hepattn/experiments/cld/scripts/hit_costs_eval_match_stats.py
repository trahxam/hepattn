#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import torch
import yaml

from hepattn.experiments.cld.data import CLDDataModule
from hepattn.experiments.cld.task import CLDTask
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
num_events = 100
pred_object = "reco"


def event_filename_to_event_id(event_filename: Path) -> int:
    id_parts = str(event_filename.stem.replace("_condor", "")).split("_")
    job_id = id_parts[-3]
    proc_id = id_parts[-2]
    event_id = id_parts[-1]
    return int(job_id + proc_id.zfill(4) + event_id.zfill(4))


def build_dir_index(test_dir: Path) -> dict[tuple[str, str], list[Path]]:
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
    return dir_index


def resolve_sample_id_to_file(sample_id: int, dir_index: dict[tuple[str, str], list[Path]]) -> str:
    sample_id_str = str(sample_id)
    job_id = sample_id_str[:-8]
    proc_id_4 = sample_id_str[-8:-4]
    event_id_4 = sample_id_str[-4:]
    event_id = str(int(event_id_4))

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
                    if event_filename_to_event_id(alt) == sample_id:
                        return str(alt)
                except (ValueError, IndexError):
                    continue

    raise FileNotFoundError(f"Failed to resolve sample_id {sample_id}")


def load_sample_ids(path: Path) -> list[int]:
    with h5py.File(path, "r") as f:
        return [int(sample_id) for sample_id in f.keys()]


def collect_datasets(group: h5py.Group, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for name, item in group.items():
        key = f"{prefix}{name}"
        if isinstance(item, h5py.Dataset):
            out[key] = item[()]
        else:
            out.update(collect_datasets(item, prefix=f"{key}/"))
    return out


def find_dataset(datasets: dict[str, object], candidates: list[str]):
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


common_ids: set[int] | None = None
for run in eval_files:
    ids = set(load_sample_ids(Path(run["path"])))
    common_ids = ids if common_ids is None else common_ids & ids
if not common_ids:
    raise RuntimeError("No common sample_id found across eval files.")
sample_ids = sorted(common_ids)[:num_events]
print(f"Using {len(sample_ids)} common sample IDs.")


def open_run_context(run: dict) -> dict:
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

    datamodule = CLDDataModule(**data_cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.test_dataloader().dataset  # type: ignore[assignment]
    dir_index = build_dir_index(Path(data_cfg["test_dir"]))

    tasks_cfg = run_cfg["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
    cld_init_args = next(task["init_args"] for task in tasks_cfg if task["class_path"].endswith("CLDTask"))
    cld_task = CLDTask(**cld_init_args)
    hits = list(cld_task.hits_included)

    return {
        "label": label,
        "file": h5py.File(eval_file_path, "r"),
        "dataset": dataset,
        "dir_index": dir_index,
        "cld_task": cld_task,
        "hits": hits,
        "matcher": Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False),
    }


run_contexts = [open_run_context(run) for run in eval_files]
totals = {ctx["label"]: {"valid": 0, "matched": 0, "charged_valid": 0, "charged_matched": 0} for ctx in run_contexts}

for sample_id in sample_ids:
    stats: dict[str, tuple[int, int]] = {}
    for ctx in run_contexts:
        label = ctx["label"]
        f = ctx["file"]
        dataset = ctx["dataset"]
        dir_index = ctx["dir_index"]
        cld_task = ctx["cld_task"]
        hits = ctx["hits"]
        matcher = ctx["matcher"]

        sample_key = str(sample_id)
        if sample_key not in f:
            continue

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
            continue

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

        sample_path = resolve_sample_id_to_file(sample_id, dir_index)
        dataset.event_ids_to_event_filenames = {sample_id: sample_path}
        sample = dataset.load_sample(sample_id)
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

        costs = cld_task.cost(outputs, targets)
        if "object_bce" in costs:
            costs["object_bce"] = costs["object_bce"] + 1.0
        total_cost = None
        for cost in costs.values():
            total_cost = cost if total_cost is None else total_cost + cost

        pred_idx = matcher(total_cost, object_valid_mask=targets["particle_valid"].bool())
        valid_truth = targets["particle_valid"].bool()[0]
        if "particle_class_idx" in targets:
            class_idx = targets["particle_class_idx"].long()[0]
            charged_truth = torch.isin(class_idx, torch.tensor([2, 4, 5], device=class_idx.device)) & valid_truth
        else:
            charged_truth = valid_truth
        num_targets = valid_truth.numel()
        matched_idx = pred_idx[0][:num_targets]

        flow_logit = outputs["flow_logit"][0].to(torch.float32)
        logit_null = flow_logit[..., 0]
        logit_nonnull = torch.logsumexp(flow_logit[..., 1:], dim=-1)
        pred_valid = (logit_nonnull - logit_null).sigmoid() >= 0.5
        matched_pred_valid = pred_valid[matched_idx]

        num_valid_truth = int(valid_truth.sum().item())
        num_matched_valid = int(matched_pred_valid[valid_truth].sum().item()) if num_valid_truth else 0
        num_charged_truth = int(charged_truth.sum().item())
        num_charged_matched = int(matched_pred_valid[charged_truth].sum().item()) if num_charged_truth else 0
        stats[label] = (num_valid_truth, num_matched_valid, num_charged_truth, num_charged_matched)
        totals[label]["valid"] += num_valid_truth
        totals[label]["matched"] += num_matched_valid
        totals[label]["charged_valid"] += num_charged_truth
        totals[label]["charged_matched"] += num_charged_matched

    default_stats = stats.get(run_contexts[0]["label"], (0, 0, 0, 0))
    valid_truth = stats.get("all", default_stats)[0]
    matched_all = stats.get("all", default_stats)[1]
    matched_tracking = stats.get("tracking", default_stats)[1]
    charged_all = stats.get("all", default_stats)[2]
    charged_matched_all = stats.get("all", default_stats)[3]
    charged_tracking = stats.get("tracking", default_stats)[2]
    charged_matched_tracking = stats.get("tracking", default_stats)[3]
    charged_frac_all = (charged_matched_all / charged_all) if charged_all else 0.0
    charged_frac_tracking = (charged_matched_tracking / charged_tracking) if charged_tracking else 0.0
    print(
        f"sample {sample_id}: valid_truth={valid_truth} "
        f"matched_valid_all={matched_all} matched_valid_tracking={matched_tracking} "
        f"charged_all={charged_all} charged_matched_all={charged_matched_all} ({charged_frac_all:.3f}) "
        f"charged_tracking={charged_tracking} charged_matched_tracking={charged_matched_tracking} ({charged_frac_tracking:.3f})"
    )

for ctx in run_contexts:
    label = ctx["label"]
    totals_label = totals[label]
    total_valid = totals_label["valid"]
    total_matched = totals_label["matched"]
    frac = (total_matched / total_valid) if total_valid else 0.0
    total_charged = totals_label["charged_valid"]
    total_charged_matched = totals_label["charged_matched"]
    charged_frac = (total_charged_matched / total_charged) if total_charged else 0.0
    print(f"Summary ({label}): valid_truth={total_valid} matched_valid={total_matched} frac={frac:.4f}")
    print(f"Summary ({label}): charged_truth={total_charged} matched_charged={total_charged_matched} frac={charged_frac:.4f}")
    ctx["file"].close()
