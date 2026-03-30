"""Lightning callback that produces track-residual plots every validation epoch
and uploads them to Comet.

Shows three series per field:
  Pandora (blue)  — baseline from targets
  Helix fit (green) — stage_0 (classical helix, no model correction)
  Model (orange)   — final stage prediction

Five figures are uploaded:
  val_track_residuals            — fixed-range histograms
  val_track_residuals_fullrange  — auto-scaled histograms
  val_track_residuals_scaled     — arcsinh(residual/MAD) density
  val_track_residuals_vs_truth   — median residual vs binned truth
  val_track_resolutions_vs_truth — median resolution vs binned truth
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from hepattn.experiments.cld.trkfit.plotting import (
    RESIDUAL_YLABEL,
    RESIDUALS,
    RESOLUTION_DIVIDER,
    RESOLUTION_YLABEL,
    make_arcsinh_fig,
    make_residual_fig,
    make_residual_fullrange_fig,
    make_residual_vs_truth_fig,
)

log = logging.getLogger(__name__)

# Map model field → (RESIDUALS index, display scale factor)
# d0/z0: model outputs metres, plots use mm
_MODEL_FIELD_TO_RESIDUAL: dict[str, tuple[int, float]] = {
    "eta":          (0, 1.0),
    "phi_perigee":  (1, 1.0),
    "qopt":         (2, 1.0),
    "d0_perigee_m": (3, 1e3),   # metres → mm
    "z0_perigee_m": (4, 1e3),   # metres → mm
}

# Pandora baseline keys in targets dict (all in metres for d0/z0)
_PANDORA_TARGET_KEY: dict[str, str] = {
    "eta":  "track_eta",
    "phi":  "track_phi",
    "qopt": "track_qopt",
    "d0":   "track_d0",    # metres
    "z0":   "track_z0",    # metres
}

# Truth target keys (all in metres for d0/z0)
_TRUTH_TARGET_KEY: dict[str, str] = {
    "eta":          "track_matched_particle_eta",
    "phi_perigee":  "track_matched_particle_phi_perigee",
    "qopt":         "track_matched_particle_qopt",
    "d0_perigee_m": "track_matched_particle_d0_perigee_m",   # metres
    "z0_perigee_m": "track_matched_particle_z0_perigee_m",   # metres
}

# Display scale for Pandora/truth when building residuals (matching _MODEL_FIELD_TO_RESIDUAL)
_DISPLAY_SCALE: dict[str, float] = {
    "d0_perigee_m": 1e3,
    "z0_perigee_m": 1e3,
}


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


class TrackResidualPlotCallback(Callback):
    """Accumulate residuals during validation and upload plots to Comet."""

    def __init__(self, max_events: int = 100) -> None:
        self.max_events = max_events
        self._reset()

    def _reset(self) -> None:
        n = len(RESIDUALS)
        self._pandora: list[list] = [[] for _ in range(n)]
        self._helix:   list[list] = [[] for _ in range(n)]
        self._model:   list[list] = [[] for _ in range(n)]
        self._pandora_truth: list[list] = [[] for _ in range(n)]
        self._helix_truth:   list[list] = [[] for _ in range(n)]
        self._model_truth:   list[list] = [[] for _ in range(n)]
        self._n = 0

    def _concat(self, lst: list[list]) -> list[np.ndarray | None]:
        out = []
        for chunks in lst:
            arrs = [a for a in chunks if a is not None and len(a) > 0]
            out.append(np.concatenate(arrs) if arrs else None)
        return out

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self._n >= self.max_events:
            return
        if not hasattr(pl_module, "_step_preds") or not hasattr(pl_module, "_step_targets"):
            return

        preds   = pl_module._step_preds
        targets = pl_module._step_targets

        valid_key = "track_matched_particle_valid"
        if valid_key not in targets:
            return
        valid = _to_np(targets[valid_key][0]).astype(bool)

        # ---- Pandora baseline residuals ------------------------------------
        for i, (track_field, particle_field, transform, *_) in enumerate(RESIDUALS):
            pan_key   = _PANDORA_TARGET_KEY.get(track_field)
            truth_key = _TRUTH_TARGET_KEY.get(particle_field)
            if pan_key not in targets or truth_key not in targets:
                self._pandora[i].append(None)
                self._pandora_truth[i].append(None)
                continue
            scale     = _DISPLAY_SCALE.get(particle_field, 1.0)
            pan_v     = scale * _to_np(targets[pan_key][0])[valid]
            truth_v   = scale * _to_np(targets[truth_key][0])[valid]
            finite    = np.isfinite(pan_v) & np.isfinite(truth_v)
            if finite.any():
                self._pandora[i].append(transform(pan_v[finite], truth_v[finite]))
                self._pandora_truth[i].append(truth_v[finite])
            else:
                self._pandora[i].append(None)
                self._pandora_truth[i].append(None)

        # ---- Helix (stage_0) and model (final stage) residuals ------------
        stage_names = sorted(preds.keys())
        helix_stage = "stage_0"
        final_stage = stage_names[-1]
        task_name   = next(iter(preds[final_stage]))

        for model_field, (res_idx, scale) in _MODEL_FIELD_TO_RESIDUAL.items():
            particle_field = RESIDUALS[res_idx][1]
            truth_key      = _TRUTH_TARGET_KEY.get(particle_field)
            if truth_key not in targets:
                for bucket in (self._helix, self._model):
                    bucket[res_idx].append(None)
                for bucket in (self._helix_truth, self._model_truth):
                    bucket[res_idx].append(None)
                continue

            pred_key  = f"track_{model_field}"
            transform = RESIDUALS[res_idx][2]
            truth_raw = _to_np(targets[truth_key][0])[valid]
            truth_v   = scale * truth_raw   # display units

            for bucket, bucket_truth, stage in [
                (self._helix, self._helix_truth, helix_stage),
                (self._model, self._model_truth, final_stage),
            ]:
                stage_preds = preds[stage].get(task_name, {})
                if pred_key not in stage_preds:
                    bucket[res_idx].append(None)
                    bucket_truth[res_idx].append(None)
                    continue
                pred_v = scale * _to_np(stage_preds[pred_key][0])[valid]
                finite = np.isfinite(pred_v) & np.isfinite(truth_v)
                if finite.any():
                    bucket[res_idx].append(transform(pred_v[finite], truth_v[finite]))
                    bucket_truth[res_idx].append(truth_v[finite])
                else:
                    bucket[res_idx].append(None)
                    bucket_truth[res_idx].append(None)

        self._n += 1

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._n == 0:
            return

        logger   = trainer.logger
        has_comet = hasattr(logger, "experiment") and hasattr(logger.experiment, "log_figure")
        step     = trainer.global_step

        # EMA scales from last step for arcsinh plot
        ema_scales: dict[str, float] = {}
        if hasattr(pl_module, "_step_preds"):
            preds       = pl_module._step_preds
            final_stage = sorted(preds.keys())[-1]
            task_name   = next(iter(preds[final_stage]))
            for model_field, (res_idx, scale) in _MODEL_FIELD_TO_RESIDUAL.items():
                field_label = RESIDUALS[res_idx][0]
                val = preds[final_stage].get(task_name, {}).get(f"track_{model_field}_ema_scale")
                if val is not None:
                    ema_scales[field_label] = float(val) * scale

        pandora_data = self._concat(self._pandora)
        helix_data   = self._concat(self._helix)
        model_data   = self._concat(self._model)
        pandora_truth = self._concat(self._pandora_truth)
        helix_truth   = self._concat(self._helix_truth)
        model_truth   = self._concat(self._model_truth)

        series = {
            "Pandora":   {"color": "cornflowerblue",  "ls": "-",  "data": pandora_data},
            "Helix fit": {"color": "mediumseagreen",  "ls": ":",  "data": helix_data},
            "Model":     {"color": "mediumvioletred", "ls": "--", "data": model_data},
        }

        def _apply_resolution(resid_list, truth_list):
            out = []
            for i, (track_field, *_) in enumerate(RESIDUALS):
                res = resid_list[i]
                tr  = truth_list[i]
                fn  = RESOLUTION_DIVIDER.get(track_field)
                out.append(res / fn(tr) if (res is not None and tr is not None and fn is not None) else res)
            return out

        series_vs_truth = {
            "Pandora":   {"color": "cornflowerblue",  "truth": pandora_truth, "data": pandora_data},
            "Helix fit": {"color": "mediumseagreen",  "truth": helix_truth,   "data": helix_data},
            "Model":     {"color": "mediumvioletred", "truth": model_truth,   "data": model_data},
        }
        series_vs_truth_resol = {
            "Pandora":   {"color": "cornflowerblue",  "truth": pandora_truth, "data": _apply_resolution(pandora_data, pandora_truth)},
            "Helix fit": {"color": "mediumseagreen",  "truth": helix_truth,   "data": _apply_resolution(helix_data,   helix_truth)},
            "Model":     {"color": "mediumvioletred", "truth": model_truth,   "data": _apply_resolution(model_data,   model_truth)},
        }

        plots = [
            ("val_track_residuals",
             make_residual_fig(series, "Track Residuals — validation")),
            ("val_track_residuals_fullrange",
             make_residual_fullrange_fig(series, "Track Residuals (full range) — validation")),
            ("val_track_residuals_scaled",
             make_arcsinh_fig(series, "Track Residuals — arcsinh(residual/MAD) — validation",
                              ema_scales=ema_scales or None)),
            ("val_track_residuals_vs_truth",
             make_residual_vs_truth_fig(series_vs_truth, RESIDUAL_YLABEL,
                                        "Track Residuals vs Truth — validation")),
            ("val_track_resolutions_vs_truth",
             make_residual_vs_truth_fig(series_vs_truth_resol, RESOLUTION_YLABEL,
                                        "Track Resolutions vs Truth — validation")),
        ]

        import matplotlib.pyplot as plt
        for name, fig in plots:
            if has_comet:
                logger.experiment.log_figure(figure_name=name, figure=fig, step=step)
            else:
                log.warning("No Comet logger; skipping figure %s", name)
            plt.close(fig)

        self._reset()
