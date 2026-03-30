"""CLD track-fitting task: residual metrics with Pandora as the baseline."""

import torch
from torch import Tensor, nn

# Display scales applied to residuals before logging.
# d0_perigee_m / z0_perigee_m are in metres; plots use mm.
_DISPLAY_SCALE: dict[str, float] = {
    "d0_perigee_m": 1e3,   # m → mm
    "z0_perigee_m": 1e3,
}

# Maps each model field to (pandora_targets_field, truth_field, scale).
# Pandora d0/z0 are stored in metres (same as model output); truth also in metres.
# _DISPLAY_SCALE will convert the logged residual from metres to mm.
_PANDORA_FIELD_MAP: dict[str, tuple[str, str, float]] = {
    "eta":          ("eta",   "eta",          1.0),
    "phi_perigee":  ("phi",   "phi_perigee",  1.0),
    "qopt":         ("qopt",  "qopt",         1.0),
    "d0_perigee_m": ("d0",    "d0_perigee_m", 1.0),   # both in metres
    "z0_perigee_m": ("z0",    "z0_perigee_m", 1.0),
}


def _tensor_mad(x: Tensor) -> Tensor:
    x = x[torch.isfinite(x)]
    if x.numel() < 1:
        return x.new_tensor(float("nan"))
    return torch.median(torch.abs(x - torch.median(x)))


def _tensor_iqr(x: Tensor) -> Tensor:
    x = x[torch.isfinite(x)].float()
    if x.numel() < 2:
        return x.new_tensor(float("nan"))
    return torch.quantile(x, 0.75) - torch.quantile(x, 0.25)


def _tensor_fwhm(x: Tensor, n_bins: int = 100) -> Tensor:
    x_f = x.float()
    x_f = x_f[torch.isfinite(x_f)]
    if x_f.numel() < 2:
        return x.new_tensor(float("nan"))
    lo, hi = float(x_f.min().detach()), float(x_f.max().detach())
    if lo == hi:
        return x.new_tensor(0.0)
    counts      = torch.histc(x_f, bins=n_bins, min=lo, max=hi)
    bin_width   = (hi - lo) / n_bins
    bin_centres = x_f.new_tensor([lo + bin_width * (i + 0.5) for i in range(n_bins)])
    half_max    = counts.max() / 2.0
    above       = counts >= half_max
    if not above.any():
        return x.new_tensor(float("nan"))
    left_idx  = int(above.long().argmax())
    right_idx = n_bins - 1 - int(above.long().flip(0).argmax())
    return bin_centres[right_idx] - bin_centres[left_idx]


class CLDTrackTask(nn.Module):
    """Metrics-only task for the CLD BoostedTrackFitter.

    Loss is computed inside BoostedTrackFitter.  This task provides per-field
    residual / resolution statistics and Pandora-baseline comparisons.

    Expected prediction keys:  ``track_{field}``            (1, N_tracks)
    Expected target keys:
        ``track_matched_particle_{field}``                  (1, N_tracks)
        ``track_matched_particle_valid``                    (1, N_tracks) bool
        ``track_{pandora_field}``  e.g. track_eta, track_d0 (1, N_tracks) — Pandora baseline
    """

    has_intermediate_loss: bool = False

    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        angular_fields: list[str] | None = None,
    ):
        super().__init__()
        self.name          = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields        = fields
        self.angular_fields = angular_fields or []

        self.register_buffer(
            "angular_mask",
            torch.tensor([f in self.angular_fields for f in fields], dtype=torch.bool),
            persistent=False,
        )

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        valid = targets[f"{self.target_object}_valid"]   # (1, N_tracks)
        out["match_fraction"] = valid.float().mean()

        for field in self.fields:
            pred   = preds[f"{self.output_object}_{field}"][valid]
            target = targets[f"{self.target_object}_{field}"][valid]
            scale  = _DISPLAY_SCALE.get(field, 1.0)

            if field in self.angular_fields:
                res = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
            else:
                res = pred - target

            finite = torch.isfinite(res)
            res    = res[finite]
            target = target[finite]

            if res.numel() == 0:
                continue

            out[f"{field}_residual_mean"] = scale * torch.mean(res)
            out[f"{field}_residual_std"]  = scale * torch.std(res)
            out[f"{field}_residual_mad"]  = scale * _tensor_mad(res)
            out[f"{field}_residual_iqr"]  = scale * _tensor_iqr(res)
            out[f"{field}_residual_fwhm"] = scale * _tensor_fwhm(res)

            if field not in self.angular_fields:
                resolution = res / target
                out[f"{field}_resolution_mean"] = torch.mean(resolution)
                out[f"{field}_resolution_std"]  = torch.std(resolution)
                out[f"{field}_resolution_mad"]  = _tensor_mad(resolution)
                out[f"{field}_resolution_iqr"]  = _tensor_iqr(resolution)
                out[f"{field}_resolution_fwhm"] = _tensor_fwhm(resolution)

            for stat_key in ("norm_res_mean", "norm_res_std", "ema_scale", "gate_mean", "delta_mad"):
                val = preds.get(f"{self.output_object}_{field}_{stat_key}")
                if val is not None:
                    out[f"{field}_{stat_key}"] = val

            out[f"{field}_target_mean"] = scale * torch.mean(target)
            out[f"{field}_target_std"]  = scale * torch.std(target)

        return out

    def baseline_metrics(self, targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pandora baseline residuals — computed once per step."""
        out: dict[str, Tensor] = {}
        valid = targets[f"{self.target_object}_valid"]

        for field, (pan_field, truth_field, pan_scale) in _PANDORA_FIELD_MAP.items():
            if field not in self.fields:
                continue
            pan_key   = f"track_{pan_field}"
            truth_key = f"{self.target_object}_{truth_field}"
            if pan_key not in targets or truth_key not in targets:
                continue

            pan_vals   = targets[pan_key][valid].float()
            truth_vals = targets[truth_key][valid].float()
            finite     = torch.isfinite(pan_vals) & torch.isfinite(truth_vals)
            if not finite.any():
                continue

            pan_vals   = pan_vals[finite]
            truth_vals = truth_vals[finite]

            scale = _DISPLAY_SCALE.get(field, 1.0)

            if field in self.angular_fields:
                res = torch.atan2(torch.sin(pan_vals - truth_vals), torch.cos(pan_vals - truth_vals))
            else:
                res = pan_scale * (pan_vals - truth_vals)

            out[f"pandora_{field}_residual_mean"] = scale * torch.mean(res)
            out[f"pandora_{field}_residual_std"]  = scale * torch.std(res)
            out[f"pandora_{field}_residual_mad"]  = scale * _tensor_mad(res)
            out[f"pandora_{field}_residual_iqr"]  = scale * _tensor_iqr(res)
            out[f"pandora_{field}_residual_fwhm"] = scale * _tensor_fwhm(res)

            if field not in self.angular_fields:
                resolution = (pan_vals - truth_vals) / truth_vals
                out[f"pandora_{field}_resolution_mean"] = torch.mean(resolution)
                out[f"pandora_{field}_resolution_std"]  = torch.std(resolution)
                out[f"pandora_{field}_resolution_mad"]  = _tensor_mad(resolution)
                out[f"pandora_{field}_resolution_iqr"]  = _tensor_iqr(resolution)
                out[f"pandora_{field}_resolution_fwhm"] = _tensor_fwhm(resolution)

        return out
