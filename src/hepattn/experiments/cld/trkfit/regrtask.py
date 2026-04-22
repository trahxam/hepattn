"""Object property regression task for CLD: per-particle track + calo + charge prediction.

Three heads on a shared trunk predict:
  * |1/pT| correction on the helix-seed track parameters (per-param when ``use_full_perigee``),
  * scalar log-energy for particles with calo hits,
  * charge sign (2-class over PID-charged queries).

PID routing is always on (no 3-class gate). Supports both the original single-head
|1/pT|-correction config (v7_ablation) and the per-param correction config (v12e).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import BarronAdaptiveLoss
from hepattn.models.task import REGRESSION_LOSS_FNS, RegressionLossType, Task
from hepattn.utils.helix import (
    compute_helix_residuals,
    helix_seed_3pt,
)
from hepattn.utils.regr_utils import (
    bin_interp,
    compute_calo_direction,
    compute_calo_direction_split,
    pchip_coeffs,
)


class ObjectPropertyRegressionTask(Task):
    """Particle property regression: track |1/pT| correction + calorimeter energy + charge sign.

    Three heads on a shared trunk, each with its own adapter:
    - **Track correction**: log-space multiplicative correction on |seed q/pT| magnitude
    - **Energy**: scalar log_E prediction (all particles with calo hits)
    - **Gate**: 2-class {+,-} charge-sign classifier over PID-charged queries

    Optional features (enabled via config flags):
    - ``use_split_calo``: separate ECAL/HCAL directions + per-calo track deltas + shower shape
    - ``use_helix_residuals``: helix seed fit quality features (circle_rms, z_rms)
    - ``use_full_perigee``: predict 5 perigee params (inv_pt, d0, z0, phi0, eta) with per-param MAD norm
    - ``use_barron_loss``, ``use_arcsinh``, ``use_outlier_masking``: v11+ robustness knobs

    At inference (v4/v5 path): pred_qpt = charge_sign * |seed_qpt| * exp(correction).
    Full-perigee path denormalizes each param's correction with per-param bin-stat median/MAD.

    All upstream inputs (query embed, hit masks, helix seed, calo energies) are
    fully detached — only the trunk, adapters, heads participate in backprop.

    Scalar features use physics-informed transforms (log1p, sin/cos decomposition,
    z-score) rather than learned BatchNorm.
    """

    # Legacy ckpts (v12b through v12e) saved per-param bin-stat buffers with a ``v11_``
    # prefix. New trainings drop the prefix entirely; ``_load_from_state_dict`` strips it
    # from incoming keys so old ckpts still load cleanly.
    _LEGACY_BIN_BUF_PREFIX = "v11_"

    # Deprecated buffers that older ckpts serialize but the current class no longer
    # registers. Stripped from incoming state_dicts so ``load_state_dict(strict=True)`` passes.
    _DEPRECATED_BUFFER_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        input_constituent: str | list[str],
        calo_constituents: list[str],
        input_object: str,
        output_object: str,
        target_object: str,
        dim: int,
        # Loss
        loss: RegressionLossType = "smooth_l1",
        loss_weight: float = 1.0,
        track_correction_weight: float = 1.0,
        energy_weight: float = 1.0,
        gate_loss_weight: float = 1.0,
        loss_combination: str = "weighted_sum",
        # v12e: per-param priority GLS partition (only used when loss_combination="gls").
        # Defaults set in body. Mutually exclusive with use_per_param_kendall (validated in __init__).
        gls_priority_params: list[str] | None = None,
        gls_bulk_params: list[str] | None = None,
        track_loss_beta: float = 1.0,
        # Hit assignment
        pred_threshold: float = 0.5,
        b_field: float = 2.0,
        length_unit: float = 1.0,
        eps: float = 1e-8,
        # Thresholds
        min_tracker_hits: int = 5,
        min_calo_hits: int = 3,
        # v12b: decouple seed availability from loss contribution
        seed_min_hits: int | None = None,  # defaults to min_tracker_hits
        loss_min_hits: int | None = None,  # defaults to seed_min_hits
        # Reproduce the v7-era ``helix_seed_3pt`` call bit-for-bit (``sort_mode="rt"``,
        # no ``z_valid`` vtxd-only mask, no ``time``, default ``z_phi_sort=True``).
        # Required for loading ckpts trained before the v11 hybrid-sort / vtxd-z_valid
        # improvements. New trainings should leave this off.
        use_legacy_helix_seed: bool = False,
        # 3-point seed triple selection when ``use_legacy_helix_seed=False``.
        # ``"rt"`` | ``"time"`` | ``"hybrid"``.
        helix_sort_mode: str = "hybrid",
        # Output scaling (tanh-based)
        correction_scale: float = 1.0,
        log_e_scale: float = 6.0,
        # Architecture
        trunk_hidden_layers: list[int] | None = None,
        track_adapter_hidden: list[int] | None = None,
        energy_adapter_hidden: list[int] | None = None,
        gate_adapter_hidden: list[int] | None = None,
        track_head_hidden_layers: list[int] | None = None,
        energy_head_hidden_layers: list[int] | None = None,
        gate_head_hidden_layers: list[int] | None = None,
        scalar_mlp_hidden: list[int] | None = None,
        scalar_mlp_output_dim: int = 64,
        dropout: float = 0.0,
        # Energy target field
        energy_target: str = "energy",
        energy_bias_init: float = 1.5,
        # Schedule
        has_intermediate_loss: bool = False,
        permute_loss: bool = True,
        calo_energy_field: str = "energy",
        # PID
        num_pid_classes: int = 6,
        # Muon constituents
        muon_constituents: list[str] | None = None,
        # v5 feature flags (all default to False for backward compat with v4)
        use_split_calo: bool = False,
        use_helix_residuals: bool = False,
        # PID charged class indices (default: charged_hadron=2, electron=4, muon=5)
        pid_charged_classes: list[int] | None = None,
        # Ablation flags: disable explicit calo/tracking scalar features
        use_calo_features: bool = True,
        use_track_correction: bool = True,
        # v8b: MAD-normalized additive 1/pT targets + bin stats YAML
        target_mode: str = "log",
        bin_stats_path: str | None = None,
        # Interpolation mode for bin stats lookup: "step" (hard), "linear", "cubic" (PCHIP)
        interp_mode: str = "step",
        # v8c: Barron adaptive loss
        use_barron_loss: bool = False,
        barron_alpha_init: float = 1.0,
        barron_scale_init: float = 1.0,
        # v12b: pin alpha (no collapse), keep scale c learnable
        barron_alpha_learnable: bool = True,
        # v12c: per-param alpha pinning (fallback to barron_alpha_init when missing). See v12c_alpha_per_param_pinning memo.
        barron_alpha_init_per_param: dict | None = None,
        # v12c: per-param Kendall & Gal log-σ² weighting. Replaces track_correction/calo_regression/gate_ce with a single
        # kendall_total = Σ_i exp(-clamp(s_i)) · L_i + clamp(s_i).
        use_per_param_kendall: bool = False,
        kendall_log_var_clip: tuple = (-3.0, 3.0),
        # v12d F1: inject binvars (total_calo_ET, combined_centroid_eta, sin/cos(combined_centroid_phi)) as features.
        # The neutral correction head trains on targets binned by total_calo_ET but cannot recover ET from its existing
        # inputs (needs division by cosh(combined_eta)). Exposing these explicitly removes that information bottleneck.
        use_binvar_features: bool = False,
        # v12d F2: inject per-param bin_median/bin_mad as features using N1 normalization:
        #   feat_bias   = bin_median / bin_mad                    (z-score of the bin's median)
        #   feat_spread = log(bin_mad / bin_mad_ref_p)            (log-relative spread vs per-param reference)
        # Each of the 8-10 loss-active params contributes 2 scalars.
        use_bin_stats_features: bool = False,
        # v9a: pool frozen encoder embeddings into trunk (no new parameters)
        # v9a/v11: sagitta (s/L) as scalar feature from 3-point seed
        use_sagitta: bool = False,
        # v11: unified correction-based prediction — all params as seed + correction
        use_full_perigee: bool = False,
        use_calo_head: bool = False,
        # v11b: arcsinh activation, outlier masking, aux energy loss, neutral-only energy
        use_arcsinh: bool = False,
        use_outlier_masking: bool = False,
        outlier_percentile: float = 0.95,
        # v12: per-param outlier masking percentiles (overrides outlier_percentile when set)
        # Keys: inv_pt, d0, z0, phi0, eta, energy_track, energy, neutral_deta, neutral_dphi
        per_param_outlier_percentile: dict[str, float] | None = None,
        use_aux_energy_loss: bool = False,
        neutral_energy_only: bool = False,
        neutral_calo_threshold: float = 0.1,
        # v12b: late-split per-param heads + per-head loss /N normalization + extra scalar features
        split_track_head_per_param: bool = False,
        split_energy_head_per_param: bool = False,
        track_head_per_param_hidden: list[int] | None = None,
        energy_head_per_param_hidden: list[int] | None = None,
        add_v12b_scalars: bool = False,
        # v12g: truth-hit-count gating for training-time loss masks. Decouples training signal cleanliness
        # from pred-mask quality. Inference routing unchanged (still uses PRED masks).
        # When True: track head activates only on charged with truth_n_sihit >= truth_loss_sihit_min;
        # energy/neutral head activates on truth-neutral-like (~is_charged OR truth_n_sihit==0) with
        # truth_n_calo >= truth_loss_calo_min. Default False for backwards compat.
        use_truth_hit_gating: bool = False,
        truth_loss_sihit_min: int = 5,
        truth_loss_calo_min: int = 6,
        **_deprecated_kwargs: Any,
    ):
        # Swallow deprecated kwargs from older YAML configs (e.g. ``use_eta_regression``,
        # ``use_inv_pt_weight``, ``trunk_mode``) whose behavior has been removed.
        del _deprecated_kwargs
        super().__init__(
            has_intermediate_loss=has_intermediate_loss,
            permute_loss=permute_loss,
        )

        self.name = name
        self.tracker_constituents = [input_constituent] if isinstance(input_constituent, str) else list(input_constituent)
        self.calo_constituents = list(calo_constituents)
        self.muon_constituents = list(muon_constituents) if muon_constituents else []
        self.input_constituents = self.tracker_constituents + self.calo_constituents + self.muon_constituents
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object

        self.dim = dim
        self.loss_weight = loss_weight
        self.track_correction_weight = track_correction_weight
        self.energy_weight = energy_weight
        self.gate_loss_weight = gate_loss_weight
        self.loss_combination = loss_combination
        self.track_loss_beta = track_loss_beta
        self.loss_fn_name = loss
        self.loss_fn = REGRESSION_LOSS_FNS[loss]
        self.pred_threshold = pred_threshold
        self.b_field = b_field
        self.length_unit = length_unit
        self.eps = eps
        self.min_tracker_hits = min_tracker_hits
        self.min_calo_hits = min_calo_hits
        # v12b: seed/loss threshold decoupling (defaults preserve back-compat with min_tracker_hits)
        self.seed_min_hits = seed_min_hits if seed_min_hits is not None else min_tracker_hits
        self.loss_min_hits = loss_min_hits if loss_min_hits is not None else self.seed_min_hits
        self.use_legacy_helix_seed = use_legacy_helix_seed
        self.helix_sort_mode = helix_sort_mode
        # v12b: per-head loss normalization, split heads, extra scalars
        self.split_track_head_per_param = split_track_head_per_param
        self.split_energy_head_per_param = split_energy_head_per_param
        self.add_v12b_scalars = add_v12b_scalars
        # v12g: truth-hit-count gating knobs
        self.use_truth_hit_gating = use_truth_hit_gating
        self.truth_loss_sihit_min = int(truth_loss_sihit_min)
        self.truth_loss_calo_min = int(truth_loss_calo_min)
        self.barron_alpha_learnable = barron_alpha_learnable
        # v12c: per-param alpha pins + per-param Kendall log-σ² weighting
        self.barron_alpha_init_per_param = barron_alpha_init_per_param or {}
        self.use_per_param_kendall = use_per_param_kendall
        self.kendall_log_var_clip = kendall_log_var_clip
        # v12e: per-param priority GLS group membership.
        self.gls_priority_params = gls_priority_params if gls_priority_params is not None else ["inv_pt", "energy_track", "energy"]
        self.gls_bulk_params = (
            gls_bulk_params if gls_bulk_params is not None else ["d0", "z0", "phi0", "eta", "neutral_deta", "neutral_dphi", "gate_ce"]
        )
        if self.loss_combination == "gls" and self.use_per_param_kendall:
            raise ValueError("loss_combination='gls' (priority-GLS) and use_per_param_kendall=True are mutually exclusive")
        if self.loss_combination == "gls":
            all_gls_params = self.gls_priority_params + self.gls_bulk_params
            if "energy_track" in all_gls_params and not use_aux_energy_loss:
                raise ValueError("energy_track listed in GLS group but use_aux_energy_loss=False — would silently degrade group size")

        # v12d: feature injection flags (buffers registered further below once bin_stats are loaded)
        self.use_binvar_features = use_binvar_features
        self.use_bin_stats_features = use_bin_stats_features
        self.correction_scale = correction_scale
        self.log_e_scale = log_e_scale
        self.calo_energy_field = calo_energy_field
        self.energy_target = energy_target
        self.num_pid_classes = num_pid_classes
        self.use_split_calo = use_split_calo
        self.use_helix_residuals = use_helix_residuals
        self.pid_charged_classes = pid_charged_classes or [2, 4, 5]
        self.use_calo_features = use_calo_features
        self.use_track_correction = use_track_correction
        self.target_mode = target_mode
        self.interp_mode = interp_mode
        self.use_barron_loss = use_barron_loss
        self.use_sagitta = use_sagitta
        self.use_full_perigee = use_full_perigee
        self.use_calo_head = use_calo_head
        self.use_arcsinh = use_arcsinh
        self.use_outlier_masking = use_outlier_masking
        self.outlier_percentile = outlier_percentile
        # v12: per-param masking percentiles (from offline loss magnitude analysis)
        self.per_param_outlier_pct = per_param_outlier_percentile or {}
        self.use_aux_energy_loss = use_aux_energy_loss
        self.neutral_energy_only = neutral_energy_only
        self.neutral_calo_threshold = neutral_calo_threshold

        # No-calo ablation: force split_calo off
        if not use_calo_features:
            self.use_split_calo = False
            use_split_calo = False  # local var used below for n_scalar

        # No-tracking ablation: force tracking-specific flags off
        if not use_track_correction:
            self.use_helix_residuals = False
            use_helix_residuals = False

        # Shared object_nets — populated by MaskFormer
        self.object_nets: dict[str, nn.Module] = {}

        # Output keys
        self.log_correction_key = output_object + "_log_correction"
        self.pred_qpt_key = output_object + "_pred_qpt"
        self.log_E_key = output_object + "_log_E"
        self.gate_logits_key = output_object + "_gate_logits"

        # Inputs and outputs for MaskFormer permutation
        self.inputs = [input_object + "_embed"]
        for c in self.input_constituents:
            self.inputs.extend([c + "_valid", c + "_embed"])
        self.outputs = [
            self.log_correction_key,
            self.pred_qpt_key,
            self.log_E_key,
            self.gate_logits_key,
            f"{output_object}_seed_qpt",
            f"{output_object}_seed_eta",
            f"{output_object}_seed_valid",
            f"{output_object}_gate_charged",
            f"{output_object}_gate_neutral",
            f"{output_object}_has_calo",
            f"{output_object}_calo_direction",
        ]
        if use_full_perigee:
            self.outputs.extend([
                f"{output_object}_track_corrections",
                f"{output_object}_seed_inv_pt",
                f"{output_object}_seed_d0",
                f"{output_object}_seed_z0",
                f"{output_object}_seed_phi0",
            ])
        if use_calo_head:
            self.outputs.extend([
                f"{output_object}_calo_corrections",
                f"{output_object}_calo_E_sum",
                f"{output_object}_calo_centroid_eta",
                f"{output_object}_calo_centroid_phi",
            ])
        # v11b: routing info and track-derived energy
        if neutral_energy_only:
            self.outputs.extend([
                f"{output_object}_n_tracker",
                f"{output_object}_neutral_eligible",
            ])
        if use_aux_energy_loss:
            self.outputs.append(f"{output_object}_E_track")

        # --- Scalar feature dimensions ---
        n_scalar = num_pid_classes  # PID probs (6)
        if use_track_correction:
            n_scalar += 6  # d0, sin_phi0, cos_phi0, omega, z0, tan_lambda
        if use_calo_features and use_track_correction:
            n_scalar += len(self.calo_constituents)  # calo energy sums (2)
            n_scalar += len(self.input_constituents)  # hit counts: tracker + calo + muon (5)
            if use_split_calo:
                n_scalar += 6 + 5 + 6 + 6  # calo_dirs(6) + energy_feats(5) + track_deltas(6) + shower_shape(6)
            else:
                n_scalar += 5 + 2 + 3  # v4 default: calo_dir(3) + ecal_frac(1) + total_E(1) + E/pT(1) + muon(1) + delta(3)
        elif use_calo_features and not use_track_correction:
            n_scalar += len(self.calo_constituents)  # calo energy sums (2)
            n_scalar += len(self.calo_constituents) + len(self.muon_constituents)  # calo + muon hit counts (3)
            if use_split_calo:
                n_scalar += 6 + 3 + 6  # calo_dirs(6) + energy_feats_notrack(3) + shower_shape(6)
            else:
                n_scalar += 3 + 3  # calo_dir(3) + ecal_frac(1) + total_E(1) + muon(1)
        else:
            n_scalar += len(self.tracker_constituents)  # hit counts: tracker only (2)
        if use_helix_residuals:
            n_scalar += 5  # log1p(circle_rms), log1p(z_rms), log1p(max_circle), circle_mean, z_mean
        if use_sagitta:
            n_scalar += 1  # log1p(sagitta)
        if add_v12b_scalars:
            # v12b neutral/calo-focused inductive bias: 8 extra scalars
            # n_sihit_flag, hcal_frac, delta_eta_calos, sin_dphi_calos, cos_dphi_calos,
            # ecal_max_hit_frac, hcal_max_hit_frac, signed_log1p_ecal_minus_hcal
            n_scalar += 8
        # v12d F1: 4 binvar scalars (log1p(total_calo_ET), ceta, sin(cphi), cos(cphi))
        if use_binvar_features:
            n_scalar += 4
        # v12d F2: 2 scalars (feat_bias, feat_spread) per loss-active param.
        # Define the param list here (mirrored in the post-yaml-load buffer registration and in
        # _transform_scalars' injection loop). Kept flag-consistent with the Kendall setup.
        if use_bin_stats_features:
            self._bin_stats_feature_params: list[str] = ["inv_pt", "d0", "z0", "phi0", "eta"]
            if use_aux_energy_loss:
                self._bin_stats_feature_params.append("energy_track")
            if use_calo_head:
                energy_key = "energy_neutral" if neutral_energy_only else "energy_charged"
                self._bin_stats_feature_params.extend([energy_key, "neutral_deta", "neutral_dphi"])
            self._bin_stats_binvar: dict[str, str] = {
                "inv_pt": "pt",
                "d0": "pt",
                "z0": "pt",
                "phi0": "pt",
                "eta": "pt",
                "energy_track": "pt",
                "energy_charged": "ET",
                "energy_neutral": "ET",
                "neutral_deta": "ET",
                "neutral_dphi": "ET",
            }
            n_scalar += 2 * len(self._bin_stats_feature_params)
        self.n_scalar = n_scalar

        # Scalar-feature z-score buffers. Pre-computed statistics from 1000 CLD test events for
        # features with large natural scale (omega, log1p(calo_sums), log1p(hit_counts) and the
        # v4-layout split-calo extras). Features without pre-computed stats use identity
        # (mean=0, std=1). The loaded ckpt's trained values overwrite these on ``load_state_dict``.
        scalar_mean = torch.zeros(n_scalar)
        scalar_std = torch.ones(n_scalar)
        # omega (helix slot 3 after the 6-way PID block)
        scalar_mean[num_pid_classes + 3] = -0.0926
        scalar_std[num_pid_classes + 3] = 17.6576
        # log1p(ecal_sum), log1p(hcal_sum) — offsets after PID(6) + helix(6)
        _calo_sum_offset = num_pid_classes + 6
        scalar_mean[_calo_sum_offset] = 0.7037
        scalar_std[_calo_sum_offset] = 0.6571
        scalar_mean[_calo_sum_offset + 1] = 0.3580
        scalar_std[_calo_sum_offset + 1] = 0.6330
        # log1p(hit_counts) for vtxd, trkr, ecal, hcal, muon
        _hit_count_offset = _calo_sum_offset + len(self.calo_constituents)
        _hit_count_stats = [(0.8349, 0.9624), (1.1058, 0.9119), (3.4513, 1.3558), (1.1674, 1.3981), (0.0321, 0.2439)]
        for i, (m, s) in enumerate(_hit_count_stats):
            scalar_mean[_hit_count_offset + i] = m
            scalar_std[_hit_count_offset + i] = s
        # v4 (combined-calo) layout: calo_eta, log1p(total_calo_E), log1p(E/seed_pT), log1p(muon_E_sum).
        # Split-calo / v11 layouts use identity (stats learned via trunk LayerNorm).
        if not use_split_calo:
            scalar_mean[_hit_count_offset + 5] = -0.0121  # calo_eta
            scalar_std[_hit_count_offset + 5] = 1.0604
            scalar_mean[_hit_count_offset + 9] = 0.9347   # log1p(total_calo_E)
            scalar_std[_hit_count_offset + 9] = 0.7705
            scalar_mean[_hit_count_offset + 10] = 0.8152  # log1p(E/seed_pT)
            scalar_std[_hit_count_offset + 10] = 0.4501
            scalar_mean[_hit_count_offset + 11] = 0.00002  # log1p(muon_E_sum)
            scalar_std[_hit_count_offset + 11] = 0.00025
        self.register_buffer("scalar_mean", scalar_mean)
        self.register_buffer("scalar_std", scalar_std)

        # Scalar MLP: transforms(→n_scalar) → MLP → scalar_mlp_output_dim
        scalar_hidden = scalar_mlp_hidden or [48]
        self.scalar_mlp = Dense(
            input_size=n_scalar, output_size=scalar_mlp_output_dim, hidden_layers=scalar_hidden, activation=nn.GELU(), dropout=dropout
        )

        # Trunk: query_embed + scalar_embed → shared representation
        trunk_input_dim = dim + scalar_mlp_output_dim
        trunk_hidden = trunk_hidden_layers or [256, 256, 128]
        trunk_output_dim = trunk_hidden[-1]
        self.trunk_norm = nn.LayerNorm(trunk_input_dim)
        self.trunk_net = Dense(input_size=trunk_input_dim, output_size=trunk_output_dim, hidden_layers=trunk_hidden, dropout=dropout)

        # Per-head adapters: trunk_out → adapter → head
        energy_adapt = energy_adapter_hidden or [64]
        gate_adapt = gate_adapter_hidden or [64]

        self.gate_adapter = Dense(input_size=trunk_output_dim, output_size=gate_adapt[-1], hidden_layers=gate_adapt[:-1], dropout=dropout)

        # Track correction head: skip when no tracking — avoids dead weight decay
        if use_track_correction:
            track_adapt = track_adapter_hidden or [64]
            self.track_adapter = Dense(input_size=trunk_output_dim, output_size=track_adapt[-1], hidden_layers=track_adapt[:-1], dropout=dropout)
            track_hidden = track_head_hidden_layers or []
            track_output_dim = 5 if use_full_perigee else 1
            self.track_output_dim = track_output_dim
            if split_track_head_per_param and use_full_perigee:
                # v12b: one dedicated head per track param — independent gradient path at the output layer
                per_param_hidden = track_head_per_param_hidden or []
                self.track_heads = nn.ModuleList([
                    Dense(input_size=track_adapt[-1], output_size=1, hidden_layers=per_param_hidden, dropout=dropout) for _ in range(track_output_dim)
                ])
                for head in self.track_heads:
                    nn.init.zeros_(head.net[-1].weight)
                    nn.init.zeros_(head.net[-1].bias)
            else:
                self.track_head = Dense(input_size=track_adapt[-1], output_size=track_output_dim, hidden_layers=track_hidden, dropout=dropout)
                # Zero-init output layer → start from correction=0 → exp(0)=1 → pred magnitude unchanged
                nn.init.zeros_(self.track_head.net[-1].weight)
                nn.init.zeros_(self.track_head.net[-1].bias)

        # Energy / calo head: scalar log_E or 3-param correction (energy + neutral direction)
        if use_calo_features:
            self.energy_adapter = Dense(input_size=trunk_output_dim, output_size=energy_adapt[-1], hidden_layers=energy_adapt[:-1], dropout=dropout)
            energy_hidden = energy_head_hidden_layers or []
            calo_output_dim = 3 if use_calo_head else 1
            self.calo_output_dim = calo_output_dim
            if split_energy_head_per_param and use_calo_head:
                # v12b: per-output head for energy head (E, neutral_deta, neutral_dphi)
                per_param_hidden = energy_head_per_param_hidden or []
                self.energy_heads = nn.ModuleList([
                    Dense(input_size=energy_adapt[-1], output_size=1, hidden_layers=per_param_hidden, dropout=dropout) for _ in range(calo_output_dim)
                ])
                for head in self.energy_heads:
                    nn.init.zeros_(head.net[-1].weight)
                    nn.init.zeros_(head.net[-1].bias)
            else:
                self.energy_head = Dense(input_size=energy_adapt[-1], output_size=calo_output_dim, hidden_layers=energy_hidden, dropout=dropout)
                if use_calo_head:
                    # v11: zero-init for correction-based prediction
                    nn.init.zeros_(self.energy_head.net[-1].weight)
                    nn.init.zeros_(self.energy_head.net[-1].bias)
                else:
                    nn.init.constant_(self.energy_head.net[-1].bias, energy_bias_init)

        # Gate head: 2-class {+,-} charge-sign classifier over PID-charged queries.
        gate_hidden = gate_head_hidden_layers or []
        self.gate_head = Dense(input_size=gate_adapt[-1], output_size=2, hidden_layers=gate_hidden, dropout=dropout)

        # v8c: Barron adaptive loss (learnable alpha shape + scale per head)
        # v12b: barron_alpha_learnable=False pins alpha (scale c remains learnable)
        # v12c: barron_alpha_init_per_param dict allows per-param alpha pinning (based on kept |z| tail severity).
        #       See v12c_alpha_per_param_pinning.md for motivation. DO NOT go learnable (sync bug in
        #       barron_alpha_sync_discovery.md).
        if use_barron_loss:
            if use_full_perigee:
                # v11: per-param Barron loss (8 separate learnable alpha/scale pairs)
                barron_params = ["inv_pt", "d0", "z0", "phi0", "eta", "energy", "neutral_deta", "neutral_dphi"]
                # v11b: add energy_track for aux energy loss
                if use_aux_energy_loss:
                    barron_params.append("energy_track")
                self.barron_losses = nn.ModuleDict({
                    param: BarronAdaptiveLoss(
                        alpha_init=(barron_alpha_init_per_param or {}).get(param, barron_alpha_init),
                        scale_init=barron_scale_init,
                        alpha_learnable=barron_alpha_learnable,
                    )
                    for param in barron_params
                })
            else:
                self.track_barron = BarronAdaptiveLoss(
                    alpha_init=barron_alpha_init,
                    scale_init=barron_scale_init,
                    alpha_learnable=barron_alpha_learnable,
                )

        # v12c: per-param Kendall log-σ² parameters. One per sub-loss we train (tracks + calo + gate).
        # Registered as nn.Parameter(zeros) → init weight = 1 via exp(-0) = 1. Clipped in loss() each step.
        if use_per_param_kendall:
            kendall_names = ["inv_pt", "d0", "z0", "phi0", "eta"]
            if use_aux_energy_loss:
                kendall_names.append("energy_track")
            if use_calo_features and use_full_perigee:
                kendall_names.extend(["energy", "neutral_deta", "neutral_dphi"])
            kendall_names.append("gate")
            self.kendall_log_var = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in kendall_names})
            self._kendall_names = kendall_names

        # Per-pT-bin correction target stats — loaded from YAML (compute_bin_stats.py)
        self._per_param_bins = False
        if bin_stats_path is not None:
            with Path(bin_stats_path).open() as f:
                stats = yaml.safe_load(f)

            # Detect format: v12+ has per-param bin_edges, v11 has shared top-level bin_edges
            self._per_param_bins = "bin_edges" not in stats

            if not self._per_param_bins:
                # Legacy v11/v11b: shared bin edges for all params
                self.register_buffer("corr_bin_edges", torch.tensor(stats["bin_edges"]))
                bin_centers = torch.tensor(stats["bin_centers"])
                self.register_buffer("corr_bin_centers", bin_centers)
                self.register_buffer("corr_log_bin_centers", torch.log(bin_centers))

            if use_full_perigee:
                # Per-param median / MAD / correction_scale buffers for the 9 loss-active param
                # groups (10 with energy_track aux). Reads go through ``_bin_stat`` / ``_pchip``.
                bin_params = ["inv_pt", "d0", "z0", "phi0", "eta", "energy_charged", "energy_neutral", "neutral_deta", "neutral_dphi"]
                if use_aux_energy_loss:
                    bin_params.append("energy_track")
                for param in bin_params:
                    self.register_buffer(f"{param}_median", torch.tensor(stats[param]["median"]))
                    self.register_buffer(f"{param}_mad", torch.tensor(stats[param]["mad"]).clamp(min=1e-6))
                    # v12+: per-param bin edges
                    if self._per_param_bins:
                        centers_p = torch.tensor(stats[param]["bin_centers"])
                        self.register_buffer(f"{param}_log_bin_centers", torch.log(centers_p))
                # Per-param correction scales as buffer tensors for vectorized tanh/arcsinh bounding
                self.register_buffer(
                    "track_scales",
                    torch.tensor([
                        stats["inv_pt"]["correction_scale"],
                        stats["d0"]["correction_scale"],
                        stats["z0"]["correction_scale"],
                        stats["phi0"]["correction_scale"],
                        stats["eta"]["correction_scale"],
                    ]),
                )
                # v12b bugfix: energy slot is routed only to neutrals when neutral_energy_only=true,
                # so it must use the energy_neutral scale. Otherwise (charged+neutral mix), energy_charged.
                energy_scale_key = "energy_neutral" if neutral_energy_only else "energy_charged"
                self.register_buffer(
                    "calo_scales",
                    torch.tensor([
                        stats[energy_scale_key]["correction_scale"],
                        stats["neutral_deta"]["correction_scale"],
                        stats["neutral_dphi"]["correction_scale"],
                    ]),
                )

                # Pre-compute PCHIP cubic Hermite spline coefficients for all params
                if interp_mode == "cubic":
                    import numpy as np

                    for param in bin_params:
                        if self._per_param_bins:
                            log_centers_np = np.log(np.array(stats[param]["bin_centers"]))
                        else:
                            log_centers_np = np.log(np.array(stats["bin_centers"]))
                        for stat in ["median", "mad"]:
                            vals = stats[param][stat]
                            if stat == "mad":
                                vals = [max(v, 1e-6) for v in vals]
                            self.register_buffer(f"{param}_{stat}_pchip_c", pchip_coeffs(log_centers_np, np.array(vals)))

                # v12d F2: per-param bin_mad reference for N1 normalization — median(bin_mad) across bins.
                # Computed once at load; stable across training (bin_stats are frozen yaml values).
                # The param list + binvar map were built earlier in the n_scalar block to keep the
                # scalar_mlp input size consistent. Here we just materialize the reference buffers.
                if use_bin_stats_features:
                    for param in self._bin_stats_feature_params:
                        mad_tensor = self._bin_stat(param, "mad")
                        self.register_buffer(f"_bin_mad_ref_{param}", mad_tensor.median().detach().clone())
            else:
                # Legacy single-head format: per-bin log_corr stats for target_mode='log_mad'
                # (used when not full_perigee — log_corr is always positive so global log/MAD is
                # well-defined; signed perigee params require the per-param path above).
                self.register_buffer("corr_log_median", torch.tensor(stats["log_corr"]["median"]))
                self.register_buffer("corr_log_mad", torch.tensor(stats["log_corr"]["mad"]).clamp(min=1e-6))

                # Pre-compute PCHIP cubic Hermite spline coefficients for interpolation
                if interp_mode == "cubic":
                    import numpy as np

                    log_centers_np = np.log(np.array(stats["bin_centers"]))
                    for buf_name, stat_vals in [
                        ("corr_log_median", stats["log_corr"]["median"]),
                        ("corr_log_mad", [max(v, 1e-6) for v in stats["log_corr"]["mad"]]),
                    ]:
                        self.register_buffer(f"{buf_name}_pchip_c", pchip_coeffs(log_centers_np, np.array(stat_vals)))

        if target_mode == "log_mad" and bin_stats_path is None:
            msg = "bin_stats_path is required when target_mode='log_mad'"
            raise ValueError(msg)

        if use_full_perigee and target_mode != "log_mad":
            msg = "use_full_perigee requires target_mode='log_mad'"
            raise ValueError(msg)

    def _bin_stat(self, param: str, stat: str) -> Tensor:
        """Per-param bin statistic buffer (``median`` or ``mad``)."""
        return getattr(self, f"{param}_{stat}")

    def _pchip(self, param: str, stat: str) -> Tensor | None:
        """PCHIP coefficients for ``param``'s ``stat``, or None if not cubic-interp."""
        return getattr(self, f"{param}_{stat}_pchip_c", None)

    def _bin_log_centers(self, param: str) -> Tensor:
        """Log bin centers — per-param (v12+) or shared (v11)."""
        if self._per_param_bins:
            return getattr(self, f"{param}_log_bin_centers")
        return self.corr_log_bin_centers

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Rewrite incoming state_dict for backward compatibility with older ckpts.

          * Strips the legacy ``v11_`` bin-stat buffer prefix.
          * Drops buffers the class no longer registers (``_DEPRECATED_BUFFER_KEYS``).
        """
        legacy = prefix + self._LEGACY_BIN_BUF_PREFIX
        for k in [k for k in state_dict if k.startswith(legacy)]:
            state_dict[prefix + k[len(legacy):]] = state_dict.pop(k)
        for dead in self._DEPRECATED_BUFFER_KEYS:
            state_dict.pop(prefix + dead, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _interp(self, seed_pt: Tensor, param: str, stat: str) -> Tensor:
        """Interpolate a per-param stat (median/mad) at the given pT values."""
        return bin_interp(
            seed_pt,
            self._bin_stat(param, stat),
            log_centers=self._bin_log_centers(param),
            mode=self.interp_mode,
            pchip_c=self._pchip(param, stat),
            bin_edges=self.corr_bin_edges[1:-1],
            eps=self.eps,
        )

    def _compute_logits_and_mask(self, x: dict[str, Tensor], constituent: str) -> tuple[Tensor, Tensor]:
        """Compute hard hit-assignment mask for a constituent (all within no_grad)."""
        net = self.object_nets[constituent]
        with torch.no_grad():
            mask_tokens = net(x[self.input_object + "_embed"].detach())
            logits = torch.einsum("bnc,bmc->bnm", mask_tokens, x[constituent + "_embed"].detach())
            valid = x[constituent + "_valid"]
            vm = valid.unsqueeze(-2).expand_as(logits)
            logits[~vm] = torch.finfo(logits.dtype).min
            mask = (logits.sigmoid() >= self.pred_threshold) & vm
        return valid, mask

    def _transform_scalars(
        self, *, pid_probs: Tensor, helix_feats: Tensor | None, calo_energy_sums: list[Tensor], hit_counts: Tensor, extra: dict[str, Tensor]
    ) -> Tensor:
        """Apply physics-informed transforms to scalar features and return (B, N, n_scalar).

        Args:
            helix_feats: (B, N, 5) helix params or None when use_track_correction=False.
            extra: dict with mode-dependent tensors (calo_direction, ecal_frac, etc. for v4;
                   split calo features for v5; helix residuals when enabled).
        """
        # Hit counts: log1p
        hit_counts_log1p = torch.log1p(hit_counts)

        parts = [pid_probs]  # (B, N, 6)

        # Helix params (only when tracking enabled)
        if self.use_track_correction:
            d0 = helix_feats[..., 0:1]
            phi0 = helix_feats[..., 1]
            sin_phi0 = torch.sin(phi0).unsqueeze(-1)
            cos_phi0 = torch.cos(phi0).unsqueeze(-1)
            omega = helix_feats[..., 2:3]
            z0 = helix_feats[..., 3:4]
            tan_lambda = helix_feats[..., 4:5]
            parts.extend([d0, sin_phi0, cos_phi0, omega, z0, tan_lambda])  # (B, N, 6)

        if self.use_calo_features:
            calo_log1p = torch.stack([torch.log1p(e) for e in calo_energy_sums], dim=-1)
            parts.append(calo_log1p)  # (B, N, 2)

        parts.append(hit_counts_log1p)  # (B, N, 5 or 3 or 2)

        if self.use_calo_features:
            if self.use_split_calo:
                # Per-calo direction: ecal + hcal separately
                for prefix in ["ecal", "hcal"]:
                    eta = extra[f"{prefix}_eta"]
                    phi = extra[f"{prefix}_phi"]
                    parts.extend([eta.unsqueeze(-1), torch.sin(phi).unsqueeze(-1), torch.cos(phi).unsqueeze(-1)])

                # Energy features (E/seed_pT and track-calo deltas only when tracking)
                parts.extend([
                    extra["ecal_frac"].unsqueeze(-1),
                    torch.log1p(extra["total_calo_E"]).unsqueeze(-1),
                ])
                if self.use_track_correction:
                    parts.extend([
                        torch.log1p(extra["e_ecal_over_seed_pt"]).unsqueeze(-1),
                        torch.log1p(extra["e_hcal_over_seed_pt"]).unsqueeze(-1),
                    ])
                parts.append(torch.log1p(extra["muon_e_sum"]).unsqueeze(-1))

                # Track-calo deltas: per calo type (only when tracking)
                if self.use_track_correction:
                    for prefix in ["ecal", "hcal"]:
                        parts.extend([
                            extra[f"delta_eta_{prefix}"].unsqueeze(-1),
                            extra[f"sin_delta_phi_{prefix}"].unsqueeze(-1),
                            extra[f"cos_delta_phi_{prefix}"].unsqueeze(-1),
                        ])

                # Shower shape
                for prefix in ["ecal", "hcal"]:
                    parts.extend([
                        torch.log1p(extra[f"{prefix}_mean_depth"]).unsqueeze(-1),
                        torch.log1p(extra[f"{prefix}_depth_rms"]).unsqueeze(-1),
                        torch.log1p(extra[f"{prefix}_trans_rms"]).unsqueeze(-1),
                    ])
            else:
                # v4 default: combined calo direction
                calo_direction = extra["calo_direction"]
                calo_eta = calo_direction[..., 0:1]
                calo_phi = calo_direction[..., 1]
                parts.extend([
                    calo_eta,
                    torch.sin(calo_phi).unsqueeze(-1),
                    torch.cos(calo_phi).unsqueeze(-1),
                    extra["ecal_frac"].unsqueeze(-1),
                    torch.log1p(extra["total_calo_E"]).unsqueeze(-1),
                ])
                if self.use_track_correction:
                    parts.append(torch.log1p(extra["e_over_seed_pt"]).unsqueeze(-1))
                parts.append(torch.log1p(extra["muon_e_sum"]).unsqueeze(-1))
                if self.use_track_correction:
                    parts.extend([
                        extra["delta_eta"].unsqueeze(-1),
                        extra["sin_delta_phi"].unsqueeze(-1),
                        extra["cos_delta_phi"].unsqueeze(-1),
                    ])

        if self.use_helix_residuals:
            parts.extend([
                torch.log1p(extra["circle_rms"]).unsqueeze(-1),
                torch.log1p(extra["z_rms"]).unsqueeze(-1),
                torch.log1p(extra["max_circle_res"]).unsqueeze(-1),
                extra["circle_mean"].unsqueeze(-1),
                extra["z_mean"].unsqueeze(-1),
            ])

        if self.use_sagitta:
            parts.append(torch.log1p(extra["sagitta"]).unsqueeze(-1))

        # v12b: 8 extra neutral/calo-focused scalars
        if self.add_v12b_scalars:
            # n_sihit flag — explicit tracked/untracked marker
            n_sihit = hit_counts[..., : len(self.tracker_constituents)].sum(-1)  # (B, N)
            n_sihit_flag = (n_sihit > 0).float().unsqueeze(-1)
            parts.append(n_sihit_flag)

            if self.use_calo_features and self.use_split_calo:
                ecal_frac = extra["ecal_frac"]
                ecal_sum = extra["ecal_sum"]
                hcal_sum = extra["hcal_sum"]
                ecal_eta = extra["ecal_eta"]
                hcal_eta = extra["hcal_eta"]
                ecal_phi = extra["ecal_phi"]
                hcal_phi = extra["hcal_phi"]

                # hcal_frac — symmetric to existing ecal_frac
                parts.append((1.0 - ecal_frac).unsqueeze(-1))

                # Calo centroid Δη and wrapped Δφ (sin/cos pair, consistent with existing track-calo deltas)
                d_eta_calos = ecal_eta - hcal_eta
                d_phi_calos = ecal_phi - hcal_phi
                parts.extend([
                    d_eta_calos.unsqueeze(-1),
                    torch.sin(d_phi_calos).unsqueeze(-1),
                    torch.cos(d_phi_calos).unsqueeze(-1),
                ])

                # Max-hit fraction per subdet — EM/hadronic compactness
                parts.append(extra["ecal_max_hit_frac"].unsqueeze(-1))
                parts.append(extra["hcal_max_hit_frac"].unsqueeze(-1))

                # Signed log1p of |ecal - hcal| — linear-scale imbalance not trivially recovered from log sums
                diff = ecal_sum - hcal_sum
                signed_log1p_diff = torch.sign(diff) * torch.log1p(diff.abs())
                parts.append(signed_log1p_diff.unsqueeze(-1))
            else:
                # If calo features aren't enabled, fill the 7 remaining slots with zeros to keep n_scalar consistent
                zero = torch.zeros_like(n_sihit_flag)
                parts.extend([zero] * 7)

        # v12d F1: binvars. total_calo_ET is the neutral head's bin-lookup key but isn't recoverable
        # from existing features (needs 1/cosh(combined_eta) structure). Combined centroid eta/phi is
        # the "seed" for neutral direction corrections. No normalization needed: ET → log1p; eta is
        # already clamped to [-4, 4]; phi uses (sin, cos) per the existing phi convention at L2876/2913.
        if self.use_binvar_features:
            parts.extend([
                torch.log1p(extra["total_calo_ET"]).unsqueeze(-1),
                extra["combined_centroid_eta"].unsqueeze(-1),
                torch.sin(extra["combined_centroid_phi"]).unsqueeze(-1),
                torch.cos(extra["combined_centroid_phi"]).unsqueeze(-1),
            ])

        # v12d F2: per-param bin_median/bin_mad injected as (feat_bias, feat_spread) pair per param.
        # Bin lookup uses the same (binvar, param) pair as the loss does — seed_pt for track-side,
        # total_calo_ET for calo-side. N1 normalization: both features are O(1) by construction.
        if self.use_bin_stats_features:
            seed_pt = extra["seed_pt"]
            total_calo_ET = extra["total_calo_ET"]
            for p in self._bin_stats_feature_params:
                binvar = seed_pt if self._bin_stats_binvar[p] == "pt" else total_calo_ET
                med = self._interp(binvar, p, "median")
                mad = self._interp(binvar, p, "mad").clamp(min=1e-8)
                mad_ref = getattr(self, f"_bin_mad_ref_{p}").clamp(min=1e-8)
                feat_bias = med / mad
                feat_spread = torch.log(mad / mad_ref)
                parts.extend([feat_bias.unsqueeze(-1), feat_spread.unsqueeze(-1)])

        return torch.cat(parts, dim=-1)  # (B, N, n_scalar)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Get B, N from query embed (always available)
        query_embed_raw = x[self.input_object + "_embed"]
        B, N, _ = query_embed_raw.shape
        device = query_embed_raw.device

        # 1. Tracker hit masks, helix seed, and tracking features
        hit_count_list = []
        if self.use_track_correction:
            tracker_masks = []
            tracker_x, tracker_y, tracker_z = [], [], []
            tracker_time = []

            for constituent in self.tracker_constituents:
                _, mask = self._compute_logits_and_mask(x, constituent)
                tracker_masks.append(mask)
                tracker_x.append(x["inputs"][f"{constituent}_pos.x"] * self.length_unit)
                tracker_y.append(x["inputs"][f"{constituent}_pos.y"] * self.length_unit)
                tracker_z.append(x["inputs"][f"{constituent}_pos.z"] * self.length_unit)
                time_key = f"{constituent}_time"
                if time_key in x["inputs"]:
                    tracker_time.append(x["inputs"][time_key])

            mask_tracker = torch.cat(tracker_masks, dim=-1)
            x_m = torch.cat(tracker_x, dim=-1)
            y_m = torch.cat(tracker_y, dim=-1)
            z_m = torch.cat(tracker_z, dim=-1)
            t_m = torch.cat(tracker_time, dim=-1) if tracker_time else None

            M_tracker = mask_tracker.shape[-1]

            # Build vtxd-only z_valid mask for z regression (vtxd hits have
            # less multiple scattering → much better z0/eta). Falls back to
            # all tracker hits when a particle has < 3 vtxd hits.
            # Skipped in legacy mode for v7-era ckpt reproducibility.
            z_valid_mask = None
            if not self.use_legacy_helix_seed and len(tracker_masks) >= 2:
                n_vtxd = tracker_masks[0].shape[-1]
                vtxd_only = mask_tracker.clone()
                vtxd_only[..., n_vtxd:] = False
                enough_vtxd = vtxd_only.sum(dim=-1, keepdim=True) >= 3
                z_valid_mask = torch.where(enough_vtxd, vtxd_only, mask_tracker)

            # 2. Helix seed (detached). Two modes:
            #   * legacy: reproduces the v7-era call exactly (4 positional args + min_hits/eps,
            #     no z_valid/time/z_phi_sort/sort_mode overrides). Required for v7_ablation ckpts.
            #   * non-legacy: uses ``helix_sort_mode`` (time-aware ``"hybrid"`` by default) and the
            #     vtxd-only ``z_valid_mask`` — halves catastrophic-tail rate post-v11.
            if self.use_legacy_helix_seed:
                with torch.no_grad():
                    helix = helix_seed_3pt(
                        x_m.unsqueeze(1).expand(B, N, M_tracker),
                        y_m.unsqueeze(1).expand(B, N, M_tracker),
                        z_m.unsqueeze(1).expand(B, N, M_tracker),
                        mask_tracker,
                        min_hits=self.seed_min_hits,
                        eps=self.eps,
                    )
            else:
                sort_mode = self.helix_sort_mode
                time_exp = t_m.unsqueeze(1).expand(B, N, M_tracker) if (t_m is not None and sort_mode in ("hybrid", "time")) else None
                with torch.no_grad():
                    helix = helix_seed_3pt(
                        x_m.unsqueeze(1).expand(B, N, M_tracker),
                        y_m.unsqueeze(1).expand(B, N, M_tracker),
                        z_m.unsqueeze(1).expand(B, N, M_tracker),
                        mask_tracker,
                        min_hits=self.seed_min_hits,
                        eps=self.eps,
                        z_valid=z_valid_mask,
                        z_phi_sort=False,
                        time=time_exp,
                        sort_mode=sort_mode,
                    )

            helix_feats = torch.stack([helix["d0"], helix["phi0"], helix["omega"], helix["z0"], helix["tan_lambda"]], dim=-1).detach()

            # 3. Hit counts and seed validity
            for idx, _constituent in enumerate(self.tracker_constituents):
                hit_count_list.append(tracker_masks[idx].sum(-1).float().detach())

            n_tracker = sum(hit_count_list)  # (B, N)
            seed_valid = n_tracker >= self.seed_min_hits  # (B, N) — seed produced for any particle with >= seed_min_hits
            loss_valid = n_tracker >= self.loss_min_hits  # (B, N) — only contribute to track loss when >= loss_min_hits

            # Seed q/pT: charge_sign / pT, where charge_sign = -sign(omega) * sign(B_z)
            with torch.no_grad():
                seed_pt = (0.3 * self.b_field / (helix["omega"].abs() + self.eps)).detach()
                b_sign = -1.0 if self.b_field > 0 else 1.0
                seed_qpt = (b_sign * torch.sign(helix["omega"]) / seed_pt).detach()  # (B, N)
                seed_qpt = seed_qpt.masked_fill(~seed_valid, 0.0)

            # Zero helix_feats for invalid seeds
            helix_feats = helix_feats * seed_valid.unsqueeze(-1).float()

            # 3b. Helix residuals (v5)
            if self.use_helix_residuals:
                with torch.no_grad():
                    helix_res = compute_helix_residuals(helix, x_m, y_m, z_m, mask_tracker, eps=self.eps)
        else:
            # No tracking: dummy values
            seed_valid = torch.zeros(B, N, dtype=torch.bool, device=device)
            loss_valid = torch.zeros(B, N, dtype=torch.bool, device=device)
            seed_qpt = torch.zeros(B, N, device=device)
            seed_pt = torch.zeros(B, N, device=device)
            helix_feats = None
            helix = {"eta": torch.zeros(B, N, device=device), "phi0": torch.zeros(B, N, device=device)}

        # 4. Calo energy sums, calo hit counts, and calo masks (detached)
        calo_energy_sums = []
        calo_masks = {}
        calo_direction = None
        extra_scalars: dict[str, Tensor] = {}
        calo_hit_count_start = len(hit_count_list)

        if self.use_calo_features:
            for constituent in self.calo_constituents:
                _, mask_calo = self._compute_logits_and_mask(x, constituent)
                calo_masks[constituent] = mask_calo
                hit_count_list.append(mask_calo.sum(-1).float().detach())
                energy = x["inputs"][f"{constituent}_{self.calo_energy_field}"]
                energy_sum = (mask_calo.float() * energy.unsqueeze(1)).sum(-1).detach()
                calo_energy_sums.append(energy_sum)

            n_calo = sum(hit_count_list[calo_hit_count_start:])  # (B, N)
            has_calo = n_calo >= self.min_calo_hits

            # 5. Calo direction and energy features (detached)
            with torch.no_grad():
                ecal_sum = torch.zeros(B, N, device=device)
                hcal_sum = torch.zeros(B, N, device=device)
                for i, c in enumerate(self.calo_constituents):
                    if c.startswith("e"):
                        ecal_sum = ecal_sum + calo_energy_sums[i]
                    else:
                        hcal_sum = hcal_sum + calo_energy_sums[i]
                total_calo_E = ecal_sum + hcal_sum
                ecal_frac = ecal_sum / total_calo_E.clamp(min=self.eps)
                ecal_frac = ecal_frac.masked_fill(total_calo_E < self.eps, 0.0)

                if self.use_track_correction:
                    seed_pt_safe = seed_pt.clamp(min=self.eps)
                    track_eta = helix["eta"].detach()
                    track_phi = helix["phi0"].detach()

            # v12b bugfix: compute total_calo_ET = total_calo_E / cosh(combined_eta) for use as the
            # bin-stats binning variable for neutral params (seed_pt is garbage for no-tracker queries).
            # Combined centroid matches compute_bin_stats_5params.compute_calo_info exactly
            # (energy-weighted position → angle). Exposed in result later alongside calo_E_sum.
            with torch.no_grad():
                combined_dir = compute_calo_direction(
                    x,
                    calo_masks,
                    B,
                    N,
                    calo_constituents=self.calo_constituents,
                    calo_energy_field=self.calo_energy_field,
                    input_object=self.input_object,
                    eps=self.eps,
                )  # (B, N, 2) = (eta, phi)
                combined_centroid_eta = combined_dir[..., 0]
                combined_centroid_phi = combined_dir[..., 1]
                total_calo_ET = total_calo_E / torch.cosh(combined_centroid_eta).clamp(min=1.0)

            if self.use_split_calo:
                # v5: per-constituent calo direction + shower shape
                with torch.no_grad():
                    calo_split = compute_calo_direction_split(
                        x,
                        calo_masks,
                        calo_constituents=self.calo_constituents,
                        calo_energy_field=self.calo_energy_field,
                        input_object=self.input_object,
                        eps=self.eps,
                    )

                    extra_scalars = {
                        "ecal_frac": ecal_frac,
                        "total_calo_E": total_calo_E,
                        "ecal_sum": ecal_sum,  # v12b — used for hcal_frac/diff features
                        "hcal_sum": hcal_sum,  # v12b
                        # v12d: binvars threaded through to _transform_scalars
                        "total_calo_ET": total_calo_ET,
                        "combined_centroid_eta": combined_centroid_eta,
                        "combined_centroid_phi": combined_centroid_phi,
                        "seed_pt": seed_pt,
                    }

                    # E/p ratios and track-calo deltas (only when tracking)
                    if self.use_track_correction:
                        e_ecal_over_seed_pt = ecal_sum / seed_pt_safe
                        e_ecal_over_seed_pt = e_ecal_over_seed_pt.masked_fill(~seed_valid, 0.0)
                        e_hcal_over_seed_pt = hcal_sum / seed_pt_safe
                        e_hcal_over_seed_pt = e_hcal_over_seed_pt.masked_fill(~seed_valid, 0.0)
                        extra_scalars["e_ecal_over_seed_pt"] = e_ecal_over_seed_pt
                        extra_scalars["e_hcal_over_seed_pt"] = e_hcal_over_seed_pt
                        no_both = ~(seed_valid & has_calo)

                    for prefix in ["ecal", "hcal"]:
                        extra_scalars[f"{prefix}_eta"] = calo_split[f"{prefix}_eta"]
                        extra_scalars[f"{prefix}_phi"] = calo_split[f"{prefix}_phi"]
                        extra_scalars[f"{prefix}_mean_depth"] = calo_split[f"{prefix}_mean_depth"]
                        extra_scalars[f"{prefix}_depth_rms"] = calo_split[f"{prefix}_depth_rms"]
                        extra_scalars[f"{prefix}_trans_rms"] = calo_split[f"{prefix}_trans_rms"]
                        extra_scalars[f"{prefix}_max_hit_frac"] = calo_split[f"{prefix}_max_hit_frac"]  # v12b

                        if self.use_track_correction:
                            d_eta = calo_split[f"{prefix}_eta"] - track_eta
                            d_phi = calo_split[f"{prefix}_phi"] - track_phi
                            d_eta = d_eta.masked_fill(no_both, 0.0)
                            extra_scalars[f"delta_eta_{prefix}"] = d_eta
                            extra_scalars[f"sin_delta_phi_{prefix}"] = torch.sin(d_phi).masked_fill(no_both, 0.0)
                            extra_scalars[f"cos_delta_phi_{prefix}"] = torch.cos(d_phi).masked_fill(no_both, 0.0)
            else:
                # v4: combined calo direction
                with torch.no_grad():
                    calo_direction = compute_calo_direction(
                        x,
                        calo_masks,
                        B,
                        N,
                        calo_constituents=self.calo_constituents,
                        calo_energy_field=self.calo_energy_field,
                        input_object=self.input_object,
                        eps=self.eps,
                    )

                    extra_scalars = {
                        "calo_direction": calo_direction,
                        "ecal_frac": ecal_frac,
                        "total_calo_E": total_calo_E,
                        # v12d: binvars threaded through to _transform_scalars
                        "total_calo_ET": total_calo_ET,
                        "combined_centroid_eta": combined_centroid_eta,
                        "combined_centroid_phi": combined_centroid_phi,
                        "seed_pt": seed_pt,
                    }

                    if self.use_track_correction:
                        e_over_seed_pt = total_calo_E / seed_pt_safe
                        e_over_seed_pt = e_over_seed_pt.masked_fill(~seed_valid, 0.0)
                        extra_scalars["e_over_seed_pt"] = e_over_seed_pt

                        delta_eta = calo_direction[..., 0] - track_eta
                        delta_phi = calo_direction[..., 1] - track_phi
                        sin_delta_phi = torch.sin(delta_phi)
                        cos_delta_phi = torch.cos(delta_phi)
                        no_both = ~(seed_valid & has_calo)
                        extra_scalars["delta_eta"] = delta_eta.masked_fill(no_both, 0.0)
                        extra_scalars["sin_delta_phi"] = sin_delta_phi.masked_fill(no_both, 0.0)
                        extra_scalars["cos_delta_phi"] = cos_delta_phi.masked_fill(no_both, 0.0)

            # 5c. Muon features (detached)
            muon_E_sum = torch.zeros(B, N, device=device)
            for constituent in self.muon_constituents:
                _, muon_mask = self._compute_logits_and_mask(x, constituent)
                hit_count_list.append(muon_mask.sum(-1).float().detach())
                muon_energy = x["inputs"][f"{constituent}_{self.calo_energy_field}"]
                muon_E_sum = muon_E_sum + (muon_mask.float() * muon_energy.unsqueeze(1)).sum(-1).detach()

            extra_scalars["muon_e_sum"] = muon_E_sum
        else:
            # No calo features: skip all calo/muon computation
            has_calo = torch.zeros(B, N, dtype=torch.bool, device=device)

        # Stack hit_counts (tracker only when no calo, tracker + calo + muon otherwise)
        hit_counts = torch.stack(hit_count_list, dim=-1)

        # 5d. Helix residuals into extra_scalars (v5)
        if self.use_helix_residuals:
            for k in ["circle_rms", "z_rms", "max_circle_res", "circle_mean", "z_mean"]:
                extra_scalars[k] = helix_res[k].detach()

        # 5e. Sagitta (v9a)
        if self.use_sagitta and self.use_track_correction:
            extra_scalars["sagitta"] = helix["sagitta"].detach()

        # 6. PID probs (detached) — from x["class_probs"] cached by maskformer forward
        pid_probs = x["class_probs"].detach() if "class_probs" in x else torch.zeros(B, N, self.num_pid_classes, device=device)

        # 7. Query embed (detached)
        query_embed = x[self.input_object + "_embed"].detach()

        # 8. Force fp32 for trunk + heads
        with torch.autocast(device_type="cuda", enabled=False):
            scalar_feats = self._transform_scalars(
                pid_probs=pid_probs.float(),
                helix_feats=helix_feats.float() if helix_feats is not None else None,
                calo_energy_sums=[e.float() for e in calo_energy_sums],
                hit_counts=hit_counts.float(),
                extra={k: v.float() for k, v in extra_scalars.items()},
            )

            # Apply loaded mean/std normalization — identity when buffers are defaults.
            scalar_feats = (scalar_feats - self.scalar_mean) / self.scalar_std.clamp(min=self.eps)

            # Scalar MLP
            scalar_out = self.scalar_mlp(scalar_feats)

            # Concat with query_embed, LayerNorm, trunk
            trunk_input = torch.cat([query_embed.float(), scalar_out], dim=-1)
            trunk_input = self.trunk_norm(trunk_input)

            trunk_out = self.trunk_net(trunk_input)
            gate_adapted = self.gate_adapter(trunk_out)

            if self.use_track_correction:
                track_adapted = self.track_adapter(trunk_out)
            if self.use_calo_features:
                energy_adapted = self.energy_adapter(trunk_out)

            # Track correction head
            if self.use_track_correction:
                if self.use_full_perigee:
                    # v11/v11b: 5-param output with per-param bounding
                    activation = torch.arcsinh if self.use_arcsinh else torch.tanh
                    if self.split_track_head_per_param:
                        # v12b: late-split per-param heads → concat outputs
                        track_head_out = torch.cat([h(track_adapted) for h in self.track_heads], dim=-1)  # (B, N, 5)
                    else:
                        track_head_out = self.track_head(track_adapted)  # (B, N, 5)
                    # Only apply per-param correction scales on the tanh path — arcsinh is
                    # self-bounded via log growth, and the tanh-tuned scales (12-57) would
                    # amplify its unbounded head outputs into 10^5-10^7 rel_MAE spikes.
                    if self.use_arcsinh:
                        track_corrections = activation(track_head_out)
                    else:
                        track_corrections = self.track_scales * activation(track_head_out)
                    log_correction = track_corrections[:, :, 0]  # inv_pt correction for pred_qpt compat
                elif self.target_mode == "log_mad":
                    log_correction = self.track_head(track_adapted).squeeze(-1)  # unbounded for MAD-normalized targets
                else:
                    log_correction = self.correction_scale * torch.tanh(self.track_head(track_adapted).squeeze(-1))
            else:
                log_correction = torch.zeros(B, N, device=device)

            # Energy / calo head
            if self.use_calo_features:
                if self.use_calo_head:
                    # v11/v11b: 3-param output (energy, neutral_deta, neutral_dphi) with per-param bounding
                    activation = torch.arcsinh if self.use_arcsinh else torch.tanh
                    if self.split_energy_head_per_param:
                        # v12b: late-split per-param heads → concat outputs
                        energy_head_out = torch.cat([h(energy_adapted) for h in self.energy_heads], dim=-1)  # (B, N, 3)
                    else:
                        energy_head_out = self.energy_head(energy_adapted)  # (B, N, 3)
                    # Apply per-param calo scales on tanh only (see track branch above for rationale).
                    if self.use_arcsinh:
                        calo_corrections = activation(energy_head_out)
                    else:
                        calo_corrections = self.calo_scales * activation(energy_head_out)
                    log_E = calo_corrections[:, :, 0]  # energy correction for compat
                else:
                    log_E = self.log_e_scale * torch.tanh(self.energy_head(energy_adapted).squeeze(-1))  # (B, N)
            else:
                log_E = torch.zeros(B, N, device=device)

            # Gate: logits {positive, negative}
            gate_logits = self.gate_head(gate_adapted)  # (B, N, 2) — charge sign logits {+, -}

            # Corrected q/pT (sign still from seed during forward; gate sign applied in predict)
            pred_qpt = seed_qpt.float() * torch.exp(log_correction)

        # Seed eta (from helix fit, needed for eta regression target)
        seed_eta = helix["eta"].detach()

        # Routing info
        gate_charged = torch.ones(B, N, dtype=torch.bool, device=device) if not self.use_track_correction else seed_valid
        gate_neutral = has_calo

        result = {
            self.log_correction_key: log_correction,  # (B, N)
            self.pred_qpt_key: pred_qpt,  # (B, N)
            self.log_E_key: log_E,  # (B, N)
            self.gate_logits_key: gate_logits,  # (B, N, 2) — charge sign logits {+, -}
            f"{self.output_object}_seed_qpt": seed_qpt,  # (B, N)
            f"{self.output_object}_seed_eta": seed_eta,  # (B, N)
            f"{self.output_object}_seed_valid": seed_valid,  # (B, N) — seed available (>= seed_min_hits)
            f"{self.output_object}_loss_valid": loss_valid,  # (B, N) — contributes to track loss (>= loss_min_hits)
            f"{self.output_object}_gate_charged": gate_charged,  # (B, N)
            f"{self.output_object}_gate_neutral": gate_neutral,  # (B, N)
            f"{self.output_object}_has_calo": has_calo,  # (B, N)
        }

        # Calo direction output (for predict/eval)
        if not self.use_calo_features:
            result[f"{self.output_object}_calo_direction"] = seed_qpt.new_zeros(B, N, 2)
        elif calo_direction is not None:
            result[f"{self.output_object}_calo_direction"] = calo_direction
        else:
            # v5 split mode: store combined direction from ecal for compatibility
            ecal_eta = extra_scalars.get("ecal_eta", seed_qpt.new_zeros(B, N))
            ecal_phi = extra_scalars.get("ecal_phi", seed_qpt.new_zeros(B, N))
            result[f"{self.output_object}_calo_direction"] = torch.stack([ecal_eta, ecal_phi], dim=-1)

        # v5: store PID probs for routing in predict()
        result[f"{self.output_object}_pid_probs"] = pid_probs

        # v11: store per-param corrections and seeds
        if self.use_full_perigee:
            result[f"{self.output_object}_track_corrections"] = track_corrections  # (B, N, 5)
            seed_inv_pt = (helix["omega"].abs() / (0.3 * self.b_field)).detach()
            result[f"{self.output_object}_seed_inv_pt"] = seed_inv_pt
            result[f"{self.output_object}_seed_d0"] = helix["d0"].detach()
            result[f"{self.output_object}_seed_z0"] = helix["z0"].detach()
            result[f"{self.output_object}_seed_phi0"] = helix["phi0"].detach()
        if self.use_calo_head:
            result[f"{self.output_object}_calo_corrections"] = calo_corrections  # (B, N, 3)
            result[f"{self.output_object}_calo_E_sum"] = total_calo_E.detach()
            # v12b bugfix: expose binning variables used by loss()/predict() for neutral _interp.
            result[f"{self.output_object}_total_calo_E"] = total_calo_E.detach()
            result[f"{self.output_object}_total_calo_ET"] = total_calo_ET.detach()
            if calo_direction is not None:
                result[f"{self.output_object}_calo_centroid_eta"] = calo_direction[..., 0].detach()
                result[f"{self.output_object}_calo_centroid_phi"] = calo_direction[..., 1].detach()
            else:
                ecal_eta = extra_scalars.get("ecal_eta", seed_qpt.new_zeros(B, N))
                ecal_phi = extra_scalars.get("ecal_phi", seed_qpt.new_zeros(B, N))
                result[f"{self.output_object}_calo_centroid_eta"] = ecal_eta.detach()
                result[f"{self.output_object}_calo_centroid_phi"] = ecal_phi.detach()

        # v11b: routing masks and track-derived energy
        if self.neutral_energy_only:
            result[f"{self.output_object}_n_tracker"] = n_tracker  # (B, N) float — pred sihit count
            # v12g: export pred calo-hit count for truth-gating loss path. Not redundant with has_calo
            # (boolean) since we need the count to threshold at min_calo_hits != has_calo's threshold.
            if self.use_calo_features:
                result[f"{self.output_object}_n_calo"] = n_calo  # (B, N) float — pred ecal+hcal count
            # v12c: broaden neutral_eligible from n_tracker==0 to ~seed_valid so the
            # neutral head is trained on the whole domain it is applied to at inference
            # (covers queries with 1-2 accidental sihits — photons most often).
            neutral_eligible = (~seed_valid) & (total_calo_E > self.neutral_calo_threshold)
            result[f"{self.output_object}_neutral_eligible"] = neutral_eligible

        if self.use_aux_energy_loss and self.use_full_perigee:
            # Compute E_track from denormalized corrections (gradient flows through track_corrections)
            inv_pt_med = self._interp(seed_pt, "inv_pt", "median")
            inv_pt_mad = self._interp(seed_pt, "inv_pt", "mad")
            pred_inv_pt_fwd = seed_inv_pt * torch.exp(track_corrections[:, :, 0] * inv_pt_mad + inv_pt_med)
            # v12b: physical pT cap at 365 GeV (CLD sample √s=365 — kinematic upper bound on any particle's pT).
            # Prevents pathological near-zero seeds from producing 1e8 GeV E_track through 1/eps = 1e8.
            pred_pT_fwd = 1.0 / pred_inv_pt_fwd.clamp(min=1.0 / 365.0)

            eta_med = self._interp(seed_pt, "eta", "median")
            eta_mad = self._interp(seed_pt, "eta", "mad")
            pred_eta_fwd = seed_eta + track_corrections[:, :, 4] * eta_mad + eta_med

            m_pi = 0.13957  # GeV, pion mass assumption
            E_track = torch.sqrt((pred_pT_fwd * torch.cosh(pred_eta_fwd)) ** 2 + m_pi**2)
            result[f"{self.output_object}_E_track"] = E_track  # (B, N) — carries gradient

        return result

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            target_valid = targets[f"{self.target_object}_valid"]
            is_charged = targets[f"{self.target_object}_is_charged"].bool()

            seed_valid = outputs[f"{self.output_object}_seed_valid"]
            # v12b: track losses use loss_valid (>= loss_min_hits); charge/eta losses keep seed_valid
            loss_valid = outputs.get(f"{self.output_object}_loss_valid", seed_valid)
            has_calo = outputs[f"{self.output_object}_has_calo"]

            # --- Track correction loss ---
            target_qpt = targets[f"{self.target_object}_mom.qopt"].float()

            if self.use_full_perigee and self.use_track_correction:
                # v11: 5-param track correction loss with per-param Barron
                track_corrections = outputs[f"{self.output_object}_track_corrections"].float()  # (B, N, 5)
                seed_qpt = outputs[f"{self.output_object}_seed_qpt"].float()
                seed_inv_pt = outputs[f"{self.output_object}_seed_inv_pt"].float()
                seed_d0 = outputs[f"{self.output_object}_seed_d0"].float()
                seed_z0 = outputs[f"{self.output_object}_seed_z0"].float()
                seed_phi0 = outputs[f"{self.output_object}_seed_phi0"].float()
                seed_eta = outputs[f"{self.output_object}_seed_eta"].float()

                track_mask = target_valid & is_charged & loss_valid  # v12b: loss_valid = >= loss_min_hits
                # v12g: optionally also require truth n_sihit >= truth_loss_sihit_min so that fake-assignment
                # charged tracks (PRED mask over-clusters → loss_valid but truth has <5 hits) don't contaminate training.
                if self.use_truth_hit_gating:
                    t_sihit = targets[f"{self.target_object}_vtxd_valid"].long().sum(-1) + targets[f"{self.target_object}_trkr_valid"].long().sum(-1)
                    track_mask = track_mask & (t_sihit >= self.truth_loss_sihit_min)
                seed_pt = (1.0 / seed_qpt.abs().clamp(min=self.eps)).detach()

                # Truth from perigee
                truth_omega = targets[f"{self.target_object}_perigee.omega"].float()
                truth_d0 = targets[f"{self.target_object}_perigee.d0"].float()
                truth_z0 = targets[f"{self.target_object}_perigee.z0"].float()
                truth_phi0 = targets[f"{self.target_object}_perigee.phi0"].float()
                truth_tanlam = targets[f"{self.target_object}_perigee.tan_lambda"].float()
                truth_inv_pt = (truth_omega.abs() / (0.3 * self.b_field)).detach()
                truth_eta = torch.asinh(truth_tanlam).detach()

                # Per-param targets and losses
                track_param_losses = []
                track_param_specs = [
                    ("inv_pt", "log"),  # log-ratio: log(truth/seed)
                    ("d0", "add"),  # additive: truth - seed
                    ("z0", "add"),
                    ("phi0", "add_wrap"),  # wrapped additive
                    ("eta", "add"),
                ]
                truth_vals = [truth_inv_pt, truth_d0, truth_z0, truth_phi0, truth_eta]
                seed_vals = [seed_inv_pt, seed_d0, seed_z0, seed_phi0, seed_eta]

                for i, ((param_name, mode), truth_v, seed_v) in enumerate(zip(track_param_specs, truth_vals, seed_vals, strict=False)):
                    if mode == "log":
                        raw_target = torch.log(truth_v.clamp(min=self.eps) / seed_v.clamp(min=self.eps))
                    elif mode == "add_wrap":
                        raw_target = ((truth_v - seed_v + torch.pi) % (2 * torch.pi)) - torch.pi
                    else:
                        raw_target = truth_v - seed_v

                    pn = param_name
                    median = self._interp(seed_pt, pn, "median")
                    mad = self._interp(seed_pt, pn, "mad")
                    normalized = (raw_target - median) / mad.clamp(min=self.eps)

                    param_loss = self.barron_losses[param_name](track_corrections[:, :, i] - normalized)
                    track_param_losses.append(param_loss)

                # v11b: aux energy loss from track-derived E_track
                if self.use_aux_energy_loss:
                    E_track = outputs[f"{self.output_object}_E_track"]  # (B, N), carries gradient
                    target_energy = targets[f"{self.target_object}_{self.energy_target}"].float()
                    raw_aux_target = torch.log(target_energy.clamp(min=self.eps) / E_track.clamp(min=self.eps))
                    aux_med = self._interp(seed_pt, "energy_track", "median")
                    aux_mad = self._interp(seed_pt, "energy_track", "mad")
                    aux_normalized = (raw_aux_target - aux_med) / aux_mad.clamp(min=self.eps)
                    aux_loss_pq = self.barron_losses["energy_track"](aux_normalized)
                    track_param_losses.append(aux_loss_pq)

                # Per-param outlier masking (v12) or sum-based masking (v11b fallback)
                _track_param_names = ["inv_pt", "d0", "z0", "phi0", "eta"]
                if self.use_aux_energy_loss:
                    _track_param_names.append("energy_track")

                # v12b: per-param loss values (grad-bearing) for monitor_* logging AND v12c Kendall weighting.
                track_param_raw_losses: dict[str, torch.Tensor] = {}
                if self.use_outlier_masking and self.per_param_outlier_pct and track_mask.any():
                    # v12: mask each param independently at its own percentile via shared helper.
                    raw_losses = dict(zip(_track_param_names, track_param_losses, strict=False))
                    valid_masks = dict.fromkeys(_track_param_names, track_mask)
                    track_loss, track_param_raw_losses = self._apply_per_param_outlier_mask(raw_losses, valid_masks, _track_param_names)
                elif self.use_outlier_masking and track_mask.any():
                    # v11b fallback: mask on sum of all params
                    track_loss_per_query = sum(track_param_losses)
                    valid_track_losses = track_loss_per_query[track_mask]
                    if valid_track_losses.numel() > 20:
                        thresh = torch.quantile(valid_track_losses.detach(), self.outlier_percentile)
                        keep_track = track_mask & (track_loss_per_query.detach() <= thresh)
                    else:
                        keep_track = track_mask
                    if keep_track.any():
                        track_loss = track_loss_per_query[keep_track].mean()
                    else:
                        track_loss = target_qpt.new_tensor(0.0)
                else:
                    track_loss_per_query = sum(track_param_losses)
                    if track_mask.any():
                        track_loss = track_loss_per_query[track_mask].mean()
                    else:
                        track_loss = target_qpt.new_tensor(0.0)

            elif self.use_track_correction:
                log_correction = outputs[self.log_correction_key].float()  # (B, N)
                seed_qpt = outputs[f"{self.output_object}_seed_qpt"].float()

                # Mask: valid charged with seed (correction is magnitude-only, sign handled by gate)
                track_mask = target_valid & is_charged & loss_valid  # v12b: loss_valid = >= loss_min_hits
                # v12g: optionally require truth n_sihit >= truth_loss_sihit_min (see __init__ docstring).
                if self.use_truth_hit_gating:
                    t_sihit = targets[f"{self.target_object}_vtxd_valid"].long().sum(-1) + targets[f"{self.target_object}_trkr_valid"].long().sum(-1)
                    track_mask = track_mask & (t_sihit >= self.truth_loss_sihit_min)

                # Target: log-space correction, optionally MAD-normalized per pT bin
                target_corr = torch.log(target_qpt.abs().clamp(min=self.eps) / seed_qpt.abs().clamp(min=self.eps))

                if self.target_mode == "log_mad":
                    seed_pt = (1.0 / seed_qpt.abs().clamp(min=self.eps)).detach()
                    log_median = bin_interp(
                        seed_pt,
                        self.corr_log_median,
                        log_centers=self.corr_log_bin_centers,
                        mode=self.interp_mode,
                        pchip_c=getattr(self, "corr_log_median_pchip_c", None),
                        bin_edges=self.corr_bin_edges[1:-1],
                        eps=self.eps,
                    )
                    log_mad = bin_interp(
                        seed_pt,
                        self.corr_log_mad,
                        log_centers=self.corr_log_bin_centers,
                        mode=self.interp_mode,
                        pchip_c=getattr(self, "corr_log_mad_pchip_c", None),
                        bin_edges=self.corr_bin_edges[1:-1],
                        eps=self.eps,
                    )
                    target_corr = (target_corr - log_median) / log_mad.clamp(min=self.eps)

                if self.use_barron_loss:
                    track_loss_per_query = self.track_barron(log_correction - target_corr)
                else:
                    track_loss_per_query = torch.nn.functional.smooth_l1_loss(
                        log_correction, target_corr, reduction="none", beta=self.track_loss_beta
                    )
                if track_mask.any():
                    track_loss = track_loss_per_query[track_mask].mean()
                else:
                    track_loss = target_qpt.new_tensor(0.0)
            else:
                track_loss = target_qpt.new_tensor(0.0)

            # --- Energy / calo loss ---
            if self.use_calo_head:
                # v11/v11b: correction-based energy + neutral direction
                calo_corrections = outputs[f"{self.output_object}_calo_corrections"].float()  # (B, N, 3)
                calo_E_sum = outputs[f"{self.output_object}_calo_E_sum"].float()
                target_energy = targets[f"{self.target_object}_{self.energy_target}"].float()
                seed_qpt = outputs[f"{self.output_object}_seed_qpt"].float()
                seed_pt = (1.0 / seed_qpt.abs().clamp(min=self.eps)).detach()
                # v12b bugfix: neutral params (energy_neutral, neutral_deta, neutral_dphi) bin by
                # total_calo_ET = total_calo_E / cosh(combined_centroid_eta) — matches the
                # bin-stats generation (compute_bin_stats_5params.compute_calo_info + calo_ET transform).
                # Fall back to total_calo_E, then seed_pt, for legacy ckpts that predate this.
                total_calo_ET = outputs.get(
                    f"{self.output_object}_total_calo_ET",
                    outputs.get(f"{self.output_object}_total_calo_E", seed_pt),
                ).float()

                if self.neutral_energy_only:
                    # v11b: energy head only for neutrals (n_sihit==0 & calo_E>threshold)
                    neutral_eligible = outputs[f"{self.output_object}_neutral_eligible"]
                    energy_mask = target_valid & ~is_charged & neutral_eligible & (target_energy > self.eps)
                    # v12g: truth-based gating — activate head on true neutrals OR charged-with-no-PRED-sihit,
                    # AND require BOTH truth_n_calo >= K AND pred_n_calo >= K (mask-quality + deposit).
                    # Replaces the pred-based neutral_eligible filter at training time. Inference unaffected.
                    if self.use_truth_hit_gating:
                        t_calo = targets[f"{self.target_object}_ecal_valid"].long().sum(-1) + targets[f"{self.target_object}_hcal_valid"].long().sum(
                            -1
                        )
                        p_sihit = outputs[f"{self.output_object}_n_tracker"]
                        p_calo = outputs[f"{self.output_object}_n_calo"]
                        truth_neutral_like = (~is_charged) | (is_charged & (p_sihit == 0))
                        energy_mask = (
                            target_valid
                            & truth_neutral_like
                            & (t_calo >= self.truth_loss_calo_min)
                            & (p_calo >= self.truth_loss_calo_min)
                            & (target_energy > self.eps)
                        )
                    raw_energy_target = torch.log(target_energy.clamp(min=self.eps) / calo_E_sum.clamp(min=self.eps))
                    median_ne = self._interp(total_calo_ET, "energy_neutral", "median")
                    mad_ne = self._interp(total_calo_ET, "energy_neutral", "mad")
                    energy_normalized = (raw_energy_target - median_ne) / mad_ne.clamp(min=self.eps)
                else:
                    # v11: energy for both charged and neutral with PID-routed bin stats
                    energy_mask = target_valid & has_calo & (calo_E_sum > self.eps) & (target_energy > self.eps)
                    raw_energy_target = torch.log(target_energy.clamp(min=self.eps) / calo_E_sum.clamp(min=self.eps))
                    median_ch = self._interp(seed_pt, "energy_charged", "median")
                    mad_ch = self._interp(seed_pt, "energy_charged", "mad")
                    median_ne = self._interp(total_calo_ET, "energy_neutral", "median")
                    mad_ne = self._interp(total_calo_ET, "energy_neutral", "mad")
                    energy_median = torch.where(is_charged, median_ch, median_ne)
                    energy_mad = torch.where(is_charged, mad_ch, mad_ne).clamp(min=self.eps)
                    energy_normalized = (raw_energy_target - energy_median) / energy_mad

                energy_loss_pq = self.barron_losses["energy"](calo_corrections[:, :, 0] - energy_normalized)

                # Neutral direction loss: truth - centroid
                centroid_eta = outputs[f"{self.output_object}_calo_centroid_eta"].float()
                centroid_phi = outputs[f"{self.output_object}_calo_centroid_phi"].float()
                target_eta_dir = targets[f"{self.target_object}_mom.eta"].float()
                target_phi_dir = targets[f"{self.target_object}_mom.phi"].float()

                if self.neutral_energy_only:
                    neutral_mask = target_valid & ~is_charged & neutral_eligible
                    # v12g: truth-based gating (same scheme as energy_mask above).
                    if self.use_truth_hit_gating:
                        t_calo = targets[f"{self.target_object}_ecal_valid"].long().sum(-1) + targets[f"{self.target_object}_hcal_valid"].long().sum(
                            -1
                        )
                        p_sihit = outputs[f"{self.output_object}_n_tracker"]
                        p_calo = outputs[f"{self.output_object}_n_calo"]
                        truth_neutral_like = (~is_charged) | (is_charged & (p_sihit == 0))
                        neutral_mask = target_valid & truth_neutral_like & (t_calo >= self.truth_loss_calo_min) & (p_calo >= self.truth_loss_calo_min)
                else:
                    neutral_mask = target_valid & ~is_charged & has_calo

                # deta (v12b bugfix: bin by total_calo_E, not seed_pt)
                raw_deta = target_eta_dir - centroid_eta
                deta_median = self._interp(total_calo_ET, "neutral_deta", "median")
                deta_mad = self._interp(total_calo_ET, "neutral_deta", "mad").clamp(min=self.eps)
                deta_normalized = (raw_deta - deta_median) / deta_mad
                deta_loss_pq = self.barron_losses["neutral_deta"](calo_corrections[:, :, 1] - deta_normalized)
                # dphi (wrapped, v12b bugfix: bin by total_calo_E)
                raw_dphi = ((target_phi_dir - centroid_phi + torch.pi) % (2 * torch.pi)) - torch.pi
                dphi_median = self._interp(total_calo_ET, "neutral_dphi", "median")
                dphi_mad = self._interp(total_calo_ET, "neutral_dphi", "mad").clamp(min=self.eps)
                dphi_normalized = (raw_dphi - dphi_median) / dphi_mad
                dphi_loss_pq = self.barron_losses["neutral_dphi"](calo_corrections[:, :, 2] - dphi_normalized)
                # Per-param outlier masking (v12) or sum-based (v11b fallback)
                _calo_param_names = ["energy", "neutral_deta", "neutral_dphi"]
                _calo_raw = {"energy": energy_loss_pq, "neutral_deta": deta_loss_pq, "neutral_dphi": dphi_loss_pq}
                _calo_masks = {"energy": energy_mask, "neutral_deta": neutral_mask, "neutral_dphi": neutral_mask}

                # v12b: per-param loss values (grad-bearing) for monitor_* logging AND v12c Kendall weighting.
                calo_param_raw_losses: dict[str, torch.Tensor] = {}
                if self.use_outlier_masking and self.per_param_outlier_pct:
                    # v12: mask each calo component independently via shared helper.
                    calo_loss, calo_param_raw_losses = self._apply_per_param_outlier_mask(_calo_raw, _calo_masks, _calo_param_names)
                elif self.use_outlier_masking:
                    # v11b fallback: mask on sum
                    total_neutral_pq = energy_loss_pq + deta_loss_pq + dphi_loss_pq
                    valid_neutral_losses = total_neutral_pq[neutral_mask]
                    if valid_neutral_losses.numel() > 20:
                        thresh_n = torch.quantile(valid_neutral_losses.detach(), self.outlier_percentile)
                        keep_neutral = neutral_mask & (total_neutral_pq.detach() <= thresh_n)
                    else:
                        keep_neutral = neutral_mask
                    if keep_neutral.any():
                        calo_loss = total_neutral_pq[keep_neutral].mean()
                    else:
                        calo_loss = target_energy.new_tensor(0.0)
                else:
                    calo_loss = target_energy.new_tensor(0.0)
                    if energy_mask.any():
                        calo_loss = calo_loss + energy_loss_pq[energy_mask].mean()
                    if neutral_mask.any():
                        neutral_dir_loss = (deta_loss_pq + dphi_loss_pq)[neutral_mask].mean()
                        calo_loss = calo_loss + neutral_dir_loss

            else:
                pred_log_E = outputs[self.log_E_key].float()
                energy_mask = target_valid & has_calo

                target_energy = targets[f"{self.target_object}_{self.energy_target}"].float()
                target_log_E = torch.log(target_energy.clamp(min=self.eps))

                energy_loss_per_query = self.loss_fn(pred_log_E, target_log_E, reduction="none")
                energy_loss = energy_loss_per_query[energy_mask].mean() if energy_mask.any() else pred_log_E.new_tensor(0.0)
                calo_loss = energy_loss

            # --- Gate/charge sign loss ---
            gate_logits = outputs[self.gate_logits_key].float()  # (B, N, 2) — charge sign logits {+, -}

            # v5: 2-class {positive, negative} on charged particles only
            charge_sign = torch.sign(target_qpt)
            charge_target = (charge_sign < 0).long()  # 0 = positive, 1 = negative
            charge_mask = target_valid & is_charged if not self.use_track_correction else target_valid & is_charged & seed_valid

            if charge_mask.any():
                gate_ce = torch.nn.functional.cross_entropy(
                    gate_logits[charge_mask].reshape(-1, 2),
                    charge_target[charge_mask].reshape(-1),
                    reduction="mean",
                )
            else:
                gate_ce = gate_logits.new_tensor(0.0)

        if self.loss_combination == "gls":
            # v12e: paper-faithful product-form GLS partitioned by physics priority.
            #   L_total = (∏ L_priority)^{1/N_p} · (∏ L_bulk)^{1/N_b}
            #           = exp[ mean(log L_priority) + mean(log L_bulk) ]
            # Gate CE is grouped into bulk (knob-free, paper-faithful).
            # Priority/bulk gradient ratio is locked at (N_b/N_p)·L_b/L_p throughout training.
            local_vars = locals()
            track_raw = local_vars.get("track_param_raw_losses", {})
            calo_raw = local_vars.get("calo_param_raw_losses", {})
            # Build component dict; defensive access for gate_ce (only computed in PID-gate paths).
            # Skip gate_ce when it's the "no charged particles in batch" fallback (=0.0) — feeding 0 through
            # log() with eps clamp gives log(1e-8)=-18.4, which would drag GLS_b down catastrophically.
            components = {**track_raw, **calo_raw}
            gate_ce_local = local_vars.get("gate_ce", None)
            if gate_ce_local is not None and gate_ce_local.detach().item() > 0.0:
                components["gate_ce"] = gate_ce_local

            gls_eps = 1e-8

            priority_logs = [torch.log(components[p].clamp(min=gls_eps)) for p in self.gls_priority_params if p in components]
            bulk_logs = [torch.log(components[p].clamp(min=gls_eps)) for p in self.gls_bulk_params if p in components]

            # Reference tensor for type/device when constructing zero fallbacks
            ref_tensor = next(iter(components.values())) if components else track_loss

            if priority_logs and bulk_logs:
                log_gls_p = torch.stack(priority_logs).mean()
                log_gls_b = torch.stack(bulk_logs).mean()
                gls_value = torch.exp(log_gls_p + log_gls_b)
            elif priority_logs:
                log_gls_p = torch.stack(priority_logs).mean()
                log_gls_b = ref_tensor.new_tensor(0.0)
                gls_value = torch.exp(log_gls_p)
            elif bulk_logs:
                log_gls_p = ref_tensor.new_tensor(0.0)
                log_gls_b = torch.stack(bulk_logs).mean()
                gls_value = torch.exp(log_gls_b)
            else:
                log_gls_p = ref_tensor.new_tensor(0.0)
                log_gls_b = ref_tensor.new_tensor(0.0)
                gls_value = ref_tensor.new_tensor(0.0)

            losses = {"gls_total": self.loss_weight * gls_value}
            # Group-level monitors
            losses["monitor_log_gls_priority"] = log_gls_p.detach()
            losses["monitor_log_gls_bulk"] = log_gls_b.detach()
            losses["monitor_gls_priority_value"] = torch.exp(log_gls_p).detach()
            losses["monitor_gls_bulk_value"] = torch.exp(log_gls_b).detach()
            # Head-sum aggregate monitors (kept for cross-version comparison with v12d)
            losses["monitor_track_correction_raw"] = track_loss.detach()
            losses["monitor_calo_regression_raw"] = calo_loss.detach()
            if gate_ce_local is not None:
                losses["monitor_gate_ce_raw"] = gate_ce_local.detach()
            # Per-param monitors (grad-free) — preserves the existing monitor_* convention used by v12c/v12d
            for pn, pv in track_raw.items():
                losses[f"monitor_track_{pn}_loss"] = pv.detach()
            for cn, cv in calo_raw.items():
                losses[f"monitor_calo_{cn}_loss"] = cv.detach()
        else:
            local_vars = locals()
            track_raw = local_vars.get("track_param_raw_losses", {})
            calo_raw = local_vars.get("calo_param_raw_losses", {})

            if self.use_per_param_kendall:
                # v12c: Kendall & Gal weighting per sub-loss.
                #   L_weighted = Σ_i [ exp(-clamp(s_i)) · L_i + clamp(s_i) ]
                # s_i is a learnable log-σ² per param; exp(-s) is the weight, +s is the regularizer
                # (prevents all weights collapsing to zero). Clipped to keep dynamics bounded.
                lo, hi = self.kendall_log_var_clip
                # Assemble the per-param dict we'll weight. Use grad-bearing raw losses (no per-head-norm).
                kendall_components: dict[str, torch.Tensor] = {}
                for pn, pv in track_raw.items():
                    kendall_components[pn] = pv
                for cn, cv in calo_raw.items():
                    # The calo branch exposes "energy" (neutral-only when neutral_energy_only=true);
                    # the log_var param is named "energy" in self.kendall_log_var.
                    kendall_components[cn] = cv
                kendall_components["gate"] = gate_ce
                # Sum the weighted contributions; skip any sub-loss whose log_var buffer is absent.
                weighted = gate_ce.new_tensor(0.0)
                for name, value in kendall_components.items():
                    if name not in self.kendall_log_var:
                        continue
                    s = self.kendall_log_var[name].squeeze().clamp(lo, hi)
                    weighted = weighted + torch.exp(-s) * value + s
                losses = {
                    "kendall_total": self.loss_weight * weighted,
                }
                # Log each clamped log-σ² for diagnostics.
                for name in self.kendall_log_var:
                    s = self.kendall_log_var[name].squeeze().clamp(lo, hi).detach()
                    losses[f"monitor_log_var_{name}"] = s
                # Also log raw (pre-Kendall) group sums for trend comparison with older runs.
                losses["monitor_track_correction_raw"] = track_loss.detach()
                losses["monitor_calo_regression_raw"] = calo_loss.detach()
                losses["monitor_gate_ce_raw"] = gate_ce.detach()
            else:
                # Default: weighted sum (v12b behavior)
                losses = {
                    "track_correction": self.loss_weight * self.track_correction_weight * track_loss,
                    "calo_regression": self.loss_weight * self.energy_weight * calo_loss,
                    "gate_ce": self.loss_weight * self.gate_loss_weight * gate_ce,
                }
            # v12b: per-param loss components as monitor_* keys (logged, not optimized).
            for pn, pv in track_raw.items():
                losses[f"monitor_track_{pn}_loss"] = pv.detach()
            for cn, cv in calo_raw.items():
                losses[f"monitor_calo_{cn}_loss"] = cv.detach()

        return losses

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        gate_logits = outputs[self.gate_logits_key].detach()  # (B, N, 2) — charge sign logits {+, -}
        seed_qpt = outputs[f"{self.output_object}_seed_qpt"].detach()
        seed_valid = outputs[f"{self.output_object}_seed_valid"].detach()
        log_correction = outputs[self.log_correction_key].detach()

        # Denormalize correction if needed (legacy path; v11 does per-param denorm below)
        if self.target_mode == "log_mad" and not self.use_full_perigee:
            # log_mad: denormalize NN output back to log-space correction
            seed_pt = (1.0 / seed_qpt.abs().clamp(min=self.eps)).detach()
            log_mad = bin_interp(
                seed_pt,
                self.corr_log_mad,
                log_centers=self.corr_log_bin_centers,
                mode=self.interp_mode,
                pchip_c=getattr(self, "corr_log_mad_pchip_c", None),
                bin_edges=self.corr_bin_edges[1:-1],
                eps=self.eps,
            )
            log_median = bin_interp(
                seed_pt,
                self.corr_log_median,
                log_centers=self.corr_log_bin_centers,
                mode=self.interp_mode,
                pchip_c=getattr(self, "corr_log_median_pchip_c", None),
                bin_edges=self.corr_bin_edges[1:-1],
                eps=self.eps,
            )
            log_correction = log_correction * log_mad + log_median

        # v5: PID routing for charged/neutral, 2-class gate for charge sign
        pid_probs = outputs[f"{self.output_object}_pid_probs"].detach()  # (B, N, 6)
        pid_charged_prob = pid_probs[:, :, self.pid_charged_classes].sum(-1)  # (B, N)
        pid_is_charged = pid_charged_prob > 0.5  # PID-only routing (no seed_valid)
        is_charged_pred = pid_is_charged if not self.use_track_correction else pid_is_charged & seed_valid

        # 2-class charge sign: class 0 = positive, class 1 = negative
        gate_probs = torch.softmax(gate_logits, dim=-1)  # (B, N, 2)
        gate_pred_class = gate_logits.argmax(dim=-1)  # 0=positive, 1=negative
        raw_gate_sign = torch.where(gate_pred_class == 0, torch.ones_like(seed_qpt), -torch.ones_like(seed_qpt))

        # For routing: neutral particles get sign=0
        gate_sign = raw_gate_sign * is_charged_pred.float()

        # For pred_qpt: use raw sign so charged evaluation is not zeroed by PID routing
        pred_qpt = raw_gate_sign * seed_qpt.abs() * torch.exp(log_correction)

        preds = {
            f"{self.output_object}_pred_qpt": pred_qpt,
            f"{self.output_object}_seed_qpt": seed_qpt,
            f"{self.output_object}_log_correction": log_correction,
            f"{self.output_object}_log_E": outputs[self.log_E_key].detach(),
            f"{self.output_object}_gate_probs": gate_probs,
            f"{self.output_object}_gate_pred_class": gate_pred_class,
            f"{self.output_object}_gate_sign": gate_sign,
            f"{self.output_object}_is_charged_pred": is_charged_pred,
            f"{self.output_object}_pid_is_charged": pid_is_charged,
            f"{self.output_object}_seed_valid": seed_valid,
            f"{self.output_object}_loss_valid": outputs.get(f"{self.output_object}_loss_valid", seed_valid).detach(),
            f"{self.output_object}_has_calo": outputs[f"{self.output_object}_has_calo"].detach(),
            f"{self.output_object}_calo_eta": outputs[f"{self.output_object}_calo_direction"][..., 0].detach(),
            f"{self.output_object}_calo_phi": outputs[f"{self.output_object}_calo_direction"][..., 1].detach(),
        }

        # Pass calo_E_sum through to metrics for seed baseline comparison
        if self.use_calo_head:
            preds[f"{self.output_object}_calo_E_sum"] = outputs[f"{self.output_object}_calo_E_sum"].detach()

        # v11: per-param denormalization for all track + calo params
        if self.use_full_perigee:
            seed_pt = (1.0 / seed_qpt.abs().clamp(min=self.eps)).detach()
            track_corr = outputs[f"{self.output_object}_track_corrections"].detach()  # (B, N, 5)
            s_inv_pt = outputs[f"{self.output_object}_seed_inv_pt"].detach()
            s_d0 = outputs[f"{self.output_object}_seed_d0"].detach()
            s_z0 = outputs[f"{self.output_object}_seed_z0"].detach()
            s_phi0 = outputs[f"{self.output_object}_seed_phi0"].detach()
            s_eta = outputs[f"{self.output_object}_seed_eta"].detach()

            track_seeds = [s_inv_pt, s_d0, s_z0, s_phi0, s_eta]
            track_names = ["inv_pt", "d0", "z0", "phi0", "eta"]
            # v12b bugfix: phi0 uses add_wrap to match the wrapped training residual at task.py:3490.
            # Prior "add" mode leaked 2π errors at the ±π boundary even when the model's denorm
            # was correct (seed near -π, truth near +π → denorm ≈ small, but seed+denorm unwrapped).
            track_modes = ["log", "add", "add", "add_wrap", "add"]

            for i, (pn, mode, seed_v) in enumerate(zip(track_names, track_modes, track_seeds, strict=False)):
                med = self._interp(seed_pt, pn, "median")
                mad = self._interp(seed_pt, pn, "mad")
                denorm = track_corr[:, :, i] * mad + med
                if mode == "log":
                    pred_v = seed_v * torch.exp(denorm)
                elif mode == "add_wrap":
                    pred_v = ((seed_v + denorm + torch.pi) % (2 * torch.pi)) - torch.pi
                else:
                    pred_v = seed_v + denorm
                preds[f"{self.output_object}_pred_{pn}"] = pred_v
                preds[f"{self.output_object}_seed_{pn}"] = seed_v

            # Override pred_qpt with v11 inv_pt prediction + raw sign (not gate_sign, which zeros PID-misclassified charged)
            preds[f"{self.output_object}_pred_qpt"] = raw_gate_sign * preds[f"{self.output_object}_pred_inv_pt"]

        if self.use_calo_head:
            calo_corr = outputs[f"{self.output_object}_calo_corrections"].detach()  # (B, N, 3)
            calo_E_sum = outputs[f"{self.output_object}_calo_E_sum"].detach()
            c_eta = outputs[f"{self.output_object}_calo_centroid_eta"].detach()
            c_phi = outputs[f"{self.output_object}_calo_centroid_phi"].detach()
            # v12b bugfix: neutral params bin by total_calo_ET = total_calo_E / cosh(combined_eta).
            # Matches bin-stats gen (calo_ET transform). Fallback chain for legacy ckpts.
            total_calo_ET = (
                outputs.get(
                    f"{self.output_object}_total_calo_ET",
                    outputs.get(f"{self.output_object}_total_calo_E", seed_pt),
                )
                .detach()
                .float()
            )

            if self.neutral_energy_only:
                # v12c: three-way routing
                #   tier1  : seed_valid                         -> charged, track-derived E
                #   tier2a : ~seed_valid & pid_charged          -> charged, raw calo_E_sum (no learned correction)
                #   tier2b : ~seed_valid & ~pid_charged         -> neutral, neutral calo head
                # Previously (v11b/v12b) this block was hit-count-only and routed all
                # ~seed_valid queries (15% of predictions, ~90% of them PID=photon) through
                # the neutral head, which is trained on n_tracker<1 — OOD for 1-2 sihit cases.
                pid_probs = outputs[f"{self.output_object}_pid_probs"].detach()
                pid_is_charged = pid_probs[..., self.pid_charged_classes].sum(-1) > 0.5

                is_charged_pred = seed_valid | pid_is_charged
                preds[f"{self.output_object}_is_charged_pred"] = is_charged_pred
                preds[f"{self.output_object}_pid_is_charged"] = is_charged_pred
                preds[f"{self.output_object}_tracking_eligible"] = seed_valid

                # Tier 1: track-derived charged energy
                pred_inv_pt = preds[f"{self.output_object}_pred_inv_pt"]
                pred_eta_v = preds[f"{self.output_object}_pred_eta"]
                # v12b: cap pT at 365 GeV (CLD sample √s). Matches the clamp in forward E_track.
                pred_pT_track = 1.0 / pred_inv_pt.clamp(min=1.0 / 365.0)
                m_pi = 0.13957
                pred_energy_charged = torch.sqrt((pred_pT_track * torch.cosh(pred_eta_v)) ** 2 + m_pi**2)

                # Tier 2b: neutral head (now trained on ~seed_valid via broadened neutral_eligible)
                # v12b bugfix: bin by total_calo_E (neutrals have no valid seed_pt)
                med_ne = self._interp(total_calo_ET, "energy_neutral", "median")
                mad_ne = self._interp(total_calo_ET, "energy_neutral", "mad")
                denorm_E_neutral = calo_corr[:, :, 0] * mad_ne + med_ne
                pred_energy_neutral = calo_E_sum * torch.exp(denorm_E_neutral)

                # Combined: seed_valid->track; ~seed & pid_charged->raw calo; ~seed & pid_neutral->neutral head
                pred_energy = torch.where(
                    seed_valid,
                    pred_energy_charged,
                    torch.where(pid_is_charged, calo_E_sum, pred_energy_neutral),
                )
                preds[f"{self.output_object}_pred_energy"] = pred_energy
                preds[f"{self.output_object}_pred_energy_charged"] = pred_energy_charged
                preds[f"{self.output_object}_pred_energy_neutral"] = pred_energy_neutral
            else:
                # v11: PID-routed energy normalization (charged vs neutral bin stats)
                pid_is_ch = preds[f"{self.output_object}_pid_is_charged"]

                med_ch = self._interp(seed_pt, "energy_charged", "median")
                mad_ch = self._interp(seed_pt, "energy_charged", "mad")
                # v12b bugfix: neutral branches bin by total_calo_E
                med_ne = self._interp(total_calo_ET, "energy_neutral", "median")
                mad_ne = self._interp(total_calo_ET, "energy_neutral", "mad")
                e_med = torch.where(pid_is_ch, med_ch, med_ne)
                e_mad = torch.where(pid_is_ch, mad_ch, mad_ne)
                denorm_E = calo_corr[:, :, 0] * e_mad + e_med
                preds[f"{self.output_object}_pred_energy"] = calo_E_sum * torch.exp(denorm_E)

            # Neutral direction: denormalize (v12b bugfix — bin by total_calo_E)
            deta_med = self._interp(total_calo_ET, "neutral_deta", "median")
            deta_mad = self._interp(total_calo_ET, "neutral_deta", "mad")
            preds[f"{self.output_object}_pred_neutral_eta"] = c_eta + calo_corr[:, :, 1] * deta_mad + deta_med

            dphi_med = self._interp(total_calo_ET, "neutral_dphi", "median")
            dphi_mad = self._interp(total_calo_ET, "neutral_dphi", "mad")
            # v12b bugfix: wrap to (-π, π) — training residual (task.py:3631) is wrapped, so the
            # predicted correction is bounded. Unwrapped raw sum can straddle ±π for centroids
            # near the boundary.
            raw_nphi = c_phi + calo_corr[:, :, 2] * dphi_mad + dphi_med
            preds[f"{self.output_object}_pred_neutral_phi"] = ((raw_nphi + torch.pi) % (2 * torch.pi)) - torch.pi

        return preds

    def _apply_per_param_outlier_mask(
        self,
        raw_losses: dict[str, Tensor],
        valid_masks: dict[str, Tensor],
        param_names: list[str],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Per-param outlier masking then mean: shared helper for track & calo loss paths.

        For each param in ``param_names``, look up its per-query loss in ``raw_losses`` and
        its eligibility mask in ``valid_masks``. Clip at the param's percentile (falling back
        to ``self.outlier_percentile`` when the param is missing from ``self.per_param_outlier_pct``),
        take the mean over kept entries, and accumulate.

        Returns ``(total_loss, per_param_means)`` where ``per_param_means`` carries only
        params that actually contributed (for monitor_* logging and Kendall weighting).
        """
        # ref tensor for device/dtype-aware zeros
        ref = next(iter(raw_losses.values()))
        total = ref.new_tensor(0.0)
        per_param_means: dict[str, Tensor] = {}
        for pn in param_names:
            if pn not in raw_losses:
                continue
            pl = raw_losses[pn]
            cm = valid_masks[pn]
            if not cm.any():
                continue
            pct = self.per_param_outlier_pct.get(pn, self.outlier_percentile)
            valid_pl = pl[cm]
            if valid_pl.numel() > 20:
                thresh = torch.quantile(valid_pl.detach(), pct)
                keep = cm & (pl.detach() <= thresh)
            else:
                keep = cm
            if keep.any():
                param_mean = pl[keep].mean()
                total = total + param_mean
                per_param_means[pn] = param_mean
        return total, per_param_means

    def _pct_for(self, param_name: str) -> float:
        """Percentile used for outlier masking of this param — mirrors training loss config."""
        if self.per_param_outlier_pct is not None:
            return self.per_param_outlier_pct.get(param_name, self.outlier_percentile)
        return self.outlier_percentile

    def _masked_mean(self, residual: Tensor, mask: Tensor, pct: float) -> Tensor | None:
        """Mean of residual[mask] after discarding values above the pct'th percentile.

        Mirrors the outlier-masking the training loss applies, so the logged value reflects
        what the loss actually sees each step. Returns None if the mask is empty.
        """
        vals = residual[mask]
        if vals.numel() == 0:
            return None
        if vals.numel() <= 20:
            return vals.mean()
        thresh = torch.quantile(vals.detach(), pct)
        kept = vals[vals <= thresh]
        return kept.mean() if kept.numel() > 0 else vals.mean()

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        target_valid = targets[f"{self.target_object}_valid"]
        is_charged = targets[f"{self.target_object}_is_charged"].bool()
        is_neutral = targets[f"{self.target_object}_is_neutral"].bool()
        seed_valid = preds[f"{self.output_object}_seed_valid"]
        # v12b: metrics mirror the training scope (>= loss_min_hits) for apples-to-apples comparisons across versions
        loss_valid = preds.get(f"{self.output_object}_loss_valid", seed_valid)
        has_calo = preds[f"{self.output_object}_has_calo"]

        metrics: dict[str, Tensor] = {}

        # Track correction metrics (q/pT) — only when tracking enabled
        charged_seed_mask = target_valid & is_charged if not self.use_track_correction else target_valid & is_charged & loss_valid
        if self.use_track_correction and charged_seed_mask.any():
            pred_qpt = preds[f"{self.output_object}_pred_qpt"]
            target_qpt = targets[f"{self.target_object}_mom.qopt"]
            target_pt = targets[f"{self.target_object}_mom.r"]

            qpt_diff = pred_qpt - target_qpt
            metrics["track_qpt_mae"] = qpt_diff.abs()[charged_seed_mask].mean()

            pred_1pt = pred_qpt.abs()
            target_1pt = 1.0 / target_pt.clamp(min=self.eps)
            metrics["track_1pt_rel_mae"] = ((pred_1pt - target_1pt).abs() / target_1pt.clamp(min=self.eps))[charged_seed_mask].mean()

        # Energy metrics (log_E) — legacy head, skip when v11 calo head is active
        if not self.use_calo_head:
            pred_log_E = preds[f"{self.output_object}_log_E"]
            target_energy = targets[f"{self.target_object}_{self.energy_target}"]
            target_log_E = torch.log(target_energy.clamp(min=self.eps))

            energy_mask = target_valid & has_calo
            if energy_mask.any():
                metrics["energy_log_E_mae"] = (pred_log_E - target_log_E).abs()[energy_mask].mean()
                pred_E = torch.exp(pred_log_E)
                metrics["energy_E_rel_mae"] = ((pred_E - target_energy).abs() / target_energy.clamp(min=self.eps))[energy_mask].mean()

            charged_energy_mask = energy_mask & is_charged
            if charged_energy_mask.any():
                metrics["charged_energy_log_E_mae"] = (pred_log_E - target_log_E).abs()[charged_energy_mask].mean()

            neutral_energy_mask = energy_mask & is_neutral
            if neutral_energy_mask.any():
                metrics["neutral_energy_log_E_mae"] = (pred_log_E - target_log_E).abs()[neutral_energy_mask].mean()

        # Charge sign accuracy (charged particles with seed)
        if f"{self.output_object}_gate_pred_class" in preds and charged_seed_mask.any():
            gate_pred = preds[f"{self.output_object}_gate_pred_class"]
            truth_sign = torch.sign(targets[f"{self.target_object}_mom.qopt"])
            # v5+: 2-class head, reconstruct sign from class index
            raw_sign = torch.where(gate_pred == 0, torch.ones_like(truth_sign), -torch.ones_like(truth_sign))
            metrics["charge_sign_accuracy"] = (raw_sign == truth_sign)[charged_seed_mask].float().mean()

        # v11: per-param track metrics (5 helix params)
        if self.use_full_perigee and charged_seed_mask.any():
            truth_omega = targets[f"{self.target_object}_perigee.omega"].float()
            truth_d0 = targets[f"{self.target_object}_perigee.d0"].float()
            truth_z0 = targets[f"{self.target_object}_perigee.z0"].float()
            truth_phi0 = targets[f"{self.target_object}_perigee.phi0"].float()
            truth_tanlam = targets[f"{self.target_object}_perigee.tan_lambda"].float()
            truth_inv_pt = truth_omega.abs() / (0.3 * self.b_field)
            truth_eta = torch.asinh(truth_tanlam)

            track_truth = {"inv_pt": truth_inv_pt, "d0": truth_d0, "z0": truth_z0, "phi0": truth_phi0, "eta": truth_eta}
            for pn, truth_v in track_truth.items():
                pred_v = preds[f"{self.output_object}_pred_{pn}"]
                seed_v = preds[f"{self.output_object}_seed_{pn}"]
                # v12b bugfix: phi0 residual must be wrapped — two canonical-range angles can
                # straddle ±π and produce a 2π pseudo-error. Matches the neutral_phi metric at line 4130.
                if pn == "phi0":
                    pred_err = (((pred_v - truth_v) + torch.pi) % (2 * torch.pi) - torch.pi).abs()
                    seed_err = (((seed_v - truth_v) + torch.pi) % (2 * torch.pi) - torch.pi).abs()
                else:
                    pred_err = (pred_v - truth_v).abs()
                    seed_err = (seed_v - truth_v).abs()
                metrics[f"track_{pn}_pred_mae"] = pred_err[charged_seed_mask].mean()
                metrics[f"track_{pn}_seed_mae"] = seed_err[charged_seed_mask].mean()
                # v12b: masked mean mirrors training-loss outlier mask — filters the rare extreme-seed particle
                pct = self._pct_for(pn)
                m_pred = self._masked_mean(pred_err, charged_seed_mask, pct)
                m_seed = self._masked_mean(seed_err, charged_seed_mask, pct)
                if m_pred is not None:
                    metrics[f"track_{pn}_pred_mae_masked"] = m_pred
                    metrics[f"track_{pn}_seed_mae_masked"] = m_seed

        # v11/v11b: energy metrics (split charged/neutral + calo seed baseline)
        if self.use_calo_head and f"{self.output_object}_pred_energy" in preds:
            pred_energy = preds[f"{self.output_object}_pred_energy"]
            target_energy_v11 = targets[f"{self.target_object}_{self.energy_target}"]
            calo_E_sum_val = preds[f"{self.output_object}_calo_E_sum"]

            e_rel = (pred_energy - target_energy_v11).abs() / target_energy_v11.clamp(min=self.eps)
            e_rel_seed = (calo_E_sum_val - target_energy_v11).abs() / target_energy_v11.clamp(min=self.eps)

            energy_mask_v11 = target_valid & has_calo
            pct_e = self._pct_for("energy")
            if energy_mask_v11.any():
                metrics["calo_energy_rel_mae"] = e_rel[energy_mask_v11].mean()
                m = self._masked_mean(e_rel, energy_mask_v11, pct_e)
                if m is not None:
                    metrics["calo_energy_rel_mae_masked"] = m

            # Charged energy metrics
            charged_calo_mask = energy_mask_v11 & is_charged
            if charged_calo_mask.any():
                metrics["charged_energy_rel_mae"] = e_rel[charged_calo_mask].mean()
                metrics["charged_energy_seed_rel_mae"] = e_rel_seed[charged_calo_mask].mean()
                m_ch = self._masked_mean(e_rel, charged_calo_mask, pct_e)
                m_ch_s = self._masked_mean(e_rel_seed, charged_calo_mask, pct_e)
                if m_ch is not None:
                    metrics["charged_energy_rel_mae_masked"] = m_ch
                    metrics["charged_energy_seed_rel_mae_masked"] = m_ch_s

            # Neutral energy metrics
            neutral_calo_mask = energy_mask_v11 & is_neutral
            if neutral_calo_mask.any():
                metrics["neutral_energy_rel_mae"] = e_rel[neutral_calo_mask].mean()
                metrics["neutral_energy_seed_rel_mae"] = e_rel_seed[neutral_calo_mask].mean()
                m_n = self._masked_mean(e_rel, neutral_calo_mask, pct_e)
                m_n_s = self._masked_mean(e_rel_seed, neutral_calo_mask, pct_e)
                if m_n is not None:
                    metrics["neutral_energy_rel_mae_masked"] = m_n
                    metrics["neutral_energy_seed_rel_mae_masked"] = m_n_s

                if f"{self.output_object}_pred_neutral_eta" in preds:
                    pred_neta = preds[f"{self.output_object}_pred_neutral_eta"]
                    pred_nphi = preds[f"{self.output_object}_pred_neutral_phi"]
                    target_neta = targets[f"{self.target_object}_mom.eta"]
                    target_nphi = targets[f"{self.target_object}_mom.phi"]
                    metrics["neutral_eta_mae"] = (pred_neta - target_neta).abs()[neutral_calo_mask].mean()
                    dphi = (pred_nphi - target_nphi + torch.pi) % (2 * torch.pi) - torch.pi
                    metrics["neutral_phi_mae"] = dphi.abs()[neutral_calo_mask].mean()

        # v11b: track-derived charged energy (E_track vs truth)
        if self.use_aux_energy_loss and f"{self.output_object}_pred_energy_charged" in preds and charged_seed_mask.any():
            pred_e_charged = preds[f"{self.output_object}_pred_energy_charged"]
            target_energy_ch = targets[f"{self.target_object}_{self.energy_target}"]
            e_rel_ch = (pred_e_charged - target_energy_ch).abs() / target_energy_ch.clamp(min=self.eps)
            metrics["track_energy_rel_mae"] = e_rel_ch[charged_seed_mask].mean()
            # v12b: masked mirrors training outlier mask on "energy_track" — reflects actual loss scope
            m_te = self._masked_mean(e_rel_ch, charged_seed_mask, self._pct_for("energy_track"))
            if m_te is not None:
                metrics["track_energy_rel_mae_masked"] = m_te

        # v8c: log Barron loss learned params
        if self.use_barron_loss:
            if self.use_full_perigee:
                for param_name, barron in self.barron_losses.items():
                    metrics[f"barron_{param_name}_alpha"] = barron.get_alpha()
                    metrics[f"barron_{param_name}_scale"] = barron.get_scale()
            else:
                metrics["barron_track_alpha"] = self.track_barron.get_alpha()
                metrics["barron_track_scale"] = self.track_barron.get_scale()

        return metrics
