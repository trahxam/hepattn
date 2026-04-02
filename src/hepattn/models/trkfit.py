"""Boosted track parameter fitter with iterative cross-attention refinement.

Architecture inspired by gradient boosting and Deformable DETR's iterative
box refinement (Zhu et al., 2021, https://arxiv.org/abs/2010.04159):
each stage predicts a residual correction to the current parameter estimate,
with the track embedding attending to encoded hits via cross-attention.

Key design decisions:
  - Cross-attention (track query → hits) over simple pooling: the attention
    weights adapt to the current parameter estimate, giving the fitter
    explicit read-access into the hit set conditioned on where it currently
    thinks the track is.
  - Shared hit encoder across stages: forces hit representations to be
    reusable and keeps parameter count manageable.
  - No detach between stages by default: end-to-end gradients let the
    encoder learn representations that are useful for all refinement stages;
    auxiliary losses at each stage prevent early-stage collapse.
  - EMA of per-field, per-stage residual std used to normalise the smooth-L1
    loss, so that fields with very different natural scales (pt in GeV vs
    d0_m in metres) contribute comparably throughout training.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.norm import NORM_TYPES
from hepattn.utils.helix import _fit_helices_flat

# Hit fields that are computed from the helix perigee (not loaded from data).
# When use_relative_coords=True these are injected into track_inputs after the
# helix fit so that the input_net can use them, but they are NOT gathered from
# the raw input dict.
_RELATIVE_HIT_FIELDS: frozenset[str] = frozenset({
    "x_rel", "y_rel", "z_rel", "r_rel", "phi_rel", "eta_rel",
    "dz_helix", "dxy_helix", "sagitta",
    "deta", "dtheta", "dphi",
    "x_prime", "y_prime", "dz_signed",
})


def _barron_loss(x: Tensor, alpha: Tensor, scale: Tensor) -> Tensor:
    """General adaptive robust loss — Barron (CVPR 2019).

    Unifies L2, Charbonnier, Cauchy, Welsch/Geman-McClure and all
    intermediates under a single formula parameterised by shape α ∈ ℝ:

        ρ(x, α, c) = (|α-2|/α) · ((x/c)² / |α-2| + 1)^(α/2) − 1)

    Special cases (all recovered as continuous limits):
        α = 2  →  (x/c)² / 2                        (L2 / MSE)
        α = 1  →  2·(√((x/c)²/2 + 1) − 1)           (Charbonnier / pseudo-Huber)
        α = 0  →  log((x/c)²/2 + 1)                  (Cauchy / Lorentzian)
        α → −∞ →  1 − exp(−(x/c)²/2)                (Welsch / Geman-McClure)

    Implemented in log-space via expm1 for numerical stability.  The
    L'Hôpital limits α→0 (Cauchy) and α→2 (L2) are applied with a small
    threshold to avoid 0/0 and ∞·0 pathologies.

    Args:
        x:     Residuals, shape (n_tracks, n_fields).
        alpha: Per-field shape parameter, shape (n_fields,).
               Constraining α ≤ 2 keeps the loss convex near the origin.
        scale: Per-field scale (e.g. EMA MAD), shape (n_fields,).

    Returns:
        Elementwise loss, shape (n_tracks, n_fields).
    """
    sq = (x / scale.clamp(min=1e-8)).pow(2)          # (n, f) normalised sq residual

    eps = 1e-5
    # |α − 2| clamped away from 0 to avoid division-by-zero in log1p(sq/|α-2|)
    abs_a2 = (alpha - 2.0).abs().clamp(min=eps)       # (f,)
    # α clamped away from 0 (preserving sign) for the |α-2|/α prefactor
    alpha_nz = torch.where(alpha >= 0, alpha.clamp(min=eps), alpha.clamp(max=-eps))  # (f,)

    # General case in log/exp space:
    #   (|α-2|/α) · expm1(α/2 · log1p(sq/|α-2|))
    log_inner = torch.log1p(sq / abs_a2)              # (n, f)
    loss_gen  = (abs_a2 / alpha_nz) * torch.expm1(0.5 * alpha * log_inner)  # (n, f)

    # L'Hôpital limits override near the degenerate points
    loss_cauchy = torch.log1p(0.5 * sq)               # α → 0
    loss_l2     = 0.5 * sq                            # α → 2

    near_zero = alpha.abs()         < eps * 100        # (f,)
    near_two  = (alpha - 2.0).abs() < eps * 100        # (f,)

    return torch.where(near_two, loss_l2, torch.where(near_zero, loss_cauchy, loss_gen))


def _gather_track_hits(
    inputs: dict[str, Tensor],
    fields: list[str],
    input_name: str,
) -> tuple[dict[str, Tensor], Tensor]:
    """Convert CSR track-hit assignment to padded per-track feature tensors.

    Returns:
        track_inputs: dict mapping ``{input_name}_{field}`` → (N_tracks, max_hits)
        track_hit_valid: bool mask (N_tracks, max_hits), True for real hits
    """
    device = next(iter(inputs.values())).device

    indptr  = inputs[f"track_{input_name}_indptr"][0]
    indices = inputs[f"track_{input_name}_indices"][0]

    N_tracks = len(indptr) - 1
    hits_per_track = (indptr[1:] - indptr[:-1]).long()
    max_hits = int(hits_per_track.max().item()) if N_tracks > 0 else 1

    row_range = torch.arange(max_hits, device=device)
    track_hit_valid = row_range.unsqueeze(0) < hits_per_track.unsqueeze(1)

    track_hit_idx = torch.zeros(N_tracks, max_hits, dtype=torch.long, device=device)
    if indices.numel() > 0:
        hit_to_track = torch.repeat_interleave(
            torch.arange(N_tracks, device=device), hits_per_track
        )
        hit_to_local = torch.arange(indices.numel(), device=device) - indptr[hit_to_track]
        track_hit_idx[hit_to_track, hit_to_local] = indices

    track_inputs: dict[str, Tensor] = {}
    for field in fields:
        feat = inputs[f"{input_name}_{field}"][0]
        track_inputs[f"{input_name}_{field}"] = feat[track_hit_idx]

    return track_inputs, track_hit_valid


class BoostStage(nn.Module):
    """One cross-attention refinement stage for boosted track fitting.

    The current track embedding (query) attends to encoded hit embeddings
    (keys/values), producing a refined track embedding and a raw parameter
    correction ``delta``.  Structure mirrors a standard transformer decoder
    cross-attention block: pre-norm → cross-attention → residual, then
    pre-norm → FFN → residual, then a linear head predicting the correction.

    The head is initialised with near-zero weights so that early in training
    all stages start close to the identity (no correction), allowing the
    helix estimate to dominate until the network learns useful corrections.
    """

    def __init__(
        self,
        dim: int,
        n_fields: int,
        n_heads: int = 4,
        norm: str = "FastLayerNorm",
        gate_min: float = 0.0,
        use_gates: bool = True,
        n_layers: int = 1,
        delta_head_init_std: float = 1e-3,
    ):
        super().__init__()
        norm_cls = NORM_TYPES[norm]
        self.n_layers = n_layers

        # Stack of cross-attention + FFN blocks.  n_layers=1 recovers the
        # original single-block behaviour.  More layers let the track embedding
        # make multiple passes over the encoded hits before the delta head fires,
        # which is useful for fields (e.g. qopt) where a single attention pass
        # is insufficient to extract the relevant correction signal.
        self.norm_qs    = nn.ModuleList([norm_cls(dim) for _ in range(n_layers)])
        self.cross_attns = nn.ModuleList([
            Attention(dim, num_heads=n_heads, attn_type="torch", bias=False)
            for _ in range(n_layers)
        ])
        self.norm_ffs = nn.ModuleList([norm_cls(dim) for _ in range(n_layers)])
        self.ffs      = nn.ModuleList([Dense(dim, activation="SwiGLU") for _ in range(n_layers)])

        mid = dim // 2

        # Single MLP predicting corrections for all fields: dim → dim//2 → n_fields.
        # delta_head_init_std controls the scale of initial corrections:
        # larger values give stronger gradient signal to the encoder early in training.
        self.delta_head = nn.Sequential(nn.Linear(dim, mid), nn.SiLU(), nn.Linear(mid, n_fields))
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.normal_(self.delta_head[-1].weight, std=delta_head_init_std)

        # Single gate MLP: dim → dim//2 → n_fields, then sigmoid.
        # Bias=0 → gate ≈ 0.5 at init.
        self.gate_head = nn.Sequential(nn.Linear(dim, mid), nn.SiLU(), nn.Linear(mid, n_fields))
        nn.init.constant_(self.gate_head[-1].bias, 0.0)
        nn.init.normal_(self.gate_head[-1].weight, std=1e-3)
        self.gate_min  = gate_min
        self.use_gates = use_gates

        # Single log-sigma MLP for Gaussian NLL loss: dim → dim//2 → n_fields.
        self.log_sigma_head = nn.Sequential(nn.Linear(dim, mid), nn.SiLU(), nn.Linear(mid, n_fields))
        nn.init.zeros_(self.log_sigma_head[-1].bias)
        nn.init.normal_(self.log_sigma_head[-1].weight, std=1e-3)

    def forward(
        self,
        track_embed:  Tensor,  # (n_tracks, 1, dim)
        encoded_hits: Tensor,  # (n_tracks, n_hits, dim)
        hit_mask:     Tensor,  # (n_tracks, n_hits) bool — True for real hits
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            track_embed: Refined embedding, shape (n_tracks, 1, dim).
            delta:       Gated parameter corrections, shape (n_tracks, n_fields).
            gates:       Sigmoid gate values, shape (n_tracks, n_fields).
            log_sigma:   Log-sigma per track per field, shape (n_tracks, n_fields).
        """
        # Iterative cross-attention + FFN refinement.
        # Each layer attends to the same shared encoded_hits; the track
        # embedding accumulates progressively richer context before the
        # delta/gate heads fire.
        for norm_q, cross_attn, norm_ff, ff in zip(
            self.norm_qs, self.cross_attns, self.norm_ffs, self.ffs
        ):
            track_embed = track_embed + cross_attn(
                norm_q(track_embed), encoded_hits, encoded_hits, kv_mask=hit_mask
            )
            track_embed = track_embed + ff(norm_ff(track_embed))
        embed = track_embed.squeeze(1)  # (n_tracks, dim)
        # Gates always computed (for monitoring) but only applied when use_gates=True.
        gates = torch.sigmoid(self.gate_head(embed))  # (n_tracks, n_fields)
        if self.use_gates:
            if self.gate_min > 0.0:
                gates = gates.clamp(min=self.gate_min)
            delta = gates * self.delta_head(embed)
        else:
            delta = self.delta_head(embed)
        log_sigma = self.log_sigma_head(embed)  # (n_tracks, n_fields)
        return track_embed, delta, gates, log_sigma


class BoostedTrackFitter(nn.Module):
    """Iterative track parameter fitter using boosted cross-attention refinement.

    Architecture overview:
      1. Classical helix fit → initial parameter estimate (pt, phi, eta, d0, z0).
      2. Project helix parameters → initial track embedding via a small MLP.
      3. Embed hits (InputNet) and encode them with a shared Transformer encoder.
         The encoder is shared across all boosting stages so that hit
         representations capture information useful for all levels of refinement.
      4. For each boosting stage k:
           a. Track embedding cross-attends to encoded hits (BoostStage).
           b. A linear head predicts a raw parameter correction delta_k.
           c. Update estimate: params_k = params_{k-1} + delta_k.
           d. (Optional) Re-project the updated params → residual added to the
              track embedding, giving the next stage an explicit signal about
              the current position in parameter space.
      5. Auxiliary smooth-L1 loss at every stage; earlier stages use a lower
         weight (``stage_loss_weight``) so that the final stage dominates.
      6. Final prediction = accumulated estimate after the last stage.

    Loss normalisation:
      A running EMA of the per-field, per-stage residual standard deviation is
      maintained during training and used to normalise the smooth-L1 loss.
      This balances learning across fields with very different natural scales
      (e.g. pt in GeV vs d0_m in metres) without requiring hand-tuned weights.
      The EMA is updated *after* the loss is computed each batch to avoid
      using current-batch statistics to scale current-batch gradients.

    Args:
        input_net: Hit feature embedding network (InputNet). Mutually exclusive with input_nets.
        input_nets: ModuleList of InputNets. One net = shared embedding (same as input_net).
            Two nets = separate vtxd (index 0) and trkr (index 1) embeddings, blended
            by the sihit_is_vtxd mask. Mutually exclusive with input_net.
        encoder: Shared transformer encoder over per-track hit sequences.
        tasks: ModuleList of task modules (used for metrics only; loss is
            computed directly in this model).
        dim: Embedding dimension.
        fields: Ordered list of track parameter field names to regress.
            Must match the keys in the targets dict (prefixed with
            ``track_matched_particle_``).
        angular_fields: Subset of ``fields`` that are angles in radians.
            Residuals for these are wrapped via atan2 before loss/metrics.
        n_stages: Number of boosting (refinement) stages.
        B_field: Magnetic field in Tesla, used to convert helix radius to pt.
        n_cross_heads: Number of attention heads in each BoostStage.
        norm: Normalisation type for BoostStage layers.
        inject_params: If True, re-project the updated parameter estimate
            into the track embedding before each stage (except the last).
            Gives the next stage explicit knowledge of the current estimate.
        detach_stages: If True, stop gradients between consecutive stage
            predictions (closer to true gradient boosting). Default False
            allows end-to-end training.
        ema_momentum: Momentum for the EMA correction-std buffers.
        normalize_loss: If True, normalise residuals by the EMA std before
            computing smooth-L1, balancing loss scale across fields.
        stage_loss_weight: Loss weight for all stages except the last
            (which always has weight 1.0).
    """

    def __init__(
        self,
        encoder: nn.Module,
        tasks: nn.ModuleList,
        dim: int,
        fields: list[str],
        input_net: nn.Module | None = None,
        input_nets: nn.ModuleList | None = None,
        angular_fields: list[str] | None = None,
        n_stages: int = 3,
        B_field: float = 3.0,
        n_cross_heads: int = 4,
        n_stage_layers: int = 1,
        norm: str = "FastLayerNorm",
        inject_params: bool = True,
        detach_stages: bool = False,
        ema_momentum: float = 0.9,
        normalize_loss: bool = True,
        stage_loss_weight: float = 0.5,
        ema_init: dict[str, float] | None = None,
        trim_fraction: float = 0.0,
        loss_type: str = "cauchy",
        welsch_c: float = 1.0,
        arcsinh_fixed_scale: dict[str, float] | None = None,
        use_relative_coords: bool = False,
        field_loss_weights: dict[str, float] | None = None,
        param_proj_scales: dict[str, float] | None = None,
        gate_min: float = 0.0,
        use_gates: bool = False,
        delta_head_init_std: float = 1e-3,
        barron_learn_scale: bool = False,
        use_sagitta: bool = False,
        field_combination: str = "mean",
        debug: bool = False,
    ):
        super().__init__()

        self.debug           = debug
        self.use_sagitta        = use_sagitta
        self.field_combination  = field_combination

        # Accept either a single input_net (backward compat) or a ModuleList.
        # When two nets are provided: nets[0] embeds vtxd hits (sihit_is_vtxd=1),
        # nets[1] embeds trkr hits (sihit_is_vtxd=0); outputs are blended.
        assert (input_net is None) != (input_nets is None), \
            "Provide exactly one of input_net or input_nets"
        if input_net is not None:
            self.input_nets = nn.ModuleList([input_net])
        else:
            self.input_nets = input_nets

        self.encoder         = encoder
        self.tasks           = tasks
        self.dim             = dim
        self.fields          = fields
        self.angular_fields  = angular_fields or []
        self.n_stages        = n_stages
        self.B_field         = B_field
        self.inject_params   = inject_params
        self.detach_stages   = detach_stages
        self.ema_momentum    = ema_momentum
        self.normalize_loss  = normalize_loss
        self.stage_loss_weight = stage_loss_weight
        self.trim_fraction   = trim_fraction
        self.loss_type       = loss_type
        self.welsch_c        = welsch_c
        self.use_fixed_mad_scale   = arcsinh_fixed_scale is not None
        self.use_relative_coords   = use_relative_coords

        n_fields = len(fields)

        # Pairwise geometric attention bias for the hit self-attention encoder.
        # 5 features: dr, dz, dphi, seg_angle, ds (arc-length difference).
        # The ds column is zero-initialised so it starts as a no-op and only
        # learns to use arc-length information once the encoder is stable.
        n_encoder_heads = encoder.layers[0].attn.fn.num_heads
        self.pair_net = nn.Linear(5, n_encoder_heads, bias=False)
        nn.init.zeros_(self.pair_net.weight[:, 4:])

        # Project helix parameter vector → initial track embedding.
        # MLP rather than a single linear layer so it can learn to handle
        # the very different natural scales of the input fields.
        self.param_proj = Dense(n_fields, dim, hidden_layers=[dim])

        # One BoostStage per refinement iteration.
        self.stages = nn.ModuleList([
            BoostStage(dim, n_fields, n_cross_heads, norm, gate_min=gate_min, use_gates=use_gates, n_layers=n_stage_layers, delta_head_init_std=delta_head_init_std)
            for _ in range(n_stages)
        ])

        # Optional per-stage parameter re-injection.
        if inject_params:
            self.param_inject = Dense(n_fields, dim, hidden_layers=[dim])

        # Per-field angular mask (non-persistent; recreated if device changes).
        self.register_buffer(
            "angular_mask",
            torch.tensor([f in self.angular_fields for f in fields], dtype=torch.bool),
            persistent=False,
        )

        # Running EMA of the per-field residual std at each stage.
        # Shape: (n_stages, n_fields).  Initialised from ema_init (if given)
        # so that the loss is well-scaled from step 0, then updated every
        # training batch so that the normalisation adapts to actual corrections.
        if ema_init is not None:
            field_scales = torch.tensor([ema_init.get(f, 1.0) for f in fields])
            init_scale   = field_scales.unsqueeze(0).expand(n_stages, -1).clone()
        else:
            init_scale = torch.ones(n_stages, n_fields)
        self.register_buffer("ema_correction_std", init_scale)
        # EMA of raw MAD (without the 1.4826 Gaussian-equivalence factor).
        # Used by the "arcsinh_mad" loss type.
        self.register_buffer("ema_mad", init_scale / 1.4826)

        # Per-field unit scales: delta_heads predict corrections in normalised
        # units; multiplying by field_unit_scales converts back to model-native
        # units.  This makes gate_bias=0 safe for all fields simultaneously —
        # with std=1e-3 init, initial corrections are ~0.4% of ACTS MAD for
        # every field regardless of its natural scale (metres vs GeV⁻¹ etc.).
        if ema_init is not None:
            unit_scales = torch.tensor([ema_init.get(f, 1.0) for f in fields])
        else:
            unit_scales = torch.ones(n_fields)
        self.register_buffer("field_unit_scales", unit_scales, persistent=False)

        # Per-field scales for normalising helix parameters before param_proj /
        # param_inject.  These should be set to the *typical parameter magnitude*
        # (so the MLP receives O(1) inputs), which is very different from the
        # residual-MAD scale used for delta scaling.  Defaults to field_unit_scales
        # for backward compatibility, but should always be set explicitly.
        if param_proj_scales is not None:
            proj_scales = torch.tensor([param_proj_scales.get(f, 1.0) for f in fields])
        else:
            proj_scales = unit_scales.clone()
        self.register_buffer("param_proj_scales", proj_scales, persistent=False)

        # Optional fixed per-field scale for the arcsinh_mad loss.  When set,
        # replaces the EMA MAD with a constant scale (EMA is still updated for
        # monitoring but not used for the loss).  Shape: (n_stages, n_fields).
        if arcsinh_fixed_scale is not None:
            fixed = torch.tensor([arcsinh_fixed_scale.get(f, 1e-3) for f in fields])
            self.register_buffer("fixed_mad_scale", fixed.unsqueeze(0).expand(n_stages, -1).clone())
        else:
            self.fixed_mad_scale = None

        # Per-field loss weights (n_fields,).  Allows up-weighting fields that
        # are harder to improve (e.g. z0, eta, qopt) relative to fields that
        # already converge easily (e.g. d0, phi).  Defaults to uniform.
        fw = torch.tensor([field_loss_weights.get(f, 1.0) if field_loss_weights else 1.0 for f in fields])
        self.register_buffer("field_loss_weights", fw, persistent=False)

        # Barron adaptive loss: learnable shape α per field per stage.
        # Parameterised as α = 2 − softplus(alpha_raw) so that α ≤ 2 always.
        # alpha_raw = 0  →  α = 2 − ln(2) ≈ 1.31  (between L2 and Cauchy)
        # alpha_raw ≫ 0  →  α → −∞                  (Welsch — most robust)
        # alpha_raw ≪ 0  →  α → 2                    (L2 — least robust)
        # Only trained when loss_type == "barron"; otherwise acts as dead weight.
        self.alpha_raw = nn.Parameter(torch.zeros(n_stages, n_fields))

        # Learnable scale offset for the Barron loss (only used when
        # barron_learn_scale=True).  c = ema_mad × exp(log_c_raw), so
        # log_c_raw=0 recovers the EMA-MAD baseline and the parameter learns a
        # multiplicative correction per field per stage.  Positive log_c_raw
        # widens the quadratic bowl (fewer tracks treated as outliers);
        # negative log_c_raw tightens it (more aggressive core focus).
        self.barron_learn_scale = barron_learn_scale
        self.log_c_raw = nn.Parameter(torch.zeros(n_stages, n_fields))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pair_bias(self, track_inputs: dict[str, Tensor]) -> Tensor:
        """Compute pairwise geometric bias for the hit encoder attention."""
        r   = track_inputs["sihit_r"]
        z   = track_inputs["sihit_z"]
        phi = track_inputs["sihit_phi"]

        dr        = r.unsqueeze(2) - r.unsqueeze(1)
        dz        = z.unsqueeze(2) - z.unsqueeze(1)
        dphi      = torch.atan2(
            torch.sin(phi.unsqueeze(2) - phi.unsqueeze(1)),
            torch.cos(phi.unsqueeze(2) - phi.unsqueeze(1)),
        )
        seg_angle = torch.atan2(dz, dr)
        s         = track_inputs["sihit_s"]
        ds        = s.unsqueeze(2) - s.unsqueeze(1)

        pair_feats = torch.stack([dr, dz, dphi, seg_angle, ds], dim=-1)
        return self.pair_net(pair_feats)

    def _helix_init(
        self,
        track_inputs: dict[str, Tensor],
        track_hit_valid: Tensor,
        sihit_det: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the classical helix fit and return initial params in field order.

        Returns:
            helix_params: (n_tracks, n_fields) — zeros for fields not in helix_map.
            helix_ok:     (n_tracks,) bool — True if the fit converged.
        """
        N_tracks = track_hit_valid.shape[0]
        device   = track_hit_valid.device
        dtype    = track_inputs["sihit_x"].dtype
        n_fields = len(self.fields)

        if N_tracks == 0:
            return (
                torch.zeros(0, n_fields, device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=torch.bool),
            )

        # Sort hits by transverse radius so that _fit_helices_flat sees a
        # monotonically increasing arc (required for phi-unwrapping).
        r      = track_inputs["sihit_r"]
        r_sort = r.masked_fill(~track_hit_valid, float("inf"))
        order  = r_sort.argsort(dim=-1)

        x_s = track_inputs["sihit_x"].gather(-1, order)
        y_s = track_inputs["sihit_y"].gather(-1, order)
        z_s = track_inputs["sihit_z"].gather(-1, order)
        w_s = track_hit_valid.float().gather(-1, order)

        zfit_w: Tensor | None = None
        if sihit_det is not None:
            det_s = sihit_det.gather(-1, order)
            # Exclude long-strip hits (detector code > 5) from the circle fit.
            w_s = w_s * (det_s <= 5).float()
            # Pixel-only z(r) weights (detector code <= 2).
            # Fall back to all non-long-strip hits when fewer than 3 pixels per track.
            pixel_w = w_s * (det_s <= 2).float()
            n_pixel  = pixel_w.sum(dim=-1, keepdim=True)
            zfit_w   = torch.where(n_pixel >= 3, pixel_w, w_s)
        elif "sihit_is_vtxd" in track_inputs:
            # Trkr strip hits have cm-scale z resolution that produces
            # catastrophic z0 outliers.  Restrict z(r) fit to vtxd-only hits;
            # tracks with < 3 vtxd hits will be marked as failed fits (z0 = 0).
            vtxd_s = track_inputs["sihit_is_vtxd"].gather(-1, order)
            zfit_w = vtxd_s * w_s

        R_h, phi0_h, eta_h, d0_h, z0_h, helix_ok, charge_sign = _fit_helices_flat(x_s, y_s, z_s, w_s, zfit_w=zfit_w)
        pt_h   = 0.3 * self.B_field * R_h            # pt [GeV] = 0.3 * B[T] * R[m]
        qopt_h = charge_sign / pt_h.clamp(min=1e-6)  # q/pT [GeV⁻¹]; linear in curvature
        # For failed fits _fit_helices_flat already zeros R/eta/d0/z0; do the
        # same for qopt — without this, R=0 → qopt=1/clamp=1e6, which would
        # overflow the embedding network and produce NaN in the attention.
        qopt_h = torch.where(helix_ok, qopt_h, torch.zeros_like(qopt_h))

        inv_pt_h = 1.0 / pt_h.clamp(min=1e-6)
        inv_pt_h = torch.where(helix_ok, inv_pt_h, torch.zeros_like(inv_pt_h))

        theta_h = 2.0 * torch.atan(torch.exp(-eta_h))

        helix_map: dict[str, Tensor] = {
            "eta":          eta_h,
            "theta":        theta_h,
            "phi_perigee":  phi0_h,
            "pt":           pt_h,
            "qopt":         qopt_h,
            "inv_pt":       inv_pt_h,
            "d0_perigee_m": d0_h,
            "z0_perigee_m": z0_h,
        }
        helix_params = torch.stack(
            [helix_map.get(f, torch.zeros_like(pt_h)) for f in self.fields],
            dim=-1,
        )  # (n_tracks, n_fields)
        return helix_params, helix_ok

    def _compute_sagitta(
        self,
        track_inputs: dict[str, Tensor],
        track_hit_valid: Tensor,  # (n_tracks, max_hits) bool
    ) -> Tensor:
        """Compute signed three-point sagitta per hit in the transverse plane.

        For each hit B with its radius-sorted neighbours A (inner) and C (outer):

            sagitta = ((B - A) × (C - A)) / |C - A|

        where × is the 2D cross product in the transverse (x-y) plane.
        The sign encodes charge: negative for positive charge (q>0) in +Bz,
        so sagitta ≈ -qopt × 0.3 × B × |AC|² / 8.

        Boundary hits (innermost and outermost after r-sorting) and padding hits
        are assigned sagitta = 0.

        Returns:
            sagitta: (n_tracks, max_hits) in the original (unsorted) hit order.
        """
        x = track_inputs["sihit_x"]  # (n_tracks, max_hits)
        y = track_inputs["sihit_y"]
        r = track_inputs["sihit_r"]

        # Sort hits by transverse radius so neighbours are geometrically adjacent.
        r_for_sort = r.masked_fill(~track_hit_valid, float("inf"))
        order     = r_for_sort.argsort(dim=-1)   # (n_tracks, max_hits)
        inv_order = order.argsort(dim=-1)          # inverse permutation

        x_s     = x.gather(-1, order)             # (n_tracks, max_hits), sorted by r
        y_s     = y.gather(-1, order)
        valid_s = track_hit_valid.gather(-1, order).float()

        # Pad each side with one zero-hit (invalid) so boundary indexing is uniform.
        x_pad = F.pad(x_s, (1, 1), value=0.0)    # (n_tracks, max_hits+2)
        y_pad = F.pad(y_s, (1, 1), value=0.0)
        v_pad = F.pad(valid_s, (1, 1), value=0.0)

        max_hits = x_s.shape[1]
        Ax, Ay = x_pad[:, :max_hits],        y_pad[:, :max_hits]
        Bx, By = x_pad[:, 1:max_hits + 1],   y_pad[:, 1:max_hits + 1]
        Cx, Cy = x_pad[:, 2:max_hits + 2],   y_pad[:, 2:max_hits + 2]

        # Triplet is valid only when all three neighbours are real hits.
        triplet_valid = v_pad[:, :max_hits] * valid_s * v_pad[:, 2:max_hits + 2]

        # 2D cross product: (B-A) × (C-A) — positive when B is left of A→C.
        cross  = (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
        AC_len = ((Cx - Ax).pow(2) + (Cy - Ay).pow(2)).sqrt().clamp(min=1e-6)

        sagitta_s = (cross / AC_len) * triplet_valid  # zero boundary/padding hits

        # Unsort back to the original per-track hit order expected by input_net.
        return sagitta_s.gather(-1, inv_order)

    def _compute_relative_coords(
        self,
        track_inputs: dict[str, Tensor],
        helix_params: Tensor,  # (n_tracks, n_fields)
        input_name: str = "sihit",
    ) -> dict[str, Tensor]:
        """Inject hit coordinates relative to the helix perigee into track_inputs.

        The helix perigee (closest approach to the origin in 3D) is:
            x_vtx = -d0 * sin(phi0)
            y_vtx =  d0 * cos(phi0)
            z_vtx =  z0
        where d0, z0 are in metres (model-native units) and phi0 is in radians.

        For each hit the following fields are added (prefixed by input_name):
            {n}_x_rel, {n}_y_rel, {n}_z_rel : Cartesian offsets from perigee [m]
            {n}_r_rel   : transverse distance from perigee [m]
            {n}_phi_rel : azimuthal direction from perigee to hit [rad]
            {n}_eta_rel : pseudorapidity from perigee to hit
            {n}_dz_helix  : z residual from helix z(r) prediction [m]
            {n}_dxy_helix : signed transverse residual from helix circle [m]
            {n}_deta : hit global eta minus track eta (no wrapping) [dimensionless]
            {n}_dphi : hit global phi minus track phi0, wrapped to (-π, π) [rad]
        """
        n = input_name
        n_tracks = helix_params.shape[0]
        device   = helix_params.device
        dtype    = helix_params.dtype

        def _get(name: str) -> Tensor:
            """Return helix_params column for field *name*, or zeros if absent."""
            if name in self.fields:
                return helix_params[:, self.fields.index(name)]
            return torch.zeros(n_tracks, device=device, dtype=dtype)

        phi0 = _get("phi_perigee")   # track direction at perigee [rad]
        d0   = _get("d0_perigee_m")  # transverse impact parameter  [m]
        z0   = _get("z0_perigee_m")  # longitudinal impact parameter [m]

        # Perigee in the transverse plane (ACTS perigee convention):
        # the perigee lies along the direction perpendicular to phi0.
        x_vtx = (-d0 * torch.sin(phi0)).unsqueeze(1)  # (n_tracks, 1)
        y_vtx = ( d0 * torch.cos(phi0)).unsqueeze(1)
        z_vtx = z0.unsqueeze(1)

        x_rel = track_inputs[f"{n}_x"] - x_vtx  # (n_tracks, max_hits)
        y_rel = track_inputs[f"{n}_y"] - y_vtx
        z_rel = track_inputs[f"{n}_z"] - z_vtx
        r_rel = torch.sqrt(x_rel.pow(2) + y_rel.pow(2)).clamp(min=1e-6)

        out = dict(track_inputs)
        out[f"{n}_x_rel"]   = x_rel
        out[f"{n}_y_rel"]   = y_rel
        out[f"{n}_z_rel"]   = z_rel
        out[f"{n}_r_rel"]   = r_rel
        out[f"{n}_phi_rel"] = torch.atan2(y_rel, x_rel)
        out[f"{n}_eta_rel"] = torch.asinh(z_rel / r_rel)

        # x_prime / y_prime: hit position rotated into perigee frame (phi0 → 0).
        # x_prime points along the track direction at perigee; y_prime is transverse to it.
        cp = torch.cos(phi0).unsqueeze(1)  # (n_tracks, 1)
        sp = torch.sin(phi0).unsqueeze(1)
        out[f"{n}_x_prime"] =  x_rel * cp + y_rel * sp
        out[f"{n}_y_prime"] = -x_rel * sp + y_rel * cp

        # dz_helix: per-hit z residual from helix prediction z(r_xy) = z0 + r_xy*cot(theta).
        # Uses r_rel (2D transverse distance from helix perigee) as the arc-length proxy —
        # correct for low-curvature tracks.
        if "theta" in self.fields:
            theta_trk = _get("theta").unsqueeze(1)  # (n_tracks, 1)
            cot_theta = torch.cos(theta_trk) / torch.sin(theta_trk).clamp(min=1e-6)
            out[f"{n}_dz_helix"] = z_rel - r_rel * cot_theta
            # dz_signed: z_rel flipped so it always points in the forward hemisphere.
            # sign(pi/2 - theta) == sign(eta): positive for forward, negative for backward.
            out[f"{n}_dz_signed"] = z_rel * torch.sign(torch.pi / 2 - theta_trk)
        else:
            eta_trk = _get("eta").unsqueeze(1)  # (n_tracks, 1)
            out[f"{n}_dz_helix"] = z_rel - r_rel * torch.sinh(eta_trk)
            out[f"{n}_dz_signed"] = z_rel * torch.sign(eta_trk)

        # dxy_helix: per-hit signed transverse residual from the helix circle.
        # Positive = hit lies outside the circle, negative = inside.
        qopt_trk    = _get("qopt")                                              # (n_tracks,)
        R_helix     = (1.0 / (qopt_trk.abs().clamp(min=1e-6) * 0.3 * self.B_field)).unsqueeze(1)
        charge_sign = torch.sign(qopt_trk).unsqueeze(1)
        x_c = x_vtx + charge_sign * R_helix * torch.sin(phi0).unsqueeze(1)
        y_c = y_vtx - charge_sign * R_helix * torch.cos(phi0).unsqueeze(1)
        dist_xy = torch.sqrt(
            (track_inputs[f"{n}_x"] - x_c).pow(2) +
            (track_inputs[f"{n}_y"] - y_c).pow(2)
        ).clamp(min=1e-6)
        out[f"{n}_dxy_helix"] = dist_xy - R_helix

        # deta / dtheta / dphi: angular difference between the hit's global position
        # and the track's direction from the helix fit.
        # dphi is wrapped to (-π, π) via atan2 to handle the ±π boundary.
        if "theta" in self.fields:
            out[f"{n}_dtheta"] = track_inputs[f"{n}_theta"] - theta_trk
        if "eta" in self.fields:
            out[f"{n}_deta"] = track_inputs[f"{n}_eta"] - _get("eta").unsqueeze(1)
        hit_phi = track_inputs[f"{n}_phi"]
        out[f"{n}_dphi"] = torch.atan2(
            torch.sin(hit_phi - phi0.unsqueeze(1)),
            torch.cos(hit_phi - phi0.unsqueeze(1)),
        )

        return out

    def _assert_finite(self, t: Tensor, name: str) -> None:
        """Raise immediately if *t* contains any NaN or Inf.

        This performs a GPU→CPU sync; only call when ``self.debug`` is True.
        """
        if not torch.isfinite(t).all():
            n_nan = t.isnan().sum().item()
            n_inf = t.isinf().sum().item()
            raise ValueError(
                f"NaN/Inf in {name}: shape={tuple(t.shape)}, "
                f"nan={n_nan}, inf={n_inf}, "
                f"min={t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else 'N/A'}, "
                f"max={t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else 'N/A'}"
            )

    def _combine_fields(self, per_track_field: Tensor) -> Tensor:
        """Combine (n_valid, n_fields) per-track losses into a scalar stage loss.

        Two modes controlled by ``self.field_combination``:

        ``"mean"`` (default):
            Weighted arithmetic mean over fields of the per-track-mean losses.
            Equivalent to a weighted sum over fields of MADs/means.

        ``"geometric"``:
            Weighted geometric mean over fields of the per-field mean loss:
                exp( sum(w_f * log(mean_tracks L_f)) / sum(w_f) )
            Equivalent to minimising the product (times exponent weights) of
            per-field losses.  Fields with very poor loss cannot dominate the
            gradient signal — all fields are forced to improve simultaneously.
        """
        w = self.field_loss_weights  # (n_fields,)
        if self.field_combination == "geometric":
            per_field = per_track_field.mean(dim=0)  # (n_fields,)
            return torch.exp((w * torch.log(per_field.clamp(min=1e-10))).sum() / w.sum())
        # Default: weighted arithmetic mean over fields, mean over tracks.
        return (per_track_field * w).sum() / (w.sum() * per_track_field.shape[0])

    def _wrapped_residual(self, output: Tensor, target: Tensor) -> Tensor:
        """Per-field residuals with atan2 wrapping for angular fields."""
        diff    = output - target
        wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
        return torch.where(self.angular_mask, wrapped, diff)

    def _assemble_truth(self, targets: dict[str, Tensor]) -> Tensor:
        """Stack truth fields into a (1, n_tracks, n_fields) tensor."""
        parts = [
            targets[f"track_matched_particle_{f}"].unsqueeze(-1)
            for f in self.fields
        ]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Forward / predict / loss
    # ------------------------------------------------------------------

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        # ── 0. Gather hits ────────────────────────────────────────────────────
        # Always gather the combined sihit sequence for the helix fit and
        # pair_bias (needs x,y,z,r,s,phi,eta + is_vtxd for vtxd-only z-fit).
        sihit_geom = ["x", "y", "z", "r", "s", "theta", "phi", "is_vtxd"]
        track_inputs, track_hit_valid = _gather_track_hits(inputs, sihit_geom, "sihit")

        if len(self.input_nets) == 2:
            # Separate per-type gather for embedding.
            # nets[0] = vtxd (pixel), nets[1] = trkr (strip).
            vtxd_fields = [f for f in self.input_nets[0].fields if f not in _RELATIVE_HIT_FIELDS]
            trkr_fields = [f for f in self.input_nets[1].fields if f not in _RELATIVE_HIT_FIELDS]
            vtxd_inputs, vtxd_valid = _gather_track_hits(inputs, vtxd_fields, "vtxd")
            trkr_inputs, trkr_valid = _gather_track_hits(inputs, trkr_fields, "trkr")

        # 1. Classical helix fit on the combined sihit sequence.
        # Gather detector codes if available (non-CLD detectors with detector_int).
        sihit_det: Tensor | None = None
        if "sihit_detector_int" in inputs:
            det_gathered, _ = _gather_track_hits(inputs, ["detector_int"], "sihit")
            sihit_det = det_gathered["sihit_detector_int"]
        helix_params, helix_ok = self._helix_init(track_inputs, track_hit_valid, sihit_det)
        if self.debug:
            self._assert_finite(helix_params, "helix_params")

        # Optionally add per-hit signed three-point sagitta.
        if self.use_sagitta:
            track_inputs["sihit_sagitta"] = self._compute_sagitta(track_inputs, track_hit_valid)

        # 2. Embed and encode hits.
        if len(self.input_nets) == 1:
            # Single shared net: compute relative coords on the combined sihit
            # sequence (if requested) then embed directly.
            if self.use_relative_coords:
                track_inputs = self._compute_relative_coords(track_inputs, helix_params, input_name="sihit")
            pair_bias      = self._pair_bias(track_inputs)
            hit_embeds     = self.input_nets[0](track_inputs)
            combined_valid = track_hit_valid
        else:
            # Two nets: compute relative coords on vtxd and trkr separately,
            # embed separately, then concatenate for the encoder.
            if self.use_relative_coords:
                vtxd_inputs = self._compute_relative_coords(vtxd_inputs, helix_params, input_name="vtxd")
                trkr_inputs = self._compute_relative_coords(trkr_inputs, helix_params, input_name="trkr")
            # Pair bias on the vtxd-first concat sequence (sequence length must
            # match hit_embeds; pairwise geometric features are order-independent).
            vtxd_trkr_inputs = {
                "sihit_r":   torch.cat([vtxd_inputs["vtxd_r"],   trkr_inputs["trkr_r"]],   dim=1),
                "sihit_z":   torch.cat([vtxd_inputs["vtxd_z"],   trkr_inputs["trkr_z"]],   dim=1),
                "sihit_phi": torch.cat([vtxd_inputs["vtxd_phi"], trkr_inputs["trkr_phi"]], dim=1),
                "sihit_s":   torch.cat([vtxd_inputs["vtxd_s"],   trkr_inputs["trkr_s"]],   dim=1),
            }
            pair_bias      = self._pair_bias(vtxd_trkr_inputs)
            hit_embeds     = torch.cat([self.input_nets[0](vtxd_inputs),
                                        self.input_nets[1](trkr_inputs)], dim=1)
            combined_valid = torch.cat([vtxd_valid, trkr_valid], dim=1)

        if self.debug:
            self._assert_finite(hit_embeds, "hit_embeds")

        encoded_hits = self.encoder(hit_embeds, kv_mask=combined_valid, attn_bias=pair_bias)
        if self.debug:
            self._assert_finite(encoded_hits, "encoded_hits")

        # 3. Initialise track embedding from helix parameters.
        # Normalise by param_proj_scales (typical parameter magnitudes) so
        # param_proj sees O(1) inputs regardless of natural scale.
        current_params = helix_params                                                      # (n_tracks, n_fields)
        track_embed    = self.param_proj(helix_params / self.param_proj_scales).unsqueeze(1)  # (n_tracks, 1, dim)
        if self.debug:
            self._assert_finite(track_embed, "track_embed (after param_proj)")

        # 4. Iterative boosting stages.
        # stage_0 = raw helix (no correction), stage_1..n = after each BoostStage.
        # No loss is computed for stage_0.
        stage_outputs: dict[str, dict[str, Tensor]] = {
            "stage_0": {
                "track_preds": helix_params.unsqueeze(0),
            }
        }

        for stage_idx, stage in enumerate(self.stages):
            # Cross-attend track embedding against encoded hits.
            track_embed, delta, gates, log_sigma = stage(track_embed, encoded_hits, combined_valid)
            if self.debug:
                self._assert_finite(track_embed, f"track_embed (stage {stage_idx + 1})")
                self._assert_finite(delta, f"delta (stage {stage_idx + 1})")

            # Accumulate parameter correction.  delta is in normalised units;
            # multiply by field_unit_scales to recover model-native units.
            scaled_delta = delta * self.field_unit_scales
            if self.detach_stages:
                current_params = current_params.detach() + scaled_delta
            else:
                current_params = current_params + scaled_delta
            if self.debug:
                self._assert_finite(current_params, f"current_params (stage {stage_idx + 1})")

            # Store stage output (add leading batch dim for wrapper compatibility).
            stage_outputs[f"stage_{stage_idx + 1}"] = {
                "track_preds": current_params.unsqueeze(0),  # (1, n_tracks, n_fields)
                "gate_mean":   gates.mean(dim=0).detach(),   # (n_fields,) — monitoring only
                "delta_mad":   scaled_delta.abs().median(dim=0).values.detach() if scaled_delta.shape[0] > 0 else scaled_delta.new_zeros(scaled_delta.shape[1]),  # (n_fields,) in model units
                "log_sigma":   log_sigma,  # (n_tracks, n_fields) — used by gaussian_nll loss
            }

            # Re-inject current param estimate into track embedding before
            # the next stage, giving it explicit knowledge of current position
            # in parameter space.  Skip injection after the final stage.
            # Detach so this path is purely a positional hint: the loss from
            # stage k+1 should not flow back through the injection into stage
            # k's delta head (it already does so via the main accumulation).
            if self.inject_params and stage_idx < self.n_stages - 1:
                track_embed = track_embed + self.param_inject(current_params.detach() / self.param_proj_scales).unsqueeze(1)

        # "final" is an alias for the last stage — used by the ModelWrapper.
        stage_outputs["final"] = stage_outputs[f"stage_{self.n_stages}"]
        return stage_outputs

    def predict(
        self, outputs: dict[str, dict[str, Tensor]]
    ) -> dict[str, dict[str, Tensor]]:
        preds: dict[str, dict[str, Tensor]] = {}
        for stage_name, stage_out in outputs.items():
            params = stage_out["track_preds"]  # (1, n_tracks, n_fields)

            task_preds: dict[str, Tensor] = {}
            for i, field in enumerate(self.fields):
                task_preds[f"track_{field}"] = params[..., i]  # (1, n_tracks)
                for stat_key in ("norm_res_mean", "norm_res_std", "ema_scale", "gate_mean", "delta_mad", "barron_alpha", "barron_c"):
                    if stat_key in stage_out:
                        task_preds[f"track_{field}_{stat_key}"] = stage_out[stat_key][i]

            preds[stage_name] = {task.name: task_preds for task in self.tasks}
        return preds

    def loss(
        self,
        outputs: dict[str, dict[str, Tensor]],
        targets: dict[str, Tensor],
    ) -> tuple[dict, dict, dict]:
        truth = self._assemble_truth(targets)            # (1, n_tracks, n_fields)
        valid = targets["track_matched_particle_valid"]  # (1, n_tracks)

        losses: dict[str, dict[str, dict[str, Tensor]]] = {}

        for stage_idx in range(self.n_stages):
            stage_name = f"stage_{stage_idx + 1}"
            pred = outputs[stage_name]["track_preds"]  # (1, n_tracks, n_fields)

            # Select only truth-matched (valid) tracks.
            pred_valid  = pred[valid]   # (n_valid, n_fields)
            truth_valid = truth[valid]  # (n_valid, n_fields)
            if self.debug:
                self._assert_finite(pred_valid,  f"pred_valid  ({stage_name})")
                self._assert_finite(truth_valid, f"truth_valid ({stage_name})")

            residual = self._wrapped_residual(pred_valid, truth_valid)
            if self.debug:
                self._assert_finite(residual, f"residual ({stage_name})")

            if pred_valid.shape[0] > 0:
                # Always compute per-field scale and normalised residual for monitoring.
                scale    = self.ema_correction_std[stage_idx].clamp(min=1e-6)
                norm_res = residual / scale
                outputs[stage_name]["norm_res_mean"] = norm_res.mean(dim=0).detach()  # (n_fields,)
                outputs[stage_name]["norm_res_std"]  = norm_res.std(dim=0, correction=min(1, norm_res.shape[0] - 1)).detach()   # (n_fields,)
                outputs[stage_name]["ema_scale"]     = scale.detach()                  # (n_fields,)

                if self.loss_type in ("arcsinh_mad", "arcsinh_l1"):
                    # Shared arcsinh transform: arcsinh(r / MAD_ema).
                    # Gradient decays as 1/r for large |r| (bounded, unlike
                    # MSE) and never reaches zero (unlike Welsch), so outliers
                    # remain informative but not dominant.
                    #
                    # arcsinh_mad: squares the transform (MSE in arcsinh space).
                    #   Minimises at the arcsinh-weighted mean ≈ raw median.
                    #   For skewed distributions (e.g. d0/z0) this can introduce
                    #   a systematic positive/negative bias.
                    # arcsinh_l1: absolute value of the transform (MAE in arcsinh
                    #   space).  Minimises at the raw median → unbiased for
                    #   symmetric distributions and less biased for skewed ones.
                    if self.use_fixed_mad_scale:
                        mad_scale = self.fixed_mad_scale[stage_idx].clamp(min=1e-6)
                    else:
                        mad_scale  = self.ema_mad[stage_idx].clamp(min=1e-6)
                    loss_input = torch.arcsinh(residual / mad_scale)
                    if self.trim_fraction > 0.0 and loss_input.shape[0] > 1:
                        k = max(1, int(loss_input.shape[0] * (1.0 - self.trim_fraction)))
                        keep = loss_input.abs().mean(dim=-1).topk(k, largest=False).indices
                        loss_input = loss_input[keep]
                    if self.loss_type == "arcsinh_l1":
                        stage_loss = self._combine_fields(loss_input.abs())
                    else:
                        stage_loss = self._combine_fields(loss_input.pow(2))
                elif self.loss_type == "barron":
                    # Barron (CVPR 2019) adaptive robust loss.
                    # α = 2 - softplus(alpha_raw) ≤ 2; scale = EMA MAD (× exp(log_c_raw)
                    # when barron_learn_scale=True).
                    alpha     = 2.0 - F.softplus(self.alpha_raw[stage_idx])  # (n_fields,) α ≤ 2
                    mad_scale = self.ema_mad[stage_idx].clamp(min=1e-8)
                    if self.barron_learn_scale:
                        # Learned multiplicative offset on the scale: c = MAD × exp(log_c).
                        # log_c_raw=0 at init → c = MAD (same as fixed-scale baseline).
                        c = mad_scale * torch.exp(self.log_c_raw[stage_idx])
                    else:
                        c = mad_scale
                    loss_per = _barron_loss(residual, alpha, c)             # (n_valid, n_fields)
                    if self.trim_fraction > 0.0 and loss_per.shape[0] > 1:
                        k = max(1, int(loss_per.shape[0] * (1.0 - self.trim_fraction)))
                        keep = loss_per.mean(dim=-1).topk(k, largest=False).indices
                        loss_per = loss_per[keep]
                    stage_loss = self._combine_fields(loss_per)
                    outputs[stage_name]["barron_alpha"] = alpha.detach()    # (n_fields,)
                    if self.barron_learn_scale:
                        outputs[stage_name]["barron_c"] = c.detach()        # (n_fields,)
                elif self.loss_type == "gaussian_nll":
                    # Heteroscedastic Gaussian NLL: L = 0.5*(r̃/σ̃)² + log(σ̃)
                    # where r̃ = r / field_unit_scales (normalised residual, O(1)
                    # at ACTS-level performance) and σ̃ = exp(log_sigma) is
                    # predicted per-track per-field.  Normalising by field_unit_scales
                    # (≈ ACTS MAD per field) ensures σ̃ ≈ 1 at init is sensible,
                    # gradient magnitudes are comparable across fields, and the
                    # field_loss_weights act as intended.
                    unit_scales = self.field_unit_scales.clamp(min=1e-10)
                    r_norm = residual / unit_scales  # (n_valid, n_fields)
                    log_sigma_valid = outputs[stage_name]["log_sigma"][valid[0]]  # (n_valid, n_fields)
                    stage_loss = self._combine_fields(
                        0.5 * (r_norm / log_sigma_valid.exp()).pow(2) + log_sigma_valid
                    )
                else:
                    if not self.normalize_loss:
                        norm_res = residual

                    if self.trim_fraction > 0.0 and norm_res.shape[0] > 1:
                        # Trim the largest |residual| tracks (averaged across fields)
                        # so the loss focuses on sharpening the core rather than
                        # chasing outliers.  trim_fraction=0.1 drops the worst 10%.
                        abs_norm = norm_res.abs().mean(dim=-1)
                        k = max(1, int(norm_res.shape[0] * (1.0 - self.trim_fraction)))
                        keep = abs_norm.topk(k, largest=False).indices
                        norm_res = norm_res[keep]

                    # --- Loss on normalised residuals ---
                    # "welsch": Welsch/Gaussian-kernel loss  L = 1 - exp(-r²/2c²)
                    #   Gradient = (r/c²)·exp(-r²/2c²) — a Gaussian in r that
                    #   vanishes exponentially for |r|>c, completely ignoring
                    #   outliers and focusing entirely on the core (MAD/FWHM).
                    # "cauchy": Lorentzian NLL  L = log(1 + r²)
                    #   Gradient saturates slowly (∝ 1/r) — reduces tails (std)
                    #   but still receives non-trivial gradient from far outliers.
                    # "mse": standard mean-squared error.
                    # "l1": mean absolute error — targets the median.
                    if self.loss_type == "welsch":
                        c2 = self.welsch_c ** 2
                        stage_loss = self._combine_fields(1.0 - torch.exp(-norm_res.pow(2) / (2.0 * c2)))
                    elif self.loss_type == "cauchy":
                        stage_loss = self._combine_fields(torch.log1p(norm_res.pow(2)))
                    elif self.loss_type == "mse":
                        stage_loss = self._combine_fields(norm_res.pow(2))
                    elif self.loss_type == "l1":
                        stage_loss = self._combine_fields(norm_res.abs())
                    else:
                        raise ValueError(f"Unknown loss_type: {self.loss_type!r}")
            else:
                # No valid tracks — zero loss that still participates in the graph.
                stage_loss = pred_valid.sum() * 0.0

            # Update EMA *after* computing the loss so we normalise by
            # statistics from previous batches, not the current one.
            if self.training and pred_valid.shape[0] > 1:
                with torch.no_grad():
                    r = residual.detach()
                    # Scaled MAD: robust to the heavy tails in d0/z0/pt.
                    # 1.4826 * MAD = σ for a Gaussian, but ignores outliers.
                    current_mad = (r - r.median(dim=0).values).abs().median(dim=0).values.clamp(min=1e-6)
                    current_std = (1.4826 * current_mad).clamp(min=1e-6)
                    m = self.ema_momentum
                    self.ema_correction_std[stage_idx] = (
                        m * self.ema_correction_std[stage_idx] + (1 - m) * current_std
                    )
                    self.ema_mad[stage_idx] = (
                        m * self.ema_mad[stage_idx] + (1 - m) * current_mad
                    )

            # Earlier stages get a lower loss weight (analogous to auxiliary
            # losses in Deformable DETR); the final stage always gets 1.0.
            w = 1.0 if stage_idx == self.n_stages - 1 else self.stage_loss_weight
            losses[stage_name] = {"track_params": {"smooth_l1": w * stage_loss}}

        # "final" in outputs is an alias; do NOT add it to losses to avoid
        # double-counting the last stage.
        return outputs, targets, losses
