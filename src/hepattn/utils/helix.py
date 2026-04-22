import numpy as np
import torch


def unwrap_angle_detached(phi: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    dphi = phi[..., 1:] - phi[..., :-1]
    step = torch.where(
        dphi > torch.pi, -two_pi,
        torch.where(dphi < -torch.pi, two_pi, torch.zeros_like(dphi))
    ).detach()
    offset = torch.cumsum(step, dim=-1)
    offset = torch.cat([torch.zeros_like(phi[..., :1]), offset], dim=-1)
    return phi + offset

def wrap_to_pi(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def _fit_helices_flat(
    x: torch.Tensor,      # (B, K)
    y: torch.Tensor,      # (B, K)
    z: torch.Tensor,      # (B, K)
    w: torch.Tensor,      # (B, K) weights for circle fit, in [0, 1]
    zfit_w: torch.Tensor | None = None,  # (B, K) weights for z(r) fit; defaults to w
    eps: float = 1e-8,
    default: float = 0.0,
):
    B, K = x.shape
    device = x.device
    out_dtype = x.dtype
    compute_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    default_t = torch.as_tensor(default, device=device, dtype=compute_dtype)

    with torch.autocast(device_type=device.type, enabled=False):
        x = x.to(compute_dtype)
        y = y.to(compute_dtype)
        z = z.to(compute_dtype)

        w = w.to(compute_dtype).clamp(0.0, 1.0)
        sqrt_w = torch.sqrt(w)
        sum_w = w.sum(dim=-1)
        sum_w_safe = sum_w.clamp_min(eps)

        # z(r)-fit weights — can be pixel-only to suppress poor-z-resolution strip hits
        zfit_w_t = w if zfit_w is None else zfit_w.to(compute_dtype).clamp(0.0, 1.0)
        sqrt_zw = torch.sqrt(zfit_w_t)
        sum_zw = zfit_w_t.sum(dim=-1)
        sum_zw_safe = sum_zw.clamp_min(eps)

        # Pre-compute r for all hits; used both for circle weighting and z(r) fit.
        r_hits = torch.sqrt(x * x + y * y)

        # ---- 1) circle fit (x,y) via weighted least squares
        A = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (B,K,3)
        b = -(x * x + y * y)                                 # (B,K)

        A_w = A * sqrt_w.unsqueeze(-1)
        b_w = b * sqrt_w

        AtA = A_w.transpose(-1, -2) @ A_w
        Atb = A_w.transpose(-1, -2) @ b_w.unsqueeze(-1)

        ridge3 = eps * AtA.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
        I3 = torch.eye(3, device=device, dtype=compute_dtype).expand(B, 3, 3)

        theta, info3 = torch.linalg.solve_ex(AtA + ridge3[:, None, None] * I3, Atb, check_errors=False)
        theta = theta.squeeze(-1)
        fit3_ok = (info3 == 0)

        D, E, F = theta.unbind(-1)
        xc = -0.5 * D
        yc = -0.5 * E
        R_sq = 0.25 * (D * D + E * E) - F
        R = torch.sqrt(torch.clamp(R_sq, min=0.0))

        # ---- 2) transverse params
        C = torch.stack([xc, yc], dim=-1)
        c = torch.linalg.norm(C, dim=-1)
        c_safe = torch.where(c > 0, c, torch.ones_like(c))
        uC = C / c_safe.unsqueeze(-1)

        x_dca = xc - R * uC[..., 0]
        y_dca = yc - R * uC[..., 1]

        phi_dca_raw = torch.atan2(y_dca - yc, x_dca - xc)
        # phi0 assuming counterclockwise motion; corrected for charge sign below.
        phi0 = wrap_to_pi(phi_dca_raw + 0.5 * torch.pi)

        # ---- 3) z(r) weighted fit: z ≈ sinh(eta) * r + z0
        # z(phi) is numerically unstable for high-pT tracks because the total
        # phi span is ~r/R << 1 rad, so errors in the circle centre bias every
        # phi value and corrupt the slope.  z vs r is exact for d0=0 and
        # accurate to O((r/R)^2) in general — negligible for r << R.
        #
        # zfit_w_t may be pixel-only (strips have poor z resolution), so use
        # sum_zw_safe and sqrt_zw here rather than the circle-fit quantities.
        r_mean = (zfit_w_t * r_hits).sum(dim=-1) / sum_zw_safe
        dr = r_hits - r_mean.unsqueeze(-1)
        var_r = (zfit_w_t * dr * dr).sum(dim=-1) / sum_zw_safe

        # phi_hits still needed for charge-sign inference below
        phi_hits_raw = torch.atan2(y - yc.unsqueeze(-1), x - xc.unsqueeze(-1))
        phi_hits = unwrap_angle_detached(phi_hits_raw)

        X = torch.stack([r_hits, torch.ones_like(r_hits)], dim=-1)  # (B,K,2)
        X_w = X * sqrt_zw.unsqueeze(-1)
        z_w = z * sqrt_zw

        XtX = X_w.transpose(-1, -2) @ X_w
        Xtz = X_w.transpose(-1, -2) @ z_w.unsqueeze(-1)

        ridge2 = eps * XtX.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
        I2 = torch.eye(2, device=device, dtype=compute_dtype).expand(B, 2, 2)

        ab, info2 = torch.linalg.solve_ex(XtX + ridge2[:, None, None] * I2, Xtz, check_errors=False)
        ab = ab.squeeze(-1)
        fit2_ok = (info2 == 0)

        slope, z0 = ab.unbind(-1)   # z = slope * r + z0,  slope = dz/dr = sinh(eta)
        eta = torch.asinh(slope)

        R_safe = torch.where(R > 0, R, torch.ones_like(R))

        # ---- 4) Charge-sign inference + ACTS-convention corrections
        # Hits are sorted by r (ascending) so phi_hits gives the angular
        # progression along the helix.  CCW (Δφ > 0) → negative charge;
        # CW (Δφ < 0) → positive charge.
        dphi_consec = phi_hits[..., 1:] - phi_hits[..., :-1]           # (B, K-1)
        w_pair      = (w[..., :-1] * w[..., 1:]).clamp(0.0, 1.0)       # (B, K-1)
        sum_dphi    = (w_pair * dphi_consec).sum(dim=-1)                # (B,)
        is_cw       = sum_dphi < 0  # clockwise → positive charge

        # Flip phi by π for CW tracks to match the ACTS perigee convention.
        phi0 = wrap_to_pi(phi0 + torch.where(is_cw,
                                              torch.full_like(phi0, torch.pi),
                                              torch.zeros_like(phi0)))

        # Numerically stable d0: F / (c + R) instead of c - R.
        # F = c² - R² from the circle fit; dividing by the *sum* (c + R) avoids
        # catastrophic cancellation when d0 ≪ R (e.g. d0 ~ 1 mm, R ~ 0.8 m).
        d0_unsigned = F / (c_safe + R_safe)
        d0 = torch.where(is_cw, -d0_unsigned, d0_unsigned)

        fit_successful = (sum_w >= 3.0) & (sum_zw >= 3.0) & (var_r > 0) & (R_sq > 0) & fit3_ok & fit2_ok

        def fill(v):
            return torch.where(fit_successful, v, default_t.expand_as(v))

        charge_sign_fit = torch.where(is_cw, torch.ones_like(R), -torch.ones_like(R))

        R, phi0, eta, d0, z0 = map(fill, (R, phi0, eta, d0, z0))
        # Default charge_sign to +1 for failed fits (masked out by fit_successful anyway).
        charge_sign_fit = torch.where(fit_successful, charge_sign_fit, torch.ones_like(charge_sign_fit))

    R, phi0, eta, d0, z0, charge_sign_fit = (t.to(out_dtype) for t in (R, phi0, eta, d0, z0, charge_sign_fit))
    return R, phi0, eta, d0, z0, fit_successful, charge_sign_fit


def helix_params_to_track_params(
    R: torch.Tensor,
    phi0: torch.Tensor,
    eta: torch.Tensor,
    d0: torch.Tensor,
    z0: torch.Tensor,
    B_field: float = 3.0,
    charge_sign: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw helix fit output to standard physical track parameters.

    ``_fit_helices_flat`` already infers the charge sign from the hit geometry
    and returns phi0/d0 in the ACTS perigee convention, so no correction is
    needed in the normal case.

    Pass ``charge_sign`` only to *override* the fit's inferred convention —
    for example when the caller independently knows the particle charge.

    Args:
        R:           helix radius [m]
        phi0:        azimuthal angle at DCA [rad] (counterclockwise convention)
        eta:         pseudorapidity
        d0:          transverse impact parameter [m] (counterclockwise convention)
        z0:          longitudinal impact parameter at DCA [m]
        B_field:     magnetic field strength [T]
        charge_sign: optional tensor with same shape as R; sign(q)

    Returns:
        pt [GeV], phi [rad], eta, d0_mm [mm], z0_mm [mm]
    """
    pt    = 0.3 * B_field * R
    phi   = phi0
    d0_mm = d0 * 1e3
    z0_mm = z0 * 1e3
    if charge_sign is not None:
        flip  = (charge_sign > 0).to(phi.dtype)
        phi   = wrap_to_pi(phi + flip * torch.pi)
        d0_mm = d0_mm * (1.0 - 2.0 * flip)   # negate where positive charge
    return pt, phi, eta, d0_mm, z0_mm


def fit_helices(
    hit_x: torch.Tensor,                 # (B, M)
    hit_y: torch.Tensor,                 # (B, M)
    hit_z: torch.Tensor,                 # (B, M)
    particle_hit_weight: torch.Tensor,   # (B, N, M) float in [0,1]  (or bool for old mode)
    particle_fittable: torch.Tensor,     # (B, N) bool
    default: float = 0.0,
    eps: float = 1e-8,
):
    B, M = hit_x.shape
    _, N, _ = particle_hit_weight.shape
    device = hit_x.device
    BN = B * N

    fit_mask = particle_fittable.to(torch.bool)

    # Backward compatible: if caller passes bool mask, treat it as weights {0,1}
    if particle_hit_weight.dtype == torch.bool:
        particle_hit_weight = particle_hit_weight.to(hit_x.dtype)

    # For packing/unwrap: treat strictly-zero weight as "absent"
    pos = (particle_hit_weight > 0)  # bool (B,N,M)

    n_pos = pos.sum(dim=-1)
    K = int(n_pos.masked_fill(~fit_mask, 0).max().item())

    R = hit_x.new_full((BN,), default)
    P = hit_x.new_full((BN,), default)
    E = hit_x.new_full((BN,), default)
    D = hit_x.new_full((BN,), default)
    Z = hit_x.new_full((BN,), default)
    CS = hit_x.new_full((BN,), 1.0)   # charge_sign, default +1
    fit_successful = torch.zeros((BN,), device=device, dtype=torch.bool)

    if K == 0 or not fit_mask.any():
        return (
            R.view(B, N), P.view(B, N), E.view(B, N),
            D.view(B, N), Z.view(B, N), fit_successful.view(B, N), CS.view(B, N)
        )

    # pack positive-weight hit indices to the front (avoids unwrap seeing gaps)
    idx = torch.arange(M, device=device).view(1, 1, M).expand(B, N, M)
    sorted_idx = idx.masked_fill(~pos, M).sort(dim=-1).values[..., :K]
    valid_bnk = sorted_idx < M
    gather_idx = sorted_idx.clamp_max(M - 1)

    x_bnk = hit_x[:, None, :].expand(B, N, M).gather(-1, gather_idx)
    y_bnk = hit_y[:, None, :].expand(B, N, M).gather(-1, gather_idx)
    z_bnk = hit_z[:, None, :].expand(B, N, M).gather(-1, gather_idx)

    w_bnk = particle_hit_weight.gather(-1, gather_idx)
    w_bnk = torch.where(valid_bnk, w_bnk, torch.zeros_like(w_bnk))

    x_flat = x_bnk.reshape(BN, K)
    y_flat = y_bnk.reshape(BN, K)
    z_flat = z_bnk.reshape(BN, K)
    w_flat = w_bnk.reshape(BN, K)
    m_flat = fit_mask.reshape(BN)

    R_sel, P_sel, E_sel, D_sel, Z_sel, ok_sel, cs_sel = _fit_helices_flat(
        x_flat[m_flat], y_flat[m_flat], z_flat[m_flat], w_flat[m_flat],
        default=default, eps=eps,
    )

    R[m_flat] = R_sel
    P[m_flat] = P_sel
    E[m_flat] = E_sel
    D[m_flat] = D_sel
    Z[m_flat] = Z_sel
    CS[m_flat] = cs_sel
    fit_successful[m_flat] = ok_sel

    return R.view(B, N), P.view(B, N), E.view(B, N), D.view(B, N), Z.view(B, N), fit_successful.view(B, N), CS.view(B, N)


# ---------------------------------------------------------------------------
# Added for ObjectPropertyRegressionTask: 3-point helix seed + perigee transport +
# residual helpers. Independent of the Gauss-Newton fit_helices pipeline above.
# ---------------------------------------------------------------------------

def kasa_circle_fit_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted Kasa circle fit in xy plane using torch.linalg.lstsq.

    Fits: x^2 + y^2 + D x + E y + F = 0.
    Denote M as number of hits.

    Args:
      x: Hit x-positions (..., M).
      y: Hit y-positions (..., M).
      w: Optional weights (..., M), non-negative.
      valid: Optional mask, so we do not fit on the padded slots (..., M) (bool or {0,1}).
      eps: Numerical stability.

    Returns:
      xc, yc, R: (...)  (same leading dims as x without last dim)
      theta: (..., 3) with [D, E, F]

    Raises:
      ValueError: If x, y, valid, or w shapes are inconsistent.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
    if valid is not None and valid.shape != x.shape:
        raise ValueError(f"valid_mask must match x/y shape, got {valid.shape} vs {x.shape}")
    if w is not None and w.shape != x.shape:
        raise ValueError(f"weights must match x/y shape, got {w.shape} vs {x.shape}")

    out_dtype = x.dtype
    solve_dtype = torch.float32 if out_dtype in {torch.float16, torch.bfloat16} else out_dtype

    x_s = x.to(dtype=solve_dtype)
    y_s = y.to(dtype=solve_dtype)

    # prepare weights
    sqrtw = None
    if w is not None:
        w_s = w.to(dtype=solve_dtype)
        if valid is not None:
            w_s = w_s * valid.to(dtype=solve_dtype)
        w_s = torch.clamp(w_s, min=0.0)  # make sure non-negative weights for differentiability
        sqrtw = torch.sqrt(w_s + eps)
    elif valid is not None:
        sqrtw = valid.to(dtype=solve_dtype)

    ones = torch.ones_like(x_s)
    A = torch.stack([x_s, y_s, ones], dim=-1)  # (..., M, 3)
    b = -(x_s * x_s + y_s * y_s)  # (..., M)

    # Weighted Kasa fit: scale the rows by square root of weights
    if sqrtw is not None:
        sw = sqrtw.unsqueeze(-1)  # (..., M, 1)
        A = A * sw
        b = b.unsqueeze(-1) * sw  # (..., M, 1)
    else:
        b = b.unsqueeze(-1)

    # Kasa equation: A * theta - b = 0, where theta = [D,E,F]^T
    theta_lead_shape = A.shape[:-2]
    M = A.shape[-2]
    A2 = A.reshape(-1, M, 3)
    b2 = b.reshape(-1, M, 1)

    sol = torch.linalg.lstsq(A2, b2).solution  # (..., 3, 1)
    theta = sol.squeeze(-1).reshape(*theta_lead_shape, 3)
    D, E, F_ = theta.unbind(-1)

    xc = -D / 2.0
    yc = -E / 2.0
    rad2 = (D * D + E * E) * 0.25 - F_
    rad2 = torch.clamp(rad2, min=eps)
    R = torch.sqrt(rad2)

    return xc.to(out_dtype), yc.to(out_dtype), R.to(out_dtype), theta.to(out_dtype)


def theta_from_tan_lambda(tan_lambda: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.ones_like(tan_lambda), tan_lambda)


def eta_from_theta(theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -torch.log(torch.tan(0.5 * theta).clamp_min(eps))


@torch.compiler.disable


def helix_seed_3pt(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    valid: torch.Tensor,
    min_hits: int = 5,
    eps: float = 1e-8,
    z_valid: torch.Tensor | None = None,
    z_phi_sort: bool = True,
    time: torch.Tensor | None = None,
    sort_mode: str = "rt",
) -> dict[str, torch.Tensor]:
    """Analytic helix seed from 3 representative hits.

    Picks three representative hits, fits an exact circle through them, then
    uses a linear z vs arc-length regression to obtain the full 5-parameter
    perigee state (d0, phi0, omega, z0, tan_lambda).

    Three sorting strategies control how hits are ordered:

    * ``"rt"`` (default): sort by transverse radius for everything.
    * ``"time"``: sort by hit time for everything (3-point selection,
      omega sign, arc-length accumulation).
    * ``"hybrid"``: rT sort for 3-point circle fit (best geometric lever
      arm), time sort for omega sign and arc-length ordering.

    Args:
        x: Hit x-positions (..., M).
        y: Hit y-positions (..., M).
        z: Hit z-positions (..., M).
        valid: Boolean mask (..., M).  Used for circle fit and hit counting.
        min_hits: Tracks with fewer valid hits get all-zero output.
        eps: Numerical stability constant.
        z_valid: Optional mask (..., M) for z regression.  If None, uses ``valid``.
        z_phi_sort: If True (default), refine rT ordering with a second pass
            that re-sorts by arc-length (phi around circle).  Ignored when
            ``sort_mode`` is ``"time"`` or ``"hybrid"``.
        time: Hit times (..., M).  Required when ``sort_mode`` is
            ``"time"`` or ``"hybrid"``.
        sort_mode: ``"rt"`` | ``"time"`` | ``"hybrid"``.

    Returns:
        Dict with keys: d0, phi0, omega, z0, tan_lambda, xc, yc, R, theta, eta, sagitta
        (all shape (...)).

    Raises:
        ValueError: If ``sort_mode`` requires time but ``time`` is None.
    """
    dtype = x.dtype
    device = x.device
    if z_valid is None:
        z_valid = valid
    if sort_mode in {"time", "hybrid"} and time is None:
        raise ValueError(f"sort_mode={sort_mode!r} requires the `time` tensor")

    # --- Step 1: Sort hits and select 3 representative points ---
    r_T = torch.sqrt(x * x + y * y)
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)
    n_valid = valid.sum(dim=-1)  # (...)

    # rT ordering (always needed for "rt" and "hybrid")
    rt_sort_key = torch.where(valid, r_T, inf)
    rt_order = torch.argsort(rt_sort_key, dim=-1)

    if sort_mode == "time":
        # Time ordering for everything
        time_sort_key = torch.where(valid, time, inf)
        primary_order = torch.argsort(time_sort_key, dim=-1)
    else:
        # rT ordering for 3-point selection ("rt" and "hybrid")
        primary_order = rt_order

    # Gather sorted coordinates using primary ordering
    x_s = torch.gather(x, -1, primary_order)
    y_s = torch.gather(y, -1, primary_order)
    z_s = torch.gather(z, -1, primary_order)

    # Select first / middle / last from primary ordering for circle fit
    idx_first = torch.zeros_like(n_valid)
    idx_mid = n_valid // 2
    idx_last = (n_valid - 1).clamp(min=0)

    idx = torch.stack([idx_first, idx_mid, idx_last], dim=-1)  # (..., 3)
    x3 = torch.gather(x_s, -1, idx)
    y3 = torch.gather(y_s, -1, idx)

    x1, x2, x3_ = x3[..., 0], x3[..., 1], x3[..., 2]
    y1, y2, y3_ = y3[..., 0], y3[..., 1], y3[..., 2]

    # --- Step 2: Exact 3-point circle fit ---
    D = 2.0 * (x1 * (y2 - y3_) + x2 * (y3_ - y1) + x3_ * (y1 - y2))
    D = torch.where(D.abs() < eps, torch.sign(D + eps) * eps, D)

    sq1 = x1 * x1 + y1 * y1
    sq2 = x2 * x2 + y2 * y2
    sq3 = x3_ * x3_ + y3_ * y3_

    xc = (sq1 * (y2 - y3_) + sq2 * (y3_ - y1) + sq3 * (y1 - y2)) / D
    yc = (sq1 * (x3_ - x2) + sq2 * (x1 - x3_) + sq3 * (x2 - x1)) / D
    R = torch.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2 + eps)

    # --- Step 3: Signed curvature omega ---
    if sort_mode == "hybrid":
        # For hybrid: use time-ordered 3 hits for omega sign (correct physical direction),
        # but keep the rT-selected circle fit above (better geometric lever arm).
        time_sort_key = torch.where(valid, time, inf)
        time_order = torch.argsort(time_sort_key, dim=-1)
        x_t = torch.gather(x, -1, time_order)
        y_t = torch.gather(y, -1, time_order)
        # Re-select first/mid/last from time ordering
        xt3 = torch.gather(x_t, -1, idx)
        yt3 = torch.gather(y_t, -1, idx)
        xt1, xt2, xt3_ = xt3[..., 0], xt3[..., 1], xt3[..., 2]
        yt1, yt2, yt3_ = yt3[..., 0], yt3[..., 1], yt3[..., 2]
        ux, uy = xt2 - xt1, yt2 - yt1
        vx, vy = xt3_ - xt2, yt3_ - yt2
    else:
        # For "rt" and "time": omega sign from primary ordering
        ux, uy = x2 - x1, y2 - y1
        vx, vy = x3_ - x2, y3_ - y2

    delta = ux * vy - uy * vx
    omega = torch.sign(delta) / R

    # --- Step 4-5: Perigee point, d0, phi0 ---
    rc = torch.sqrt(xc * xc + yc * yc + eps)
    x0 = xc - R * xc / rc
    y0 = yc - R * yc / rc

    # Radius vector from center to perigee
    rx = x0 - xc
    ry = y0 - yc

    # Tangent at perigee (oriented by rotation sense)
    s_rot = torch.sign(delta)
    tx = s_rot * (-ry)
    ty = s_rot * rx

    phi0 = torch.atan2(ty, tx)
    # EDM4HEP / LCIO d0 sign convention: d0 = -x * sin(phi) + y * cos(phi)
    # (previously used the opposite sign; empirical test on 2026-04-19 showed this
    #  flip tightens the sitrack residual IQR 65x — see d0_sign_convention_mismatch.md)
    d0 = y0 * torch.cos(phi0) - x0 * torch.sin(phi0)

    # --- Step 6: Arc-lengths for ALL valid hits ---
    alpha0 = torch.atan2(y0 - yc, x0 - xc)  # (...)

    def wrap(a):
        return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

    if sort_mode in {"time", "hybrid"}:
        # Use time ordering for arc-length accumulation
        if sort_mode == "hybrid":
            # time_order already computed above
            pass
        else:
            time_order = primary_order  # already time-sorted

        x_time = torch.gather(x, -1, time_order)
        y_time = torch.gather(y, -1, time_order)
        z_time = torch.gather(z, -1, time_order)

        alpha_time = torch.atan2(y_time - yc.unsqueeze(-1), x_time - xc.unsqueeze(-1))
        alpha_seq = torch.cat([alpha0.unsqueeze(-1), alpha_time], dim=-1)
        da_cum = torch.cumsum(wrap(alpha_seq[..., 1:] - alpha_seq[..., :-1]), dim=-1)
        s_all = R.unsqueeze(-1) * da_cum * s_rot.unsqueeze(-1)

        z_fin = z_time
        z_valid_fin = torch.gather(z_valid, -1, time_order)
    else:
        # Original rT-based arc-length computation
        alpha_rT = torch.atan2(y_s - yc.unsqueeze(-1), x_s - xc.unsqueeze(-1))  # (..., M)
        alpha_seq_rT = torch.cat([alpha0.unsqueeze(-1), alpha_rT], dim=-1)  # (..., M+1)
        da_cum_rT = torch.cumsum(wrap(alpha_seq_rT[..., 1:] - alpha_seq_rT[..., :-1]), dim=-1)
        s_rT = R.unsqueeze(-1) * da_cum_rT * s_rot.unsqueeze(-1)  # (..., M)

        valid_rT = torch.gather(valid, -1, rt_order)  # (..., M)
        z_valid_rT = torch.gather(z_valid, -1, rt_order)  # (..., M) z-regression mask

        if z_phi_sort:
            # Pass 2: Re-sort by arc-length (phi refinement), redo accumulation.
            s_sort_key = torch.where(valid_rT, s_rT, inf)
            arc_order = torch.argsort(s_sort_key, dim=-1)  # (..., M)

            alpha_fin = torch.gather(alpha_rT, -1, arc_order)
            z_fin_rT = torch.gather(z_s, -1, arc_order)
            z_valid_fin_rT = torch.gather(z_valid_rT, -1, arc_order)

            alpha_seq = torch.cat([alpha0.unsqueeze(-1), alpha_fin], dim=-1)
            da_cum = torch.cumsum(wrap(alpha_seq[..., 1:] - alpha_seq[..., :-1]), dim=-1)
            s_all = R.unsqueeze(-1) * da_cum * s_rot.unsqueeze(-1)
            z_fin = z_fin_rT
            z_valid_fin = z_valid_fin_rT
        else:
            z_fin = z_s
            z_valid_fin = z_valid_rT
            s_all = s_rT

    # --- Step 7: Masked linear regression z(s) over z_valid hits ---
    w = z_valid_fin.to(dtype)  # (..., M)
    n_w = w.sum(dim=-1).clamp(min=1)

    s_mean = (w * s_all).sum(dim=-1) / n_w
    z_mean = (w * z_fin).sum(dim=-1) / n_w

    ds = s_all - s_mean.unsqueeze(-1)
    dz = z_fin - z_mean.unsqueeze(-1)

    cov = (w * ds * dz).sum(dim=-1)
    var = (w * ds * ds).sum(dim=-1)
    tan_lambda = cov / (var + eps)
    z0_fit = z_mean - tan_lambda * s_mean

    # --- Step 6 (derived): theta and eta ---
    theta = theta_from_tan_lambda(tan_lambda)
    eta = eta_from_theta(theta, eps=eps)

    # --- Sagitta: dimensionless bending measure s/L ---
    # Perpendicular distance from chord midpoint (hit1→hit3) to middle hit (hit2)
    chord_mid_x = (x1 + x3_) * 0.5
    chord_mid_y = (y1 + y3_) * 0.5
    chord_len = torch.sqrt((x3_ - x1) ** 2 + (y3_ - y1) ** 2 + eps)
    sagitta_dist = torch.sqrt((x2 - chord_mid_x) ** 2 + (y2 - chord_mid_y) ** 2 + eps)
    sagitta = sagitta_dist / (chord_len + eps)

    # --- Step 7: Zero masking for tracks with too few hits ---
    enough = (n_valid >= min_hits).to(dtype)

    return {
        "d0": d0 * enough,
        "phi0": phi0 * enough,
        "omega": omega * enough,
        "z0": z0_fit * enough,
        "tan_lambda": tan_lambda * enough,
        "xc": xc * enough,
        "yc": yc * enough,
        "R": R * enough,
        "theta": theta * enough,
        "eta": eta * enough,
        "sagitta": sagitta * enough,
    }


def pt_from_radius(r: torch.Tensor, b_field_t: float) -> torch.Tensor:
    """pT[GeV] = 0.3 * B[T] * R[m]."""
    return 0.3 * float(b_field_t) * r


def helix_perigee_from_vertex(
    vtx_x: np.ndarray,
    vtx_y: np.ndarray,
    vtx_z: np.ndarray,
    mom_x: np.ndarray,
    mom_y: np.ndarray,
    mom_z: np.ndarray,
    charge: np.ndarray,
    b_field: float = 2.0,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Transport truth particle parameters from creation vertex to perigee.

    Computes the 5 perigee track parameters (d0, phi0, omega, z0, tan_lambda)
    by propagating a helical trajectory from the MC truth creation vertex back
    to the point of closest approach to the z-axis (beam line).

    Convention follows helix_seed_3pt: CCW → omega > 0, CW → omega < 0.
    For B_z > 0: positive charge → CW → omega < 0.

    Args:
        vtx_x: Creation vertex x-position in mm, shape (N,).
        vtx_y: Creation vertex y-position in mm, shape (N,).
        vtx_z: Creation vertex z-position in mm, shape (N,).
        mom_x: Momentum x-component at creation vertex in GeV, shape (N,).
        mom_y: Momentum y-component at creation vertex in GeV, shape (N,).
        mom_z: Momentum z-component at creation vertex in GeV, shape (N,).
        charge: Particle charge (+-1 or 0 for neutrals), shape (N,).
        b_field: Magnetic field B_z in Tesla (default 2.0).
        eps: Numerical stability constant.

    Returns:
        Dict with keys d0 [mm], phi0 [rad], omega [1/mm], z0 [mm], tan_lambda.
        Neutral particles (charge=0) get all zeros.
    """
    N = len(vtx_x)
    zeros = np.zeros(N, dtype=np.float64)

    # Work in float64 for precision
    vx = np.asarray(vtx_x, dtype=np.float64)
    vy = np.asarray(vtx_y, dtype=np.float64)
    vz = np.asarray(vtx_z, dtype=np.float64)
    px = np.asarray(mom_x, dtype=np.float64)
    py = np.asarray(mom_y, dtype=np.float64)
    pz = np.asarray(mom_z, dtype=np.float64)
    q = np.asarray(charge, dtype=np.float64)

    # Mask: only process charged particles with nonzero pT
    pT = np.sqrt(px**2 + py**2)
    valid = (q != 0) & (pT > eps)

    # Output arrays
    d0 = zeros.copy()
    phi0 = zeros.copy()
    omega = zeros.copy()
    z0 = zeros.copy()
    tan_lambda = zeros.copy()

    if not valid.any():
        return {"d0": d0, "phi0": phi0, "omega": omega, "z0": z0, "tan_lambda": tan_lambda}

    # Select valid particles
    vx, vy, vz = vx[valid], vy[valid], vz[valid]
    px, py, pz = px[valid], py[valid], pz[valid]
    q_v = q[valid]
    pT_v = pT[valid]

    # Helix radius in mm (positions are mm, pT in GeV)
    R = pT_v / (0.3 * b_field) * 1000.0

    # Helix invariants
    tan_l = pz / pT_v
    s_rot = np.sign(-q_v)  # sign(omega): +q → CW → omega<0 → s_rot=-1

    # Tangent and left normal at creation vertex
    tx_v = px / pT_v
    ty_v = py / pT_v
    nlx = -ty_v  # left normal
    nly = tx_v

    # Circle center: C = vertex + s_rot * R * n_left
    xc = vx + s_rot * R * nlx
    yc = vy + s_rot * R * nly

    # Perigee: closest point on circle to z-axis
    rc = np.sqrt(xc**2 + yc**2)
    rc_safe = np.where(rc > eps, rc, eps)
    x0 = xc - R * xc / rc_safe
    y0 = yc - R * yc / rc_safe

    # Tangent at perigee
    rx = x0 - xc
    ry = y0 - yc
    tx_p = s_rot * (-ry)
    ty_p = s_rot * rx
    phi0_v = np.arctan2(ty_p, tx_p)

    # Signed transverse impact parameter — EDM4HEP / LCIO convention
    # d0 = -x * sin(phi) + y * cos(phi)  (see d0_sign_convention_mismatch.md, 2026-04-19)
    d0_v = y0 * np.cos(phi0_v) - x0 * np.sin(phi0_v)

    # z0: propagate via arc-length from creation vertex to perigee
    alpha_vtx = np.arctan2(vy - yc, vx - xc)
    alpha_0 = np.arctan2(y0 - yc, x0 - xc)
    dalpha = (alpha_vtx - alpha_0 + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-pi, pi]
    s_arc = dalpha * R * s_rot
    z0_v = vz - s_arc * tan_l

    # Omega: signed curvature in 1/mm
    omega_v = s_rot / R

    # Write back to output arrays
    d0[valid] = d0_v
    phi0[valid] = phi0_v
    omega[valid] = omega_v
    z0[valid] = z0_v
    tan_lambda[valid] = tan_l

    return {
        "d0": d0.astype(np.float32),
        "phi0": phi0.astype(np.float32),
        "omega": omega.astype(np.float32),
        "z0": z0.astype(np.float32),
        "tan_lambda": tan_lambda.astype(np.float32),
    }


def compute_helix_residuals(
    helix: dict[str, torch.Tensor],
    x_m: torch.Tensor,
    y_m: torch.Tensor,
    z_m: torch.Tensor,
    mask_tracker: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute per-hit residuals from 3-point helix seed to all assigned tracker hits.

    Returns (B, N) summary statistics: circle_rms, z_rms, max_circle_res, circle_mean, z_mean.
    """
    B, N, M = mask_tracker.shape
    xc = helix["xc"].unsqueeze(-1)
    yc = helix["yc"].unsqueeze(-1)
    R = helix["R"].unsqueeze(-1)
    z0 = helix["z0"].unsqueeze(-1)
    tan_lambda = helix["tan_lambda"].unsqueeze(-1)

    xh = x_m.unsqueeze(1).expand(B, N, M)
    yh = y_m.unsqueeze(1).expand(B, N, M)
    zh = z_m.unsqueeze(1).expand(B, N, M)

    dist = torch.sqrt((xh - xc) ** 2 + (yh - yc) ** 2 + eps)
    circle_res = dist - R

    alpha_hit = torch.atan2(yh - yc, xh - xc)

    rc = torch.sqrt(xc**2 + yc**2 + eps)
    x0 = xc - R * xc / rc
    y0 = yc - R * yc / rc
    alpha0 = torch.atan2(y0 - yc, x0 - xc)

    r_T = torch.sqrt(xh**2 + yh**2)
    sort_key = torch.where(mask_tracker, r_T, torch.full_like(r_T, float("inf")))
    order = torch.argsort(sort_key, dim=-1)

    alpha_sorted = torch.gather(alpha_hit, -1, order)
    zh_sorted = torch.gather(zh, -1, order)
    mask_sorted = torch.gather(mask_tracker, -1, order)
    circle_res_sorted = torch.gather(circle_res, -1, order)

    def wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

    deltas = torch.zeros_like(alpha_sorted)
    deltas[..., 0] = wrap(alpha_sorted[..., 0] - alpha0.squeeze(-1))
    deltas[..., 1:] = wrap(alpha_sorted[..., 1:] - alpha_sorted[..., :-1])
    da = torch.cumsum(deltas, dim=-1)

    s = R * da
    z_pred = z0 + s * tan_lambda
    z_res = zh_sorted - z_pred

    mask_f = mask_sorted.float()
    n_valid = mask_f.sum(-1).clamp(min=1.0)

    circle_rms = torch.sqrt((circle_res_sorted**2 * mask_f).sum(-1) / n_valid)
    z_rms = torch.sqrt((z_res**2 * mask_f).sum(-1) / n_valid)
    max_circle_res = (circle_res_sorted.abs() * mask_f).amax(dim=-1)
    circle_mean = (circle_res_sorted * mask_f).sum(-1) / n_valid
    z_mean = (z_res * mask_f).sum(-1) / n_valid

    return {
        "circle_rms": circle_rms,
        "z_rms": z_rms,
        "max_circle_res": max_circle_res,
        "circle_mean": circle_mean,
        "z_mean": z_mean,
    }
