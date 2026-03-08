import torch


def unwrap_angle_detached(phi: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    dphi = phi[..., 1:] - phi[..., :-1]
    step = torch.where(dphi > torch.pi, -two_pi, torch.where(dphi < -torch.pi, two_pi, torch.zeros_like(dphi))).detach()
    offset = torch.cumsum(step, dim=-1)
    offset = torch.cat([torch.zeros_like(phi[..., :1]), offset], dim=-1)
    return phi + offset


def wrap_to_pi(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi), torch.cos(phi))


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


def helix_seed_3pt(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    valid: torch.Tensor,
    min_hits: int = 5,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Analytic helix seed from 3 representative hits (first/middle/last by r_T).

    Picks the first, middle and last valid hit sorted by transverse radius,
    fits an exact circle through them, and performs a linear z vs arc-length
    regression to obtain the full 5-parameter perigee state
    (d0, phi0, omega, z0, tan_lambda).

    Args:
        x: Hit x-positions (..., M).
        y: Hit y-positions (..., M).
        z: Hit z-positions (..., M).
        valid: Boolean mask (..., M).
        min_hits: Tracks with fewer valid hits get all-zero output.
        eps: Numerical stability constant.

    Returns:
        Dict with keys: d0, phi0, omega, z0, tan_lambda, xc, yc, R, theta, eta
        (all shape (...)).
    """
    dtype = x.dtype
    device = x.device

    # --- Step 1: Sort by transverse radius and select 3 hits ---
    r_T = torch.sqrt(x * x + y * y)
    # Invalid hits sort last
    sort_key = torch.where(valid, r_T, torch.tensor(float("inf"), dtype=dtype, device=device))
    order = torch.argsort(sort_key, dim=-1)

    # Gather sorted coordinates
    x_s = torch.gather(x, -1, order)
    y_s = torch.gather(y, -1, order)
    z_s = torch.gather(z, -1, order)

    n_valid = valid.sum(dim=-1)  # (...)

    # Indices for first / middle / last
    idx_first = torch.zeros_like(n_valid)
    idx_mid = n_valid // 2
    idx_last = (n_valid - 1).clamp(min=0)

    # Stack indices and gather the 3 selected hits
    idx = torch.stack([idx_first, idx_mid, idx_last], dim=-1)  # (..., 3)
    x3 = torch.gather(x_s, -1, idx)
    y3 = torch.gather(y_s, -1, idx)
    z3 = torch.gather(z_s, -1, idx)

    x1, x2, x3_ = x3[..., 0], x3[..., 1], x3[..., 2]
    y1, y2, y3_ = y3[..., 0], y3[..., 1], y3[..., 2]
    z1, z2, z3_ = z3[..., 0], z3[..., 1], z3[..., 2]

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
    # Hits are sorted inner→outer by r_T, so u and v point in the direction of travel.
    # delta > 0 → CCW (turning left outward); delta < 0 → CW (turning right outward).
    # Convention matches helix_propagate_to_plane: omega > 0 → rho > 0 → CCW.
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
    d0 = x0 * torch.sin(phi0) - y0 * torch.cos(phi0)

    # --- Step 6: Arc-lengths for the 3 selected hits ---
    alpha1 = torch.atan2(y1 - yc, x1 - xc)
    alpha2 = torch.atan2(y2 - yc, x2 - xc)
    alpha3 = torch.atan2(y3_ - yc, x3_ - xc)
    alpha0 = torch.atan2(y0 - yc, x0 - xc)

    # Wrap angle difference to (-pi, pi]
    def wrap(a):
        return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

    # Sequential accumulation: each consecutive step is well under π,
    # so wrap is unambiguous even for near-looping tracks.
    da_01 = wrap(alpha1 - alpha0)
    da_12 = wrap(alpha2 - alpha1)
    da_23 = wrap(alpha3 - alpha2)

    da1 = da_01
    da2 = da_01 + da_12
    da3 = da_01 + da_12 + da_23

    s1 = R * da1
    s2 = R * da2
    s3 = R * da3

    # --- Step 7: Linear regression z(s) = z0_fit + s * tan_lambda ---
    s_bar = (s1 + s2 + s3) / 3.0
    z_bar = (z1 + z2 + z3_) / 3.0

    ds1, ds2, ds3 = s1 - s_bar, s2 - s_bar, s3 - s_bar
    dz1, dz2, dz3 = z1 - z_bar, z2 - z_bar, z3_ - z_bar

    cov = ds1 * dz1 + ds2 * dz2 + ds3 * dz3
    var = ds1 * ds1 + ds2 * ds2 + ds3 * ds3
    tan_lambda = cov / (var + eps)
    z0_fit = z_bar - tan_lambda * s_bar

    # --- Step 6 (derived): theta and eta ---
    theta = theta_from_tan_lambda(tan_lambda)
    eta = eta_from_theta(theta, eps=eps)

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
    }


def pt_from_radius(r: torch.Tensor, b_field_t: float) -> torch.Tensor:
    """pT[GeV] = 0.3 * B[T] * R[m]."""
    return 0.3 * float(b_field_t) * r


def _fit_helices_flat(
    x: torch.Tensor,  # (B, K)
    y: torch.Tensor,  # (B, K)
    z: torch.Tensor,  # (B, K)
    w: torch.Tensor,  # (B, K) weights in [0, 1]
    eps: float = 1e-8,
    default: float = 0.0,
):
    B, _K = x.shape
    device = x.device
    out_dtype = x.dtype
    compute_dtype = torch.float32 if out_dtype in {torch.float16, torch.bfloat16} else out_dtype
    default_t = torch.as_tensor(default, device=device, dtype=compute_dtype)

    with torch.autocast(device_type=device.type, enabled=False):
        x = x.to(compute_dtype)
        y = y.to(compute_dtype)
        z = z.to(compute_dtype)

        w = w.to(compute_dtype).clamp(0.0, 1.0)
        sqrt_w = torch.sqrt(w)
        sum_w = w.sum(dim=-1)
        sum_w_safe = sum_w.clamp_min(eps)

        # ---- 1) circle fit (x,y) via weighted least squares
        A = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (B,K,3)
        b = -(x * x + y * y)  # (B,K)

        A_w = A * sqrt_w.unsqueeze(-1)
        b_w = b * sqrt_w

        AtA = A_w.transpose(-1, -2) @ A_w
        Atb = A_w.transpose(-1, -2) @ b_w.unsqueeze(-1)

        ridge3 = eps * AtA.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
        I3 = torch.eye(3, device=device, dtype=compute_dtype).expand(B, 3, 3)

        theta, info3 = torch.linalg.solve_ex(AtA + ridge3[:, None, None] * I3, Atb, check_errors=False)
        theta = theta.squeeze(-1)
        fit3_ok = info3 == 0

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
        d0 = c - R

        phi_dca_raw = torch.atan2(y_dca - yc, x_dca - xc)
        phi0 = wrap_to_pi(phi_dca_raw + 0.5 * torch.pi)

        # ---- 3) z(phi) weighted fit
        phi_hits_raw = torch.atan2(y - yc.unsqueeze(-1), x - xc.unsqueeze(-1))
        phi_hits = unwrap_angle_detached(phi_hits_raw)

        phi_mean = (w * phi_hits).sum(dim=-1) / sum_w_safe
        dphi = phi_hits - phi_mean.unsqueeze(-1)
        var_phi = (w * dphi * dphi).sum(dim=-1) / sum_w_safe

        X = torch.stack([phi_hits, torch.ones_like(phi_hits)], dim=-1)  # (B,K,2)
        X_w = X * sqrt_w.unsqueeze(-1)
        z_w = z * sqrt_w

        XtX = X_w.transpose(-1, -2) @ X_w
        Xtz = X_w.transpose(-1, -2) @ z_w.unsqueeze(-1)

        ridge2 = eps * XtX.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
        I2 = torch.eye(2, device=device, dtype=compute_dtype).expand(B, 2, 2)

        ab, info2 = torch.linalg.solve_ex(XtX + ridge2[:, None, None] * I2, Xtz, check_errors=False)
        ab = ab.squeeze(-1)
        fit2_ok = info2 == 0

        alpha, beta = ab.unbind(-1)

        R_safe = torch.where(R > 0, R, torch.ones_like(R))
        tan_lambda = alpha / R_safe
        eta = torch.asinh(tan_lambda)

        two_pi = 2.0 * torch.pi
        k = torch.round((phi_mean - phi_dca_raw) / two_pi).detach()
        phi_dca_unwrapped = phi_dca_raw + two_pi * k
        z0 = alpha * phi_dca_unwrapped + beta

        # "expected hit count" threshold via sum of weights
        fit_successful = (sum_w >= 3.0) & (var_phi > 0) & (R_sq > 0) & fit3_ok & fit2_ok

        def fill(v):
            return torch.where(fit_successful, v, default_t.expand_as(v))

        R, phi0, eta, d0, z0 = map(fill, (R, phi0, eta, d0, z0))

    R, phi0, eta, d0, z0 = (t.to(out_dtype) for t in (R, phi0, eta, d0, z0))
    return R, phi0, eta, d0, z0, fit_successful


def fit_helices(
    hit_x: torch.Tensor,  # (B, M)
    hit_y: torch.Tensor,  # (B, M)
    hit_z: torch.Tensor,  # (B, M)
    particle_hit_weight: torch.Tensor,  # (B, N, M) float in [0,1]  (or bool for old mode)
    particle_fittable: torch.Tensor,  # (B, N) bool
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
    pos = particle_hit_weight > 0  # bool (B,N,M)

    n_pos = pos.sum(dim=-1)
    K = int(n_pos.masked_fill(~fit_mask, 0).max().item())

    R = hit_x.new_full((BN,), default)
    P = hit_x.new_full((BN,), default)
    E = hit_x.new_full((BN,), default)
    D = hit_x.new_full((BN,), default)
    Z = hit_x.new_full((BN,), default)
    fit_successful = torch.zeros((BN,), device=device, dtype=torch.bool)

    if K == 0 or not fit_mask.any():
        return (R.view(B, N), P.view(B, N), E.view(B, N), D.view(B, N), Z.view(B, N), fit_successful.view(B, N))

    # pack positive-weight hit indices to the front (avoids unwrap seeing gaps)
    idx = torch.arange(M, device=device).view(1, 1, M).expand(B, N, M)
    sorted_idx = idx.masked_fill(~pos, M).sort(dim=-1).values[..., :K]  # <- FIX HERE
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

    R_sel, P_sel, E_sel, D_sel, Z_sel, ok_sel = _fit_helices_flat(
        x_flat[m_flat],
        y_flat[m_flat],
        z_flat[m_flat],
        w_flat[m_flat],
        default=default,
        eps=eps,
    )

    R[m_flat] = R_sel
    P[m_flat] = P_sel
    E[m_flat] = E_sel
    D[m_flat] = D_sel
    Z[m_flat] = Z_sel
    fit_successful[m_flat] = ok_sel

    return R.view(B, N), P.view(B, N), E.view(B, N), D.view(B, N), Z.view(B, N), fit_successful.view(B, N)
