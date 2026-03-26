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
