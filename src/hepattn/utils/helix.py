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
    w: torch.Tensor,      # (B, K) weights in [0, 1]
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
        fit2_ok = (info2 == 0)

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
    fit_successful = torch.zeros((BN,), device=device, dtype=torch.bool)

    if K == 0 or not fit_mask.any():
        return (
            R.view(B, N), P.view(B, N), E.view(B, N),
            D.view(B, N), Z.view(B, N), fit_successful.view(B, N)
        )

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
        x_flat[m_flat], y_flat[m_flat], z_flat[m_flat], w_flat[m_flat],
        default=default, eps=eps,
    )

    R[m_flat] = R_sel
    P[m_flat] = P_sel
    E[m_flat] = E_sel
    D[m_flat] = D_sel
    Z[m_flat] = Z_sel
    fit_successful[m_flat] = ok_sel

    return R.view(B, N), P.view(B, N), E.view(B, N), D.view(B, N), Z.view(B, N), fit_successful.view(B, N)
