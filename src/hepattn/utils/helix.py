
def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """
    Unwrap angles along the last dimension.

    phi: (..., M) in (-π, π] typically
    returns: (..., M) unwrapped
    """
    two_pi = 2.0 * torch.pi
    dphi = phi[..., 1:] - phi[..., :-1]

    step = torch.where(
        dphi > torch.pi, -two_pi,
        torch.where(dphi < -torch.pi, two_pi, torch.zeros_like(dphi))
    )

    offset = torch.cumsum(step, dim=-1)
    phi_unwrapped = phi.clone()
    phi_unwrapped[..., 1:] = phi_unwrapped[..., 1:] + offset
    return phi_unwrapped


def fit_helix_padded_1d(
    x: torch.Tensor,  # (B, M)
    y: torch.Tensor,  # (B, M)
    z: torch.Tensor,  # (B, M)
    valid: torch.Tensor,  # (B, M) bool
):
    """
    Fit 3D helices with padding.

    Args
    ----
    x, y, z: (B, M)
        Hit coordinates, padded along the last dim.
    valid: (B, M) bool
        True where (x, y, z) is a real hit.

    Returns
    -------
    dict of tensors, each of shape (B,):
        "R", "d0", "z0", "phi0", "xc", "yc",
        "tan_lambda", "alpha", "beta", "phi_dca"
    """
    if x.shape != y.shape or x.shape != z.shape or x.shape != valid.shape:
        raise ValueError("x, y, z, valid must all have the same shape (B, M).")

    B, M = x.shape
    orig_dtype = x.dtype
    device = x.device

    w = valid.to(torch.float32)  # weights 0/1

    sum_w = w.sum(dim=-1)                    # (B,)
    sum_w_safe = torch.where(sum_w > 0, sum_w,
                             torch.ones_like(sum_w))

    # ---------- 1) Circle fit in (x, y) with weights ----------
    ones = torch.ones_like(x)
    A = torch.stack([x, y, ones], dim=-1)    # (B, M, 3)
    b = -(x**2 + y**2)                       # (B, M)

    sqrt_w = torch.sqrt(w)                   # (B, M)
    A_w = A * sqrt_w.unsqueeze(-1)           # (B, M, 3)
    b_w = b * sqrt_w                         # (B, M)

    sol = torch.linalg.lstsq(A_w, b_w.unsqueeze(-1)).solution  # (B, 3, 1)
    D, E, F = sol.squeeze(-1).unbind(-1)                       # (B,)

    xc = -D / 2.0
    yc = -E / 2.0
    R_sq = (D**2 + E**2) / 4.0 - F
    R = torch.sqrt(R_sq)                     # (B,)

    # ---------- 2) Transverse parameters (d0, phi0) ----------
    C = torch.stack([xc, yc], dim=-1)        # (B, 2)
    c = torch.linalg.norm(C, dim=-1)         # (B,)
    c_safe = torch.where(c > 0, c, torch.ones_like(c))
    uC = C / c_safe.unsqueeze(-1)            # (B, 2)

    x_dca = xc - R * uC[..., 0]
    y_dca = yc - R * uC[..., 1]

    d0 = c - R                               # (B,)

    phi_dca_raw = torch.atan2(
        y_dca - yc,
        x_dca - xc,
    )                                        # (B,)

    phi0 = torch.remainder(
        phi_dca_raw + 0.5 * torch.pi + torch.pi,
        2.0 * torch.pi,
    ) - torch.pi                             # (B,)

    # ---------- 3) z(φ) linear fit with UNWRAPPED φ ----------
    phi_hits_raw = torch.atan2(
        y - yc.unsqueeze(-1),
        x - xc.unsqueeze(-1),
    )                                        # (B, M)

    phi_hits = unwrap_angle(phi_hits_raw)    # (B, M)

    # weighted means
    phi_mean = (w * phi_hits).sum(dim=-1) / sum_w_safe  # (B,)
    z_mean = (w * z).sum(dim=-1) / sum_w_safe           # (B,)

    dphi = phi_hits - phi_mean.unsqueeze(-1)
    dz = z - z_mean.unsqueeze(-1)

    cov = (w * dphi * dz).sum(dim=-1) / sum_w_safe      # (B,)
    var_phi = (w * dphi**2).sum(dim=-1) / sum_w_safe    # (B,)

    alpha = cov / var_phi                               # (B,)
    tan_lambda = alpha / R                              # (B,)

    two_pi = 2.0 * torch.pi
    k = torch.round((phi_mean - phi_dca_raw) / two_pi)  # (B,)
    phi_dca_unwrapped = phi_dca_raw + two_pi * k        # (B,)

    beta = z_mean - alpha * phi_mean                    # (B,)
    z0 = alpha * phi_dca_unwrapped + beta               # (B,)

    # ---------- 4) Validity mask for tracks ----------
    n_valid = sum_w                                     # (B,)
    enough_circle = n_valid >= 3
    nondeg_var = var_phi > 0
    good = enough_circle & nondeg_var

    def masked(v):
        nan = torch.full_like(v, float("nan"))
        return torch.where(good, v, nan)

    R = masked(R)
    d0 = masked(d0)
    z0 = masked(z0)
    phi0 = masked(phi0)
    tan_lambda = masked(tan_lambda)

    return R, phi0, tan_lambda, d0, z0


def fit_helix_per_particle(
    hit_x: torch.Tensor,           # (B, M)
    hit_y: torch.Tensor,           # (B, M)
    hit_z: torch.Tensor,           # (B, M)
    particle_hit_valid: torch.Tensor,  # (B, N, M) bool
    particle_valid: torch.Tensor = None,  # (B, N) bool, optional
):
    """
    Fit helices for each (event, particle) using your fit_helix_padded_1d.

    Returns
    -------
    R, phi0, tan_lambda, d0, z0 : each (B, N)
    """

    B, M = hit_x.shape
    B2, N, M2 = particle_hit_valid.shape
    assert B == B2 and M == M2, "Shape mismatch in hits vs particle_hit_valid"

    device = hit_x.device

    # ---- 1) Broadcast hits to (B, N, M) ----
    x_bnm = hit_x.unsqueeze(1).expand(B, N, M)
    y_bnm = hit_y.unsqueeze(1).expand(B, N, M)
    z_bnm = hit_z.unsqueeze(1).expand(B, N, M)

    # ---- 2) For each (b, n) build a sorted list of hit indices ----
    # indices 0..M-1 for each event
    idx = torch.arange(M, device=device).view(1, 1, M).expand(B, N, M)

    # use M as a sentinel for "no hit"
    sentinel = torch.full_like(idx, M)
    masked_idx = torch.where(particle_hit_valid, idx, sentinel)   # (B, N, M)

    # sort so that real hits (0..M-1) come before sentinel (M)
    sorted_idx, _ = torch.sort(masked_idx, dim=-1)                # (B, N, M)

    # how many hits per (b, n)
    n_hits = particle_hit_valid.sum(dim=-1)                       # (B, N)
    K = int(n_hits.max().item())                                  # max hits in batch

    if K == 0:
        # nothing to fit anywhere
        out_shape = (B, N)
        nan = hit_x.new_full(out_shape, float("nan"))
        return nan, nan, nan, nan, nan

    # keep only the first K positions
    sorted_idx = sorted_idx[..., :K]                              # (B, N, K)
    valid_bnk = sorted_idx < M                                    # (B, N, K)

    # avoid out-of-bounds when gathering
    gather_idx = sorted_idx.clamp(max=M - 1)

    # ---- 3) Gather the squashed coordinates (B, N, K) ----
    x_bnk = torch.gather(x_bnm, dim=-1, index=gather_idx)
    y_bnk = torch.gather(y_bnm, dim=-1, index=gather_idx)
    z_bnk = torch.gather(z_bnm, dim=-1, index=gather_idx)

    # also mask out whole particles if you have particle_valid
    if particle_valid is not None:
        valid_bnk = valid_bnk & particle_valid.unsqueeze(-1)

    # ---- 4) Flatten (B, N, K) -> (B*N, K) and fit ----
    B_tracks = B * N
    x_flat = x_bnk.reshape(B_tracks, K)
    y_flat = y_bnk.reshape(B_tracks, K)
    z_flat = z_bnk.reshape(B_tracks, K)
    valid_flat = valid_bnk.reshape(B_tracks, K)

    R, phi0, tan_lambda, d0, z0 = fit_helix_padded_1d(
        x_flat, y_flat, z_flat, valid_flat
    )

    # ---- 5) Reshape back to (B, N) ----
    R = R.view(B, N)
    phi0 = phi0.view(B, N)
    tan_lambda = tan_lambda.view(B, N)
    d0 = d0.view(B, N)
    z0 = z0.view(B, N)

    return R, phi0, tan_lambda, d0, z0
