import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch

plt.rcParams["figure.dpi"] = 300

_B_FIELD_T = 2.0     # CLD solenoid field [T]
_HELIX_CLIP_M = 1.5  # clip helix paths at tracker radius [m]


def _build_helix_path(
    phi: torch.Tensor,
    eta: torch.Tensor,
    pt: torch.Tensor,
    charge_sign: torch.Tensor,
    d0: torch.Tensor,
    z0: torch.Tensor,
    magnetic_field_t: float,
    helix_radius_m: float,
    s_start: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trace a helix arc from track parameters.

    Args:
        phi:            azimuthal momentum angle at DCA [rad]
        eta:            pseudorapidity
        pt:             transverse momentum [GeV]
        charge_sign:    sign(charge), ±1
        d0:             transverse impact parameter [m]
        z0:             longitudinal impact parameter [m]
        magnetic_field_t: solenoid field strength [T]
        helix_radius_m: clip the path at this transverse radius [m]

    Returns:
        (x, y, z, y_linear) tensors, each shape (N_steps,), coordinates in metres.
    """
    b_field = float(magnetic_field_t)
    b_abs   = abs(b_field)

    if b_abs < 1e-6:
        omega = torch.zeros_like(pt)
        max_transverse_len = 2.0 * helix_radius_m
    else:
        # Angular velocity: omega = -(q * 0.3 * B) / pT
        omega = -charge_sign * ((0.3 * b_field) / pt.clamp_min(1e-6))
        curvature_radius = pt / (0.3 * b_abs)
        max_transverse_len = min(
            2.0 * helix_radius_m,
            float((4.0 * torch.pi * curvature_radius).item()),
        )
    max_transverse_len = max(max_transverse_len, 0.05)
    path_s = torch.linspace(s_start, max_transverse_len, 256, dtype=torch.float32, device=phi.device)

    x0 = -d0 * torch.sin(phi)
    y0 =  d0 * torch.cos(phi)
    y_linear = y0 + path_s * torch.sin(phi)

    if abs(float(omega.item())) < 1e-6:
        x = x0 + path_s * torch.cos(phi)
        y = y0 + path_s * torch.sin(phi)
    else:
        x = x0 + (torch.sin(phi + omega * path_s) - torch.sin(phi)) / omega
        y = y0 - (torch.cos(phi + omega * path_s) - torch.cos(phi)) / omega
    z = z0 + path_s * torch.sinh(eta)

    # Clip at the requested transverse radius
    radius_xy = torch.sqrt(x ** 2 + y ** 2)
    outside   = torch.nonzero(radius_xy >= helix_radius_m, as_tuple=False)
    if outside.numel() > 0:
        end_idx = max(int(outside[0].item()), 1)
        x, y, z, y_linear = x[:end_idx], y[:end_idx], z[:end_idx], y_linear[:end_idx]

    return x, y, z, y_linear


def _estimate_tracker_boundary(data: dict, batch_idx: int = 0) -> tuple[float, float]:
    """Return (R_max, Z_max) in metres from trkr hits, falling back to module defaults."""
    valid = data["trkr_valid"][batch_idx]
    if not valid.any():
        return _HELIX_CLIP_M, 2.5
    r_vals = data["trkr_pos.r"][batch_idx][valid]
    z_vals = data["trkr_pos.z"][batch_idx][valid].abs()
    return float(r_vals.max().item()), float(z_vals.max().item())


def _clip_helix_at_z(hx_t, hy_t, hz_t, z_max: float):
    """Trim helix tensors at the first point where |z| >= z_max."""
    outside = torch.nonzero(hz_t.abs() >= z_max, as_tuple=False)
    if outside.numel() > 0:
        end = max(int(outside[0].item()), 1)
        return hx_t[:end], hy_t[:end], hz_t[:end]
    return hx_t, hy_t, hz_t


# ── CLD detector geometry (all dimensions in metres) ──────────────────────────
_CLD_GEOM = {
    "beampipe_r": 0.015,
    "vtxd_r_in":  0.0175, "vtxd_r_out": 0.060, "vtxd_z": 0.125,
    "trkr_r_in":  0.075,  "trkr_r_out": 2.10,  "trkr_z": 2.20,
    "ecal_r_in":  2.15,   "ecal_r_out": 2.352, "ecal_z": 2.31,
    "ecal_ez_in": 2.31,   "ecal_ez_out": 2.512, "ecal_er_in": 0.20, "ecal_er_out": 2.09,
    "hcal_r_in":  2.40,   "hcal_r_out": 3.566, "hcal_z": 2.54,
    "hcal_ez_in": 2.54,   "hcal_ez_out": 3.71,  "hcal_er_in": 0.34, "hcal_er_out": 3.57,
    "coil_r_in":  3.60,   "coil_r_out": 3.90,  "coil_z": 4.00,
    "yoke_r_in":  3.90,   "yoke_r_out": 6.00,  "yoke_z": 5.30,
}

# Default axis limits and equal-aspect for known CLD views.
# Any axes_spec entry whose (x, y) pair matches gets these limits applied automatically.
_VIEW_AXIS_DEFAULTS: dict[tuple[str, str], dict] = {
    ("pos.x", "pos.y"): {"xlim": (-4.5,  4.5), "ylim": (-4.5, 4.5)},
    ("pos.z", "pos.y"): {"xlim": (-5.0,  5.0), "ylim": (-4.5, 4.5)},
}

_AXIS_LABEL_MAP = {
    "pos.x": r"$x$ [m]",
    "pos.y": r"$y$ [m]",
    "pos.z": r"$z$ [m]",
    "pos.r": r"$r$ [m]",
    "mom.x": r"$p_x$ [GeV]",
    "mom.y": r"$p_y$ [GeV]",
    "mom.z": r"$p_z$ [GeV]",
}

_GEO_COLORS = {
    "yoke": "#b2dfdb",  # pale teal
    "coil": "#b0c4de",  # light steel blue
    "hcal": "#ffe0b2",  # pale orange
    "ecal": "#c8e6c9",  # pale green
    "trkr": "#fce4ec",  # pale pink
    "vtxd": "#fff9c4",  # pale yellow
    "pipe": "#e0e0e0",  # light grey
}


def _annular_polygon_path(r_inscribed_in, r_inscribed_out, n=12):
    """Compound Path for an annular regular n-gon specified by inscribed (apothem) radii."""
    R_in  = r_inscribed_in  / np.cos(np.pi / n)
    R_out = r_inscribed_out / np.cos(np.pi / n)
    a0 = np.pi / n  # rotate so a flat face sits at the top
    angles = a0 + np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([R_out * np.cos(angles),        R_out * np.sin(angles)])
    inner = np.column_stack([R_in  * np.cos(angles[::-1]),  R_in  * np.sin(angles[::-1])])
    oc = np.vstack([outer, outer[0]])
    ic = np.vstack([inner, inner[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
    return Path(np.vstack([oc, ic]), codes + codes)


def _annular_circle_path(r_in, r_out, n=64):
    """Compound Path for an annular circle (approximated by an n-gon)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([r_out * np.cos(angles),        r_out * np.sin(angles)])
    inner = np.column_stack([r_in  * np.cos(angles[::-1]),  r_in  * np.sin(angles[::-1])])
    oc = np.vstack([outer, outer[0]])
    ic = np.vstack([inner, inner[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
    return Path(np.vstack([oc, ic]), codes + codes)


def _rect_path(x0, x1, y0, y1):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return Path(verts, codes)


def _draw_cld_geometry(ax, view_x, view_y, zorder=-10):
    """Overlay a rough CLD detector geometry on ax as a pastel background."""
    g = _CLD_GEOM
    C = _GEO_COLORS

    def _patch(path, color, fa=0.4, ea=0.8, lw=0.7):
        ax.add_patch(PathPatch(
            path,
            facecolor=mcolors.to_rgba(color, fa),
            edgecolor=mcolors.to_rgba(color, ea),
            linewidth=lw,
            zorder=zorder,
        ))

    if view_x == "pos.x" and view_y == "pos.y":
        # Transverse (XY) view — calorimeters are dodecagonal, tracker/vtxd circular
        _patch(_annular_polygon_path(g["yoke_r_in"],  g["yoke_r_out"],  12), C["yoke"])
        _patch(_annular_polygon_path(g["coil_r_in"],  g["coil_r_out"],  12), C["coil"])
        _patch(_annular_polygon_path(g["hcal_r_in"],  g["hcal_r_out"],  12), C["hcal"])
        _patch(_annular_polygon_path(g["ecal_r_in"],  g["ecal_r_out"],  12), C["ecal"])
        _patch(_annular_circle_path( g["trkr_r_in"],  g["trkr_r_out"]),      C["trkr"])
        _patch(_annular_circle_path( g["vtxd_r_in"],  g["vtxd_r_out"]),      C["vtxd"])
        _patch(_annular_circle_path( 0.001,           g["beampipe_r"]),       C["pipe"], fa=0.4)

    elif view_x == "pos.z" and view_y == "pos.y":
        # Longitudinal (ZY) view — draw annular bands (±y) for each barrel/endcap
        layers = [
            (-g["yoke_z"],      g["yoke_z"],      g["yoke_r_in"],  g["yoke_r_out"],  C["yoke"]),
            (-g["coil_z"],      g["coil_z"],      g["coil_r_in"],  g["coil_r_out"],  C["coil"]),
            (-g["hcal_z"],      g["hcal_z"],      g["hcal_r_in"],  g["hcal_r_out"],  C["hcal"]),
            ( g["hcal_ez_in"],  g["hcal_ez_out"], g["hcal_er_in"], g["hcal_er_out"], C["hcal"]),
            (-g["hcal_ez_out"], -g["hcal_ez_in"], g["hcal_er_in"], g["hcal_er_out"], C["hcal"]),
            (-g["ecal_z"],      g["ecal_z"],      g["ecal_r_in"],  g["ecal_r_out"],  C["ecal"]),
            ( g["ecal_ez_in"],  g["ecal_ez_out"], g["ecal_er_in"], g["ecal_er_out"], C["ecal"]),
            (-g["ecal_ez_out"], -g["ecal_ez_in"], g["ecal_er_in"], g["ecal_er_out"], C["ecal"]),
            (-g["trkr_z"],      g["trkr_z"],      g["trkr_r_in"],  g["trkr_r_out"],  C["trkr"]),
            (-g["vtxd_z"],      g["vtxd_z"],      g["vtxd_r_in"],  g["vtxd_r_out"],  C["vtxd"]),
        ]
        for z0, z1, r_in, r_out, color in layers:
            _patch(_rect_path(z0, z1,  r_in,  r_out), color)
            _patch(_rect_path(z0, z1, -r_out, -r_in), color)
        # Beam pipe: thin horizontal band
        _patch(_rect_path(-g["yoke_z"], g["yoke_z"], -g["beampipe_r"], g["beampipe_r"]),
               C["pipe"], fa=0.4)


def plot_cld_event(
    data,
    axes_spec,
    object_name,
    batch_idx=0,
    valid=True,
    mark_transparent=None,
    label_objects=False,
    gridspec_kw=None,
    particle_color=None,
    high_contrast=False,
    draw_helices=True,
    vtxd_inset=True,
    draw_geometry=True,
    draw_legend=True,
    usetex=False,
):
    plt.rcParams["text.usetex"] = usetex

    # Helices require momentum fields; support particle-style (mom.qopt/vtx) and pandora-style (charge/ref)
    _has_helix_fields = (
        f"{object_name}_mom.qopt" in data
        or (
            f"{object_name}_charge" in data
            and f"{object_name}_mom.x" in data
            and f"{object_name}_ref.x" in data
        )
    )
    if draw_helices and not _has_helix_fields:
        draw_helices = False

    # Setup the axes
    num_axes = len(axes_spec)

    # Per-axis view defaults (xlim/ylim for known CLD views)
    _axis_defaults = [_VIEW_AXIS_DEFAULTS.get((s["x"], s["y"]), {}) for s in axes_spec]

    # Auto-compute gridspec width ratios and figure width from xlim spans when not provided
    if gridspec_kw is None and all("xlim" in d for d in _axis_defaults):
        xlim_spans = [d["xlim"][1] - d["xlim"][0] for d in _axis_defaults]
        ylim_span  = _axis_defaults[0]["ylim"][1] - _axis_defaults[0]["ylim"][0]
        gridspec_kw = {"width_ratios": xlim_spans}
        fig_width = sum(xlim_spans) * 8.0 / ylim_span
    else:
        fig_width = 8.0 * num_axes

    fig, ax = plt.subplots(1, num_axes, gridspec_kw=gridspec_kw)
    fig.set_size_inches(fig_width, 8)

    ax = [ax] if num_axes == 1 else ax.flatten()

    # Setup the color cycler that will be used
    if particle_color is None:
        if high_contrast:
            cycler = [
                "#1f77b4",  # blue
                "#ff7f0e",  # orange
                "#2ca02c",  # green
                "#d62728",  # red
                "#9467bd",  # purple
                "#e377c2",  # pink
                "#17becf",  # cyan
                "#bcbd22",  # yellow-green
            ]
        else:
            colormap = plt.cm.tab20
            cycler = [colormap(i) for i in range(colormap.N)]
    else:
        cycler = [particle_color]

    # Used to define the different plotting styles for the differnt hits
    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]
    ecal_names = ["ecb", "ece", "ecal"]
    hcal_names = ["hcb", "hce", "hcal"]

    # Per-event tracker cylinder boundary for helix clipping
    if draw_helices and "trkr_pos.r" in data:
        R_tracker, Z_tracker = _estimate_tracker_boundary(data, batch_idx)
    else:
        R_tracker, Z_tracker = _HELIX_CLIP_M, 2.5

    # Tracks which (ax_idx, object_idx) pairs have had their helix drawn
    _helix_drawn: set[tuple[int, int]] = set()

    for ax_idx, ax_spec in enumerate(axes_spec):
        if draw_geometry:
            _draw_cld_geometry(ax[ax_idx], ax_spec["x"], ax_spec["y"])

        # Plot only the hits / subsystems specified for these axes
        for input_name in ax_spec["input_names"]:
            x = data[f"{input_name}_{ax_spec['x']}"][batch_idx]
            y = data[f"{input_name}_{ax_spec['y']}"][batch_idx]

            ax[ax_idx].scatter(x, y, alpha=0.25, s=1.0, color="black")

            num_object_slots = data[f"{object_name}_{input_name}_valid"][batch_idx].shape[-2]

            for object_idx in range(num_object_slots):
                # Plots invalid particle if valid set to be False
                if data[f"{object_name}_valid"][batch_idx][object_idx].item() == valid:
                    color = cycler[object_idx % len(cycler)]
                    mask = data[f"{object_name}_{input_name}_valid"][batch_idx][object_idx]

                    alpha = 1.0
                    linestyle = "-"

                    if mark_transparent is not None:  # noqa: SIM102
                        if not data[f"{object_name}_{mark_transparent}"][batch_idx][object_idx].item():
                            alpha = 0.5
                            linestyle = ":"

                    # Tracker hit
                    if input_name in sihit_names:
                        idx = torch.argsort(data[f"{input_name}_time"][batch_idx][mask], dim=-1)

                        if draw_helices:
                            # Scatter individual hits without connecting lines
                            if mask.any():
                                ax[ax_idx].scatter(x[mask][idx], y[mask][idx],
                                                   color="black", marker="+", alpha=alpha, s=14.0,
                                                   linewidths=0.9, zorder=3)
                                ax[ax_idx].scatter(x[mask][idx], y[mask][idx],
                                                   color=color, marker="+", alpha=alpha, s=8.0,
                                                   linewidths=0.5, zorder=4)

                            # Draw helix + production vertex once per (particle, axis)
                            if (ax_idx, object_idx) not in _helix_drawn:
                                _helix_drawn.add((ax_idx, object_idx))
                                if f"{object_name}_mom.qopt" in data:
                                    qopt = data[f"{object_name}_mom.qopt"][batch_idx][object_idx].item()
                                    q_sign = float(np.sign(qopt))
                                else:
                                    q_sign = float(np.sign(data[f"{object_name}_charge"][batch_idx][object_idx].item()))
                                if abs(q_sign) > 0.5:  # charged particle only
                                    if f"{object_name}_mom.phi" in data:
                                        phi_val = data[f"{object_name}_mom.phi"][batch_idx][object_idx].item()
                                    else:
                                        phi_val = float(np.arctan2(
                                            data[f"{object_name}_mom.y"][batch_idx][object_idx].item(),
                                            data[f"{object_name}_mom.x"][batch_idx][object_idx].item(),
                                        ))
                                    eta_val = data[f"{object_name}_mom.eta"][batch_idx][object_idx].item()
                                    pt_val = max(abs(data[f"{object_name}_mom.r"][batch_idx][object_idx].item()), 1e-6)
                                    if f"{object_name}_vtx.x" in data:
                                        vtx_x_m = data[f"{object_name}_vtx.x"][batch_idx][object_idx].item() * 1e-3
                                        vtx_y_m = data[f"{object_name}_vtx.y"][batch_idx][object_idx].item() * 1e-3
                                        vtx_z_m = data[f"{object_name}_vtx.z"][batch_idx][object_idx].item() * 1e-3
                                    else:
                                        vtx_x_m = data[f"{object_name}_ref.x"][batch_idx][object_idx].item() * 1e-3
                                        vtx_y_m = data[f"{object_name}_ref.y"][batch_idx][object_idx].item() * 1e-3
                                        vtx_z_m = data[f"{object_name}_ref.z"][batch_idx][object_idx].item() * 1e-3

                                    # Compute d0/z0 at DCA from production vertex
                                    R = pt_val / (0.3 * _B_FIELD_T)
                                    xc = vtx_x_m + q_sign * R * np.sin(phi_val)
                                    yc = vtx_y_m - q_sign * R * np.cos(phi_val)
                                    c = np.sqrt(xc**2 + yc**2)
                                    d0 = -q_sign * (c**2 - R**2) / (c + R) if (c + R) > 1e-12 else 0.0
                                    vtx_r = np.sqrt(vtx_x_m**2 + vtx_y_m**2)
                                    z0 = vtx_z_m - np.sinh(eta_val) * vtx_r

                                    hx_t, hy_t, hz_t, _ = _build_helix_path(
                                        torch.tensor(phi_val, dtype=torch.float32),
                                        torch.tensor(eta_val, dtype=torch.float32),
                                        torch.tensor(pt_val, dtype=torch.float32),
                                        torch.tensor(q_sign, dtype=torch.float32),
                                        torch.tensor(d0, dtype=torch.float32),
                                        torch.tensor(z0, dtype=torch.float32),
                                        _B_FIELD_T, R_tracker,
                                        s_start=vtx_r,
                                    )
                                    hx_t, hy_t, hz_t = _clip_helix_at_z(hx_t, hy_t, hz_t, Z_tracker)

                                    helix_coord = {"pos.x": hx_t, "pos.y": hy_t, "pos.z": hz_t}
                                    vtx_coord = {"pos.x": vtx_x_m, "pos.y": vtx_y_m, "pos.z": vtx_z_m}

                                    hx_plot = helix_coord.get(ax_spec["x"], hx_t).numpy()
                                    hy_plot = helix_coord.get(ax_spec["y"], hy_t).numpy()
                                    ax[ax_idx].plot(hx_plot, hy_plot, color="black",
                                                    linewidth=2.0, alpha=alpha,
                                                    linestyle=linestyle, zorder=1)
                                    ax[ax_idx].plot(hx_plot, hy_plot, color=color,
                                                    linewidth=1.0, alpha=alpha,
                                                    linestyle=linestyle, zorder=2)

                                    vtx_x_plot = vtx_coord.get(ax_spec["x"], vtx_x_m)
                                    vtx_y_plot = vtx_coord.get(ax_spec["y"], vtx_y_m)
                                    _star = MarkerStyle("*", transform=Affine2D().rotate_deg((object_idx * 137.508) % 72))
                                    ax[ax_idx].scatter([vtx_x_plot], [vtx_y_plot],
                                                       color=color, marker=_star, s=64,
                                                       linewidths=0.5, alpha=alpha, zorder=4,
                                                       edgecolors="black")
                        else:
                            ax[ax_idx].plot(x[mask][idx], y[mask][idx], color=color, marker="o", alpha=alpha, linewidth=1.0, ms=2.0, linestyle=linestyle, markeredgecolor="black", markeredgewidth=0.3)

                    # ECAL hit
                    elif input_name in ecal_names:
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="o", alpha=min(alpha, 0.5), s=3.0, edgecolors="black", linewidths=0.3)

                    # HCAL hit
                    elif input_name in hcal_names:
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="s", alpha=min(alpha, 0.5), s=4.0, edgecolors="black", linewidths=0.3)

                    # Muon hit
                    elif input_name == "muon":
                        ax[ax_idx].scatter(x[mask], y[mask], color="black", marker="x", alpha=alpha, s=10.0, linewidths=1.2, zorder=3)
                        ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="x", alpha=alpha, s=4.0, linewidths=0.7, zorder=4)

                    # Uncomment to leave a box denoting particle index for trkr hit
                    if label_objects and input_name in {"trkr", "hcal"} and mask.any():
                        idx = torch.argsort(data[f"{input_name}_time"][batch_idx][mask], dim=-1)
                        end_x = x[mask][idx][-1].item()
                        end_y = y[mask][idx][-1].item()

                        ax[ax_idx].text(
                            end_x,
                            end_y,
                            str(object_idx),
                            fontsize=5,
                            color="black",
                            ha="center",
                            va="center",
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "facecolor": "white",
                                "edgecolor": "black",
                                "linewidth": 0.5,
                                "alpha": 0.5,
                            },
                        )

            ax[ax_idx].set_xlabel(ax_spec.get("xlabel", _AXIS_LABEL_MAP.get(ax_spec["x"], ax_spec["x"])))
            ax[ax_idx].set_ylabel(ax_spec.get("ylabel", _AXIS_LABEL_MAP.get(ax_spec["y"], ax_spec["y"])))
            # ax[ax_idx].set_aspect("equal", "box")

    if vtxd_inset:
        vtxd_lim_xy = 0.08   # ±m for transverse directions
        vtxd_lim_z  = 0.20   # ±m for z

        coord_lims = {
            "pos.x": (-vtxd_lim_xy, vtxd_lim_xy),
            "pos.y": (-vtxd_lim_xy, vtxd_lim_xy),
            "pos.z": (-vtxd_lim_z,  vtxd_lim_z),
        }

        for ax_idx, ax_spec in enumerate(axes_spec):
            if "vtxd" not in ax_spec.get("input_names", []):
                continue

            axins = ax[ax_idx].inset_axes([0.02, 0.02, 0.38, 0.38])

            # Background: all vtxd hits
            vx_all = data[f"vtxd_{ax_spec['x']}"][batch_idx]
            vy_all = data[f"vtxd_{ax_spec['y']}"][batch_idx]
            axins.scatter(vx_all, vy_all, alpha=0.25, s=2.0, color="black", zorder=1)

            num_slots = data[f"{object_name}_vtxd_valid"][batch_idx].shape[-2]
            for object_idx in range(num_slots):
                if data[f"{object_name}_valid"][batch_idx][object_idx].item() != valid:
                    continue

                color = cycler[object_idx % len(cycler)]
                alpha = 1.0
                if mark_transparent is not None:
                    if not data[f"{object_name}_{mark_transparent}"][batch_idx][object_idx].item():
                        alpha = 0.5

                mask = data[f"{object_name}_vtxd_valid"][batch_idx][object_idx]
                if mask.any():
                    axins.scatter(vx_all[mask], vy_all[mask],
                                  color=color, marker="o", alpha=alpha, s=6.0,
                                  edgecolors="black", linewidths=0.3, zorder=3)

                if draw_helices:
                    if f"{object_name}_mom.qopt" in data:
                        qopt = data[f"{object_name}_mom.qopt"][batch_idx][object_idx].item()
                        q_sign = float(np.sign(qopt))
                    else:
                        q_sign = float(np.sign(data[f"{object_name}_charge"][batch_idx][object_idx].item()))
                    if abs(q_sign) > 0.5:
                        if f"{object_name}_mom.phi" in data:
                            phi_val = data[f"{object_name}_mom.phi"][batch_idx][object_idx].item()
                        else:
                            phi_val = float(np.arctan2(
                                data[f"{object_name}_mom.y"][batch_idx][object_idx].item(),
                                data[f"{object_name}_mom.x"][batch_idx][object_idx].item(),
                            ))
                        eta_val = data[f"{object_name}_mom.eta"][batch_idx][object_idx].item()
                        pt_val  = max(abs(data[f"{object_name}_mom.r"][batch_idx][object_idx].item()), 1e-6)
                        if f"{object_name}_vtx.x" in data:
                            vtx_x_m = data[f"{object_name}_vtx.x"][batch_idx][object_idx].item() * 1e-3
                            vtx_y_m = data[f"{object_name}_vtx.y"][batch_idx][object_idx].item() * 1e-3
                            vtx_z_m = data[f"{object_name}_vtx.z"][batch_idx][object_idx].item() * 1e-3
                        else:
                            vtx_x_m = data[f"{object_name}_ref.x"][batch_idx][object_idx].item() * 1e-3
                            vtx_y_m = data[f"{object_name}_ref.y"][batch_idx][object_idx].item() * 1e-3
                            vtx_z_m = data[f"{object_name}_ref.z"][batch_idx][object_idx].item() * 1e-3

                        R      = pt_val / (0.3 * _B_FIELD_T)
                        xc     = vtx_x_m + q_sign * R * np.sin(phi_val)
                        yc     = vtx_y_m - q_sign * R * np.cos(phi_val)
                        c      = np.sqrt(xc**2 + yc**2)
                        d0     = -q_sign * (c**2 - R**2) / (c + R) if (c + R) > 1e-12 else 0.0
                        vtx_r  = np.sqrt(vtx_x_m**2 + vtx_y_m**2)
                        z0     = vtx_z_m - np.sinh(eta_val) * vtx_r

                        hx_t, hy_t, hz_t, _ = _build_helix_path(
                            torch.tensor(phi_val, dtype=torch.float32),
                            torch.tensor(eta_val, dtype=torch.float32),
                            torch.tensor(pt_val,  dtype=torch.float32),
                            torch.tensor(q_sign,  dtype=torch.float32),
                            torch.tensor(d0,      dtype=torch.float32),
                            torch.tensor(z0,      dtype=torch.float32),
                            _B_FIELD_T, R_tracker,
                            s_start=vtx_r,
                        )
                        hx_t, hy_t, hz_t = _clip_helix_at_z(hx_t, hy_t, hz_t, Z_tracker)

                        helix_coord = {"pos.x": hx_t, "pos.y": hy_t, "pos.z": hz_t}
                        vtx_coord   = {"pos.x": vtx_x_m, "pos.y": vtx_y_m, "pos.z": vtx_z_m}

                        axins.plot(
                            helix_coord.get(ax_spec["x"], hx_t).numpy(),
                            helix_coord.get(ax_spec["y"], hy_t).numpy(),
                            color="black", linewidth=1.6, alpha=alpha, zorder=1,
                        )
                        axins.plot(
                            helix_coord.get(ax_spec["x"], hx_t).numpy(),
                            helix_coord.get(ax_spec["y"], hy_t).numpy(),
                            color=color, linewidth=0.8, alpha=alpha, zorder=2,
                        )
                        _star = MarkerStyle("*", transform=Affine2D().rotate_deg((object_idx * 137.508) % 72))
                        axins.scatter(
                            [vtx_coord.get(ax_spec["x"], vtx_x_m)],
                            [vtx_coord.get(ax_spec["y"], vtx_y_m)],
                            color=color, marker=_star, s=48, linewidths=0.5, alpha=alpha, zorder=4,
                            edgecolors="black",
                        )

            xlim = coord_lims.get(ax_spec["x"], (-vtxd_lim_xy, vtxd_lim_xy))
            ylim = coord_lims.get(ax_spec["y"], (-vtxd_lim_xy, vtxd_lim_xy))
            axins.set_xlim(*xlim)
            axins.set_ylim(*ylim)
            axins.set_aspect("equal")
            axins.xaxis.tick_top()
            axins.yaxis.tick_right()
            axins.xaxis.set_major_locator(plt.MaxNLocator(3))
            axins.yaxis.set_major_locator(plt.MaxNLocator(3))
            axins.tick_params(labelsize=5)
            axins.set_title("VTXD", fontsize=6, pad=2)
            ax[ax_idx].indicate_inset_zoom(axins, edgecolor="gray", alpha=0.7, linewidth=0.8)

    if draw_legend:
        _g = "dimgray"
        leg_handles = [
            mlines.Line2D([0], [0], color=_g, linewidth=1.2,
                          label=r"Track helix"),
            mlines.Line2D([0], [0], color=_g, linewidth=0, marker="+",
                          markersize=6, markeredgewidth=0.7,
                          label=r"Si hit"),
            mlines.Line2D([0], [0], color=_g, linewidth=0, marker="*",
                          markersize=8, markeredgewidth=0.5, markeredgecolor="black",
                          label=r"Production vertex"),
            mlines.Line2D([0], [0], color=_g, linewidth=0, marker="o",
                          markersize=4, alpha=0.5, markeredgecolor="black", markeredgewidth=0.3,
                          label=r"ECAL hit"),
            mlines.Line2D([0], [0], color=_g, linewidth=0, marker="s",
                          markersize=4, alpha=0.5, markeredgecolor="black", markeredgewidth=0.3,
                          label=r"HCAL hit"),
            mlines.Line2D([0], [0], color=_g, linewidth=0, marker="x",
                          markersize=5, markeredgewidth=0.7,
                          label=r"Muon hit"),
        ]
        ax[0].legend(handles=leg_handles, fontsize=7, loc="upper right",
                     framealpha=0.85, frameon=True)

        C = _GEO_COLORS
        geo_handles = [
            Patch(facecolor=mcolors.to_rgba(C["vtxd"], 0.6), edgecolor=mcolors.to_rgba(C["vtxd"], 0.9), label=r"VTXD"),
            Patch(facecolor=mcolors.to_rgba(C["trkr"], 0.6), edgecolor=mcolors.to_rgba(C["trkr"], 0.9), label=r"Tracker"),
            Patch(facecolor=mcolors.to_rgba(C["ecal"], 0.6), edgecolor=mcolors.to_rgba(C["ecal"], 0.9), label=r"ECAL"),
            Patch(facecolor=mcolors.to_rgba(C["hcal"], 0.6), edgecolor=mcolors.to_rgba(C["hcal"], 0.9), label=r"HCAL"),
            Patch(facecolor=mcolors.to_rgba(C["coil"], 0.6), edgecolor=mcolors.to_rgba(C["coil"], 0.9), label=r"Coil"),
            Patch(facecolor=mcolors.to_rgba(C["yoke"], 0.6), edgecolor=mcolors.to_rgba(C["yoke"], 0.9), label=r"Yoke"),
        ]
        ax[-1].legend(handles=geo_handles, fontsize=7, loc="upper right",
                      framealpha=0.85, frameon=True)

    for ax_i, defaults in zip(ax, _axis_defaults):
        if "xlim" in defaults:
            ax_i.set_xlim(*defaults["xlim"])
        if "ylim" in defaults:
            ax_i.set_ylim(*defaults["ylim"])
        if defaults:
            ax_i.set_aspect("equal")

    return fig


def _plot_matched_particle(
    ax, input_name, mc_idx, batch_idx, truth, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, mode="preds", object_name="particle"
):
    mask = truth[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]
    x_hits = inputs[f"{input_name}_{spec['x']}"][batch_idx][mask]
    y_hits = inputs[f"{input_name}_{spec['y']}"][batch_idx][mask]

    if mode == "preds":
        linestyle = "-"
        si_marker = "o"
        ecal_marker = "."
        hcal_marker = "s"
        alpha = 0.9
    else:
        linestyle = "--"
        si_marker = "x"
        ecal_marker = "x"
        hcal_marker = "x"
        alpha = 0.7

    arrow_scale = {"vtxd": 100, "trkr": 4}

    if input_name in sihit_names and mask.any():
        px_hits = truth[f"{object_name}_{input_name}_{spec['px']}"][batch_idx][mc_idx][mask]
        py_hits = truth[f"{object_name}_{input_name}_{spec['py']}"][batch_idx][mc_idx][mask]

        times = inputs[f"{input_name}_time"][batch_idx][mask]
        idx = torch.argsort(times, dim=-1)
        ax.plot(
            x_hits[idx],
            y_hits[idx],
            color=base_color,
            linestyle=linestyle,
            marker=si_marker,
            markersize=2.5,
            linewidth=1.0,
            alpha=alpha,
            label=f"mc_idx:{mc_idx}",
        )

        m = torch.sqrt(px_hits[idx] ** 2 + py_hits[idx] ** 2).clamp(min=1e-6)
        dx = px_hits[idx] / m
        dy = py_hits[idx] / m
        ax.quiver(
            x_hits[idx], y_hits[idx], dx, dy, angles="xy", scale_units="xy", scale=arrow_scale[input_name], color=base_color, width=0.002, alpha=alpha
        )

        end_x = x_hits[idx][-1].item()
        end_y = y_hits[idx][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )

    elif input_name in ecal_names:
        ax.scatter(x_hits, y_hits, color=base_color, marker=ecal_marker, alpha=alpha, s=3.0)

    elif input_name in hcal_names:
        ax.scatter(x_hits, y_hits, color=base_color, marker=hcal_marker, alpha=alpha, s=6.0)

    elif input_name == "muon":
        ax.scatter(x_hits, y_hits, color=base_color, marker="h", alpha=alpha, s=6.0)


def _plot_mismatched_particle(
    ax, input_name, mc_idx, batch_idx, truth, preds, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, object_name="particle"
):
    truth_mask = truth[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]
    pred_mask = preds[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]

    if truth_mask.any():
        t_x = inputs[f"{input_name}_{spec['x']}"][batch_idx][truth_mask]
        t_y = inputs[f"{input_name}_{spec['y']}"][batch_idx][truth_mask]

        linestyle = "-"
        alpha = 0.5
        if input_name in sihit_names:
            t_times = inputs[f"{input_name}_time"][batch_idx][truth_mask]
            tidx = torch.argsort(t_times, dim=-1)
            ax.plot(
                t_x[tidx],
                t_y[tidx],
                color=base_color,
                linestyle=linestyle,
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                alpha=alpha,
                label=f"mc_idx:{mc_idx}",
            )

        elif input_name in ecal_names:
            ax.scatter(t_x, t_y, color=base_color, marker=".", alpha=alpha, s=4.0)

        elif input_name in hcal_names:
            ax.scatter(t_x, t_y, color=base_color, marker="s", alpha=alpha, s=8.0)

        elif input_name == "muon":
            ax.scatter(t_x, t_y, color=base_color, marker="h", alpha=alpha, s=8.0)

    if pred_mask.any():
        p_x = inputs[f"{input_name}_{spec['x']}"][batch_idx][pred_mask]
        p_y = inputs[f"{input_name}_{spec['y']}"][batch_idx][pred_mask]

        linestyle = "--"
        marker = "x"
        alpha = 0.9
        if input_name in sihit_names:
            p_times = inputs[f"{input_name}_time"][batch_idx][pred_mask]
            pidx = torch.argsort(p_times, dim=-1)
            ax.plot(p_x[pidx], p_y[pidx], color=base_color, linestyle=linestyle, marker=marker, markersize=3.5, linewidth=1.0, alpha=alpha)

        elif input_name in ecal_names:
            ax.scatter(p_x, p_y, color=base_color, marker=marker, alpha=alpha, s=3.0)

        elif input_name in hcal_names:
            ax.scatter(p_x, p_y, color=base_color, marker=marker, alpha=alpha, s=6.0)

        elif input_name == "muon":
            ax.scatter(p_x, p_y, color=base_color, marker="H", alpha=alpha, s=6.0)

    if input_name == "trkr" and truth_mask.any():
        t_x = inputs[f"{input_name}_{spec['x']}"][batch_idx][truth_mask]
        t_y = inputs[f"{input_name}_{spec['y']}"][batch_idx][truth_mask]
        t_times = inputs[f"{input_name}_time"][batch_idx][truth_mask]
        tidx = torch.argsort(t_times, dim=-1)
        end_x = t_x[tidx][-1].item()
        end_y = t_y[tidx][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )


def plot_cld_event_match_vs_mismatch(inputs, truth, preds, axes_spec, object_name="particle", batch_idx=0):
    num_axes = len(axes_spec)
    fig, axes = plt.subplots(2, num_axes, squeeze=False)
    fig.set_size_inches(8 * num_axes, 16)

    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]

    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]
    ecal_names = ["ecb", "ece", "ecal"]
    hcal_names = ["hcb", "hce", "hcal"]

    for col_idx, spec in enumerate(axes_spec):
        ax_matched = axes[0, col_idx]
        ax_mismatch = axes[1, col_idx]

        for name in spec["input_names"]:
            x_all = inputs[f"{name}_{spec['x']}"][batch_idx]
            y_all = inputs[f"{name}_{spec['y']}"][batch_idx]
            ax_matched.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)
            ax_mismatch.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)

        input_names = spec["input_names"][0]
        truth_mask_all = truth[f"{object_name}_{input_names}_valid"][batch_idx]
        num_particles = truth_mask_all.shape[0]

        for mc_idx in range(num_particles):
            base_color = cycler[mc_idx % len(cycler)]

            for name in spec["input_names"]:
                truth_mask = truth[f"{object_name}_{name}_valid"][batch_idx][mc_idx]
                pred_mask = preds[f"{object_name}_{name}_valid"][batch_idx][mc_idx]
                if not (truth_mask.any() or pred_mask.any()):
                    continue

                if torch.equal(truth_mask, pred_mask):
                    _plot_matched_particle(
                        ax_matched, name, mc_idx, batch_idx, truth, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, object_name
                    )
                else:
                    _plot_mismatched_particle(
                        ax_mismatch, name, mc_idx, batch_idx, truth, preds, inputs, base_color, sihit_names, ecal_names, hcal_names, spec, object_name
                    )

        ax_matched.set_xlabel(spec["x"])
        ax_matched.set_ylabel(spec["y"])
        ax_matched.set_aspect("equal", "box")
        ax_matched.set_title("Matched Tracks")
        ax_matched.legend(fontsize=5, framealpha=0.5)

        ax_mismatch.set_xlabel(spec["x"])
        ax_mismatch.set_ylabel(spec["y"])
        ax_mismatch.set_aspect("equal", "box")
        ax_mismatch.set_title("Mismatched Tracks")
        ax_mismatch.legend(fontsize=5, framealpha=0.5)

    line_truth = mlines.Line2D([], [], color="gray", linestyle="-", marker="o", label="Truth")
    line_pred = mlines.Line2D([], [], color="gray", linestyle="--", marker="x", label="Prediction")
    fig.legend(handles=[line_truth, line_pred], loc="upper right")

    return fig


def _plot_matched_processed_particle(
    ax,
    input_name,
    mc_idx,
    batch_idx,
    orig_targets,
    post_targets,
    inputs_orig,
    inputs_post,
    base_color,
    sihit_names,
    ecal_names,
    hcal_names,
    post_idx,
    spec,
    object_name="particle",
):
    mask_pre = orig_targets[f"{object_name}_{input_name}_valid"][batch_idx][mc_idx]
    x_pre = inputs_orig[f"{input_name}_{spec['x']}"][batch_idx][mask_pre]
    y_pre = inputs_orig[f"{input_name}_{spec['y']}"][batch_idx][mask_pre]

    mask_post = post_targets[f"{object_name}_{input_name}_valid"][batch_idx][post_idx]
    x_post = inputs_post[f"{input_name}_{spec['x']}"][batch_idx][mask_post]
    y_post = inputs_post[f"{input_name}_{spec['y']}"][batch_idx][mask_post]

    linestyle_pre = "--"
    linestyle_post = "-"
    alpha_pre = 0.5
    alpha_post = 0.9
    marker_pre = "x"

    arrow_scale = {"vtxd": 100, "trkr": 4}

    if input_name in sihit_names and mask_pre.any():
        px_pre = orig_targets[f"{object_name}_{input_name}_{spec['px']}"][batch_idx][mc_idx][mask_pre]
        py_pre = orig_targets[f"{object_name}_{input_name}_{spec['py']}"][batch_idx][mc_idx][mask_pre]

        px_post = post_targets[f"{object_name}_{input_name}_{spec['px']}"][batch_idx][post_idx][mask_post]
        py_post = post_targets[f"{object_name}_{input_name}_{spec['py']}"][batch_idx][post_idx][mask_post]

        t_pre = inputs_orig[f"{input_name}_time"][batch_idx][mask_pre]
        idx_pre = torch.argsort(t_pre, dim=-1)
        ax.plot(
            x_pre[idx_pre],
            y_pre[idx_pre],
            color=base_color,
            linestyle=linestyle_pre,
            linewidth=0.5,
            marker=marker_pre,
            markersize=3.5,
            alpha=alpha_pre,
            label=f"mc_idx:{mc_idx} ({post_idx})",
        )
        end_x = x_pre[idx_pre][-1].item()
        end_y = y_pre[idx_pre][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linestyle": "--",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )
        m_pre = torch.sqrt(px_pre[idx_pre] ** 2 + py_pre[idx_pre] ** 2).clamp(min=1e-6)
        ax.quiver(
            x_pre[idx_pre],
            y_pre[idx_pre],
            (px_pre[idx_pre] / m_pre),
            (py_pre[idx_pre] / m_pre),
            angles="xy",
            scale_units="xy",
            scale=arrow_scale[input_name],
            color=base_color,
            width=0.002,
            alpha=alpha_pre,
        )

        t_post = inputs_post[f"{input_name}_time"][batch_idx][mask_post]
        idx_post = torch.argsort(t_post, dim=-1)
        ax.plot(
            x_post[idx_post],
            y_post[idx_post],
            color=base_color,
            linestyle=linestyle_post,
            linewidth=1.0,
            marker="o",
            markersize=2.5,
            alpha=alpha_post,
        )
        end_x = x_post[idx_post][-1].item()
        end_y = y_post[idx_post][-1].item()
        ax.text(
            end_x,
            end_y,
            str(mc_idx),
            fontsize=5,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.5,
            },
        )
        m_post = torch.sqrt(px_post[idx_post] ** 2 + py_post[idx_post] ** 2).clamp(min=1e-6)
        ax.quiver(
            x_post[idx_post],
            y_post[idx_post],
            (px_post[idx_post] / m_post),
            (py_post[idx_post] / m_post),
            angles="xy",
            scale_units="xy",
            scale=arrow_scale[input_name],
            color=base_color,
            width=0.002,
            alpha=alpha_post,
        )

    elif input_name in ecal_names:
        ax.scatter(x_pre, y_pre, color=base_color, marker=marker_pre, alpha=alpha_pre, s=4.0)
        ax.scatter(x_post, y_post, color=base_color, marker=".", alpha=alpha_post, s=3.0)

    elif input_name in hcal_names:
        ax.scatter(x_pre, y_pre, color=base_color, marker=marker_pre, alpha=alpha_pre, s=8.0)
        ax.scatter(x_post, y_post, color=base_color, marker="s", alpha=alpha_post, s=6.0)

    elif input_name == "muon":
        ax.scatter(x_pre, y_pre, color=base_color, marker="H", alpha=alpha_pre, s=8.0)
        ax.scatter(x_post, y_post, color=base_color, marker="h", alpha=alpha_post, s=6.0)


def plot_cld_event_pre_vs_post(inputs_orig, inputs_post, orig_targets, post_targets, axes_spec, object_name="particle", batch_idx=0):
    num_axes = len(axes_spec)
    fig, axes = plt.subplots(2, num_axes, squeeze=False)
    fig.set_size_inches(8 * num_axes, 16)

    colormap = plt.cm.tab20
    cycler = [colormap(i) for i in range(colormap.N)]

    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]
    ecal_names = ["ecb", "ece", "ecal"]
    hcal_names = ["hcb", "hce", "hcal"]

    fields = ["mom.x", "mom.y", "mom.z", "vtx.x", "vtx.y", "vtx.z"]
    orig_list = [orig_targets[f"{object_name}_{x}"][batch_idx] for x in fields]
    post_list = [post_targets[f"{object_name}_{x}"][batch_idx] for x in fields]
    orig_arr = torch.stack(orig_list, dim=1)
    post_arr = torch.stack(post_list, dim=1)

    eq_matrix = orig_arr.unsqueeze(1) == post_arr.unsqueeze(0)
    row_matched = eq_matrix.all(dim=2)
    matched_mask = row_matched.any(dim=1)

    n_orig = orig_arr.shape[0]
    match_idx = torch.full((n_orig,), -1)
    for i in range(n_orig):
        if matched_mask[i]:
            match_idx[i] = torch.nonzero(row_matched[i])[0]

    num_particles = orig_targets[f"{object_name}_valid"].shape[-1]

    for col_idx, spec in enumerate(axes_spec):
        ax_matched = axes[0, col_idx]
        ax_mismatch = axes[1, col_idx]

        for name in spec["input_names"]:
            x_all = inputs_orig[f"{name}_{spec['x']}"][batch_idx]
            y_all = inputs_orig[f"{name}_{spec['y']}"][batch_idx]
            ax_matched.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)
            ax_mismatch.scatter(x_all, y_all, color="black", alpha=0.5, s=1.0)

        for mc_i in range(num_particles):
            # if mc_i not in [18,41,63,120,143]:
            #     continue

            base_color = cycler[mc_i % len(cycler)]
            post_i = match_idx[mc_i].item()

            if post_i >= 0:
                for name in spec["input_names"]:
                    # Uncomment to plot only modified truth tracks for matched particles
                    # mask_pre = orig_targets[f"{object_name}_{name}_valid"][batch_idx][mc_i]
                    # x_pre = inputs_orig[f"{name}_{spec['x']}"][batch_idx][mask_pre]
                    # mask_post = post_targets[f"{object_name}_{name}_valid"][batch_idx][post_i]
                    # x_post = inputs_post[f"{name}_{spec['x']}"][batch_idx][mask_post]
                    # if x_pre.shape == x_post.shape:
                    #     continue

                    _plot_matched_processed_particle(
                        ax_matched,
                        name,
                        mc_i,
                        batch_idx,
                        orig_targets,
                        post_targets,
                        inputs_orig,
                        inputs_post,
                        base_color,
                        sihit_names,
                        ecal_names,
                        hcal_names,
                        post_i,
                        spec,
                        object_name,
                    )
                ax_matched.set_title("Matched Particles")
                ax_matched.set_xlabel(spec["x"])
                ax_matched.set_ylabel(spec["y"])
                ax_matched.set_aspect("equal", "box")
            else:
                for name in spec["input_names"]:
                    _plot_matched_particle(
                        ax_mismatch,
                        name,
                        mc_i,
                        batch_idx,
                        orig_targets,
                        inputs_orig,
                        base_color,
                        sihit_names,
                        ecal_names,
                        hcal_names,
                        spec,
                        mode="preprocess",
                        object_name=object_name,
                    )
                ax_mismatch.set_title("Dropped Particles")
                ax_mismatch.set_xlabel(spec["x"])
                ax_mismatch.set_ylabel(spec["y"])
                ax_mismatch.set_aspect("equal", "box")

        ax_matched.legend(fontsize=5, framealpha=0.5)
        ax_mismatch.legend(fontsize=5, framealpha=0.5)

    line_orig = mlines.Line2D([], [], color="gray", linestyle="--", marker="x", label="Original")
    line_pre = mlines.Line2D([], [], color="gray", linestyle="-", marker="o", label="After cuts")
    fig.legend(handles=[line_orig, line_pre], loc="upper right")

    return fig
