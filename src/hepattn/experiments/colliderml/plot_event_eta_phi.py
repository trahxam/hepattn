from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from hepattn.experiments.colliderml.data import ColliderMLDataset

# -------------------------
# Edit these settings only
# -------------------------
DATA_DIR = Path("/share/lustre/maxhart/data/colliderml/v1/pu200/")
OUTPUT_DIR = Path("src/hepattn/experiments/colliderml/plots")
TRACKER_OUTPUT_PATH = OUTPUT_DIR / "event_tracker_eta_phi.png"
CALO_OUTPUT_PATH = OUTPUT_DIR / "event_calo_eta_phi.png"
EVENT_INDEX = 0
EVENT_TYPE = "ttbar"
SHOW_FIGURE = False

PARTICLE_MIN_PT = 0.1
PARTICLE_MAX_ABS_ETA = 4.0
PARTICLE_INCLUDE_CHARGED = True
PARTICLE_INCLUDE_NEUTRAL = True
PARTICLE_HIT_CUTS = {
    "charged_hadron": {"min_num_sihit": 3, "min_num_hcal": 0},
    "neutral_hadron": {"min_num_hcal": 0},
    "electron": {"min_num_sihit": 3, "min_num_ecal": 0},
    "photon": {"min_num_ecal": 0},
    "muon": {"min_num_sihit": 3},
    "tau": {"min_num_sihit": 3},
}

FIGSIZE = (10, 6)
DPI = 600
MARKER_SIZE = 0.5
LINE_WIDTH_SCALE = 0.2


def build_dataset():
    return ColliderMLDataset(
        dirpath=str(DATA_DIR),
        num_events=-1,
        particle_min_pt=PARTICLE_MIN_PT,
        particle_max_abs_eta=PARTICLE_MAX_ABS_ETA,
        particle_hit_cuts=PARTICLE_HIT_CUTS,
        particle_include_charged=PARTICLE_INCLUDE_CHARGED,
        particle_include_neutral=PARTICLE_INCLUDE_NEUTRAL,
        return_calohits=True,
        return_tracks=False,
        event_type=EVENT_TYPE,
        build_calohit_associations=True,
    )


def make_hit_colors(hit_particle_index):
    noise_color = np.array([0.7, 0.7, 0.7, 0.6], dtype=np.float64)
    colors = np.tile(noise_color, (len(hit_particle_index), 1))
    particle_colors = {}

    on_valid_particle = hit_particle_index >= 0
    if on_valid_particle.any():
        unique_particle_indices, inverse = np.unique(hit_particle_index[on_valid_particle], return_inverse=True)
        denominator = max(int(inverse.max()), 1)
        mapped_colors = plt.colormaps["turbo"](inverse / denominator)
        colors[on_valid_particle] = mapped_colors
        particle_colors = {
            int(particle_idx): plt.colormaps["turbo"](i / denominator)
            for i, particle_idx in enumerate(unique_particle_indices)
        }

    return colors, particle_colors


def build_hit_particle_index(
    particle_valid,
    association_indptr,
    association_indices,
    num_hits,
    hit_valid,
    keep_first_association,
):
    num_particle_rows = max(int(association_indptr.shape[0]) - 1, 0)
    num_particles = min(int(particle_valid.shape[0]), num_particle_rows)

    hit_particle_index = np.full(num_hits, -1, dtype=np.int64)
    for particle_idx in range(num_particles):
        if not particle_valid[particle_idx]:
            continue

        start = int(association_indptr[particle_idx])
        end = int(association_indptr[particle_idx + 1])
        hit_indices = association_indices[start:end]
        hit_indices = hit_indices[(hit_indices >= 0) & (hit_indices < num_hits)]
        if hit_indices.size == 0:
            continue

        if keep_first_association:
            # Calo hits may be linked to multiple particles; keep first valid owner.
            unassigned = hit_particle_index[hit_indices] < 0
            hit_particle_index[hit_indices[unassigned]] = particle_idx
        else:
            hit_particle_index[hit_indices] = particle_idx

    hit_particle_index[~hit_valid] = -1
    return hit_particle_index, num_particles


def plot_hit_collection(
    sample_id,
    eta,
    phi,
    radial,
    hit_valid,
    particle_valid,
    association_indptr,
    association_indices,
    output_path,
    hit_label,
    keep_first_association,
    draw_connector_lines,
    scatter_specs,
):
    num_hits = int(eta.shape[0])
    hit_particle_index, num_particles = build_hit_particle_index(
        particle_valid=particle_valid,
        association_indptr=association_indptr,
        association_indices=association_indices,
        num_hits=num_hits,
        hit_valid=hit_valid,
        keep_first_association=keep_first_association,
    )

    colors, particle_colors = make_hit_colors(hit_particle_index)
    line_width = LINE_WIDTH_SCALE * np.sqrt(MARKER_SIZE)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    if draw_connector_lines:
        # Draw per-particle hit connectors in eta-phi, ordered radially.
        for particle_idx in range(num_particles):
            if not particle_valid[particle_idx]:
                continue

            start = int(association_indptr[particle_idx])
            end = int(association_indptr[particle_idx + 1])
            hit_indices = association_indices[start:end]
            hit_indices = hit_indices[(hit_indices >= 0) & (hit_indices < num_hits)]
            if hit_indices.size < 2:
                continue

            hit_indices = hit_indices[hit_valid[hit_indices]]
            if hit_indices.size < 2:
                continue

            order = np.argsort(radial[hit_indices])
            ordered_hits = hit_indices[order]
            particle_eta = eta[ordered_hits]
            particle_phi = phi[ordered_hits]
            particle_color = particle_colors.get(particle_idx)
            if particle_color is None:
                continue

            # Avoid long wrap-around segments across the phi boundary.
            breakpoints = np.where(np.abs(np.diff(particle_phi)) > np.pi)[0]
            segment_start = 0
            for breakpoint in breakpoints:
                segment_end = breakpoint + 1
                if segment_end - segment_start >= 2:
                    ax.plot(
                        particle_eta[segment_start:segment_end],
                        particle_phi[segment_start:segment_end],
                        color=particle_color,
                        linewidth=line_width,
                        alpha=0.85,
                        zorder=1,
                    )
                segment_start = segment_end

            if len(particle_eta) - segment_start >= 2:
                ax.plot(
                    particle_eta[segment_start:],
                    particle_phi[segment_start:],
                    color=particle_color,
                    linewidth=line_width,
                    alpha=0.85,
                    zorder=1,
                )

    legend_handles = []
    for mask, marker, label in scatter_specs:
        draw_mask = hit_valid & mask
        if not np.any(draw_mask):
            continue

        ax.scatter(
            eta[draw_mask],
            phi[draw_mask],
            c=colors[draw_mask],
            marker=marker,
            s=MARKER_SIZE,
            linewidths=0.0,
            alpha=0.85,
            rasterized=True,
            zorder=2,
        )
        legend_handles.append(Line2D([], [], marker=marker, color="black", linestyle="None", markersize=6, label=label))

    noise_hits = int(((hit_particle_index < 0) & hit_valid).sum())
    reco_particles = int(particle_valid[:num_particles].sum())
    num_valid_hits = int(hit_valid.sum())

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(
        f"ColliderML event {sample_id} | {hit_label}={num_valid_hits} | reco particles={reco_particles} | noise hits={noise_hits}"
    )
    ax.grid(alpha=0.2, linestyle="--")
    legend_handles.append(
        Line2D([], [], marker="o", color=(0.7, 0.7, 0.7, 0.8), linestyle="None", markersize=6, label="Noise hit")
    )
    ax.legend(handles=legend_handles, loc="best", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    print(f"Saved {hit_label} eta-phi event display to: {output_path.resolve()}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main():
    if not DATA_DIR.is_dir():
        msg = f"DATA_DIR does not exist: {DATA_DIR}"
        raise NotADirectoryError(msg)

    dataset = build_dataset()

    if EVENT_INDEX < 0 or EVENT_INDEX >= len(dataset):
        msg = f"EVENT_INDEX={EVENT_INDEX} is out of range for {len(dataset)} events."
        raise IndexError(msg)

    sample_id = int(dataset.sample_ids[EVENT_INDEX])
    inputs, targets = dataset.load_event(sample_id)

    particle_valid = targets["particle_valid"].to(dtype=bool).cpu().numpy()

    # Tracker hits (with connector lines)
    tracker_eta = inputs["sihit_eta"].cpu().numpy()
    tracker_phi = inputs["sihit_phi"].cpu().numpy()
    tracker_valid = inputs["sihit_valid"].to(dtype=bool).cpu().numpy()
    if "sihit_r" in inputs:
        tracker_radial = inputs["sihit_r"].cpu().numpy()
    else:
        tracker_x = inputs["sihit_x"].cpu().numpy()
        tracker_y = inputs["sihit_y"].cpu().numpy()
        tracker_radial = np.sqrt(tracker_x**2 + tracker_y**2)

    plot_hit_collection(
        sample_id=sample_id,
        eta=tracker_eta,
        phi=tracker_phi,
        radial=tracker_radial,
        hit_valid=tracker_valid,
        particle_valid=particle_valid,
        association_indptr=targets["particle_sihit_indptr"].cpu().numpy().astype(np.int64, copy=False),
        association_indices=targets["particle_sihit_indices"].cpu().numpy().astype(np.int64, copy=False),
        output_path=TRACKER_OUTPUT_PATH,
        hit_label="tracker hits",
        keep_first_association=False,
        draw_connector_lines=True,
        scatter_specs=[(np.ones_like(tracker_valid, dtype=bool), "o", "Tracker hits")],
    )

    # Calo hits (no connector lines)
    calo_x = inputs["calohit_x"].cpu().numpy()
    calo_y = inputs["calohit_y"].cpu().numpy()
    calo_z = inputs["calohit_z"].cpu().numpy()
    calo_radial = np.sqrt(calo_x**2 + calo_y**2)
    calo_s = np.sqrt(calo_radial**2 + calo_z**2)
    calo_cos_theta = np.clip(calo_z / np.clip(calo_s, 1e-12, None), -0.999999, 0.999999)
    calo_eta = np.arctanh(calo_cos_theta)
    calo_phi = np.arctan2(calo_y, calo_x)
    calo_valid = inputs["calohit_valid"].to(dtype=bool).cpu().numpy()
    calo_detector = inputs["calohit_detector"].cpu().numpy().astype(np.int64, copy=False)
    calo_ecal_mask = np.isin(calo_detector, ColliderMLDataset.CALO_ECAL_DETECTOR_IDS)
    calo_hcal_mask = np.isin(calo_detector, ColliderMLDataset.CALO_HCAL_DETECTOR_IDS) | ~calo_ecal_mask

    plot_hit_collection(
        sample_id=sample_id,
        eta=calo_eta,
        phi=calo_phi,
        radial=calo_radial,
        hit_valid=calo_valid,
        particle_valid=particle_valid,
        association_indptr=targets["particle_calohit_indptr"].cpu().numpy().astype(np.int64, copy=False),
        association_indices=targets["particle_calohit_indices"].cpu().numpy().astype(np.int64, copy=False),
        output_path=CALO_OUTPUT_PATH,
        hit_label="calo hits",
        keep_first_association=True,
        draw_connector_lines=False,
        scatter_specs=[
            (calo_ecal_mask, "o", "ECAL hits"),
            (calo_hcal_mask, "s", "HCAL hits"),
        ],
    )


if __name__ == "__main__":
    main()
