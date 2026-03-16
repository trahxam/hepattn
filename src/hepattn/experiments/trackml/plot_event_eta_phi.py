from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from hepattn.experiments.trackml.data import TrackMLDataset

# -------------------------
# Edit these settings only
# -------------------------
DATA_DIR = Path("/share/rcifdata/maxhart/data/trackml/prepped/test/")
OUTPUT_PATH = Path("src/hepattn/experiments/trackml/plots/event_eta_phi.png")
OUTPUT_PATH_PHI_Z = Path("src/hepattn/experiments/trackml/plots/event_phi_z.png")
EVENT_INDEX = 0
SHOW_FIGURE = False

HIT_VOLUME_IDS = [7, 8, 9]
PIXEL_VOLUME_IDS = [7, 8, 9]
STRIP_VOLUME_IDS = [12, 13, 14, 16, 17, 18]

PARTICLE_MIN_PT = 0.1
PARTICLE_MAX_ABS_ETA = 4.0
PARTICLE_MIN_NUM_HITS = 3
EVENT_MAX_NUM_PARTICLES = 6000

FIGSIZE = (10, 6)
DPI = 300
MARKER_SIZE = 2.0
LINE_WIDTH_SCALE = 0.5


def build_dataset() -> TrackMLDataset:
    return TrackMLDataset(
        dirpath=str(DATA_DIR),
        inputs={"hit": ["eta", "phi", "z"]},
        targets={"hit": ["on_valid_particle"]},
        num_events=-1,
        hit_volume_ids=HIT_VOLUME_IDS,
        particle_min_pt=PARTICLE_MIN_PT,
        particle_max_abs_eta=PARTICLE_MAX_ABS_ETA,
        particle_min_num_hits=PARTICLE_MIN_NUM_HITS,
        event_max_num_particles=EVENT_MAX_NUM_PARTICLES,
    )


def make_hit_colors(hits):
    noise_color = np.array([0.7, 0.7, 0.7, 0.6], dtype=np.float64)
    colors = np.tile(noise_color, (len(hits), 1))
    particle_colors = {}

    on_valid_particle = hits["on_valid_particle"].to_numpy(dtype=bool)
    if on_valid_particle.any():
        particle_ids = hits["particle_id"].to_numpy(copy=True)
        unique_particle_ids, inverse = np.unique(particle_ids[on_valid_particle], return_inverse=True)
        denominator = max(int(inverse.max()), 1)
        mapped_colors = plt.colormaps["turbo"](inverse / denominator)
        colors[on_valid_particle] = mapped_colors
        particle_colors = {
            int(particle_id): plt.colormaps["turbo"](idx / denominator)
            for idx, particle_id in enumerate(unique_particle_ids)
        }

    return colors, particle_colors


def plot_event_display(
    *,
    hits,
    x_hits,
    x_field: str,
    x_label: str,
    phi,
    on_valid_particle,
    colors,
    particle_colors,
    pixel_mask,
    strip_mask,
    sample_id: int,
    reco_particles: int,
    noise_hits: int,
    output_path: Path,
) -> None:
    track_line_width = LINE_WIDTH_SCALE * np.sqrt(MARKER_SIZE)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    valid_hits = hits[on_valid_particle]
    for particle_id, particle_hits in valid_hits.groupby("particle_id", sort=False):
        if len(particle_hits) < 2:
            continue

        particle_hits = particle_hits.sort_values("r")
        particle_x = particle_hits[x_field].to_numpy(copy=True)
        particle_phi = particle_hits["phi"].to_numpy(copy=True)
        particle_color = particle_colors.get(int(particle_id))
        if particle_color is None:
            continue

        # Avoid drawing long connectors when the track crosses the phi wraparound.
        breakpoints = np.where(np.abs(np.diff(particle_phi)) > np.pi)[0]
        segment_start = 0
        for breakpoint in breakpoints:
            segment_end = breakpoint + 1
            if segment_end - segment_start >= 2:
                ax.plot(
                    particle_x[segment_start:segment_end],
                    particle_phi[segment_start:segment_end],
                    color=particle_color,
                    linewidth=track_line_width,
                    alpha=0.85,
                    zorder=1,
                )
            segment_start = segment_end

        if len(particle_x) - segment_start >= 2:
            ax.plot(
                particle_x[segment_start:],
                particle_phi[segment_start:],
                color=particle_color,
                linewidth=track_line_width,
                alpha=0.85,
                zorder=1,
            )

    ax.scatter(
        x_hits[pixel_mask],
        phi[pixel_mask],
        c=colors[pixel_mask],
        marker="o",
        s=MARKER_SIZE,
        linewidths=0.0,
        alpha=0.85,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        x_hits[strip_mask],
        phi[strip_mask],
        c=colors[strip_mask],
        marker="s",
        s=MARKER_SIZE,
        linewidths=0.0,
        alpha=0.85,
        rasterized=True,
        zorder=2,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(f"TrackML event {sample_id} | hits={len(hits)} | reco particles={reco_particles} | noise hits={noise_hits}")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="black", linestyle="None", markersize=6, label="Pixel hits"),
            Line2D([], [], marker="s", color="black", linestyle="None", markersize=6, label="Strip hits"),
            Line2D([], [], marker="o", color=(0.7, 0.7, 0.7, 0.8), linestyle="None", markersize=6, label="Noise hit"),
        ],
        loc="best",
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI)
    print(f"Saved {output_path.stem.replace('event_', '').replace('_', '-')} event display to: {output_path.resolve()}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    if not DATA_DIR.is_dir():
        msg = f"DATA_DIR does not exist: {DATA_DIR}"
        raise NotADirectoryError(msg)

    dataset = build_dataset()

    if EVENT_INDEX < 0 or EVENT_INDEX >= len(dataset):
        msg = f"EVENT_INDEX={EVENT_INDEX} is out of range for {len(dataset)} events."
        raise IndexError(msg)

    hits, _particles = dataset.load_event(EVENT_INDEX)
    sample_id = int(dataset.sample_ids[EVENT_INDEX])

    eta = hits["eta"].to_numpy(copy=True)
    phi = hits["phi"].to_numpy(copy=True)
    z = hits["z"].to_numpy(copy=True)
    volume_id = hits["volume_id"].to_numpy(copy=True)
    on_valid_particle = hits["on_valid_particle"].to_numpy(dtype=bool)
    colors, particle_colors = make_hit_colors(hits)

    pixel_mask = np.isin(volume_id, PIXEL_VOLUME_IDS)
    strip_mask = np.isin(volume_id, STRIP_VOLUME_IDS) | ~pixel_mask

    noise_hits = int((~on_valid_particle).sum())
    reco_particles = int(hits.loc[on_valid_particle, "particle_id"].nunique())

    plot_event_display(
        hits=hits,
        x_hits=eta,
        x_field="eta",
        x_label=r"$\eta$",
        phi=phi,
        on_valid_particle=on_valid_particle,
        colors=colors,
        particle_colors=particle_colors,
        pixel_mask=pixel_mask,
        strip_mask=strip_mask,
        sample_id=sample_id,
        reco_particles=reco_particles,
        noise_hits=noise_hits,
        output_path=OUTPUT_PATH,
    )

    plot_event_display(
        hits=hits,
        x_hits=z,
        x_field="z",
        x_label=r"$z$",
        phi=phi,
        on_valid_particle=on_valid_particle,
        colors=colors,
        particle_colors=particle_colors,
        pixel_mask=pixel_mask,
        strip_mask=strip_mask,
        sample_id=sample_id,
        reco_particles=reco_particles,
        noise_hits=noise_hits,
        output_path=OUTPUT_PATH_PHI_Z,
    )


if __name__ == "__main__":
    main()
