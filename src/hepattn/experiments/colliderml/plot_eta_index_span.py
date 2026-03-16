from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepattn.experiments.colliderml.data import ColliderMLDataset

# -------------------------
# Edit these settings only
# -------------------------
DATA_DIR = Path("/share/lustre/maxhart/data/colliderml/v1/pu200/")
OUTPUT_PATH = Path("src/hepattn/experiments/colliderml/plots/event_eta_index_span.png")
EVENT_INDEX = 0
EVENT_TYPE = "ttbar"
HIT_COLLECTION = "calo"  # "tracker" or "calo"

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

FIGSIZE = (12, 5)
DPI = 600
HIST_BINS = 100


def build_dataset():
    if HIT_COLLECTION not in {"tracker", "calo"}:
        msg = f"Unsupported HIT_COLLECTION={HIT_COLLECTION!r}. Use 'tracker' or 'calo'."
        raise ValueError(msg)

    return ColliderMLDataset(
        dirpath=str(DATA_DIR),
        num_events=-1,
        particle_min_pt=PARTICLE_MIN_PT,
        particle_max_abs_eta=PARTICLE_MAX_ABS_ETA,
        particle_hit_cuts=PARTICLE_HIT_CUTS,
        particle_include_charged=PARTICLE_INCLUDE_CHARGED,
        particle_include_neutral=PARTICLE_INCLUDE_NEUTRAL,
        return_calohits=(HIT_COLLECTION == "calo"),
        return_tracks=False,
        event_type=EVENT_TYPE,
        build_calohit_associations=True,
    )


def get_hit_eta_and_associations(inputs, targets):
    if HIT_COLLECTION == "tracker":
        eta = inputs["sihit_eta"].cpu().numpy()
        hit_valid = inputs["sihit_valid"].to(dtype=bool).cpu().numpy()
        association_indptr = targets["particle_sihit_indptr"].cpu().numpy().astype(np.int64, copy=False)
        association_indices = targets["particle_sihit_indices"].cpu().numpy().astype(np.int64, copy=False)
        label = "tracker hits"
    else:
        x = inputs["calohit_x"].cpu().numpy()
        y = inputs["calohit_y"].cpu().numpy()
        z = inputs["calohit_z"].cpu().numpy()
        s = np.sqrt(x**2 + y**2 + z**2)
        cos_theta = np.clip(z / np.clip(s, 1e-12, None), -0.999999, 0.999999)
        eta = np.arctanh(cos_theta)
        hit_valid = inputs["calohit_valid"].to(dtype=bool).cpu().numpy()
        association_indptr = targets["particle_calohit_indptr"].cpu().numpy().astype(np.int64, copy=False)
        association_indices = targets["particle_calohit_indices"].cpu().numpy().astype(np.int64, copy=False)
        label = "calo hits"

    return eta, hit_valid, association_indptr, association_indices, label


def compute_particle_eta_spans(eta, hit_valid, association_indptr, association_indices, particle_valid):
    valid_hit_indices = np.flatnonzero(hit_valid)
    if valid_hit_indices.size == 0:
        return np.zeros(0, dtype=np.int64), 0

    # Map each hit index to its rank after sorting hits by eta.
    eta_sorted_hit_indices = valid_hit_indices[np.argsort(eta[valid_hit_indices], kind="mergesort")]
    eta_rank = np.full(eta.shape[0], -1, dtype=np.int64)
    eta_rank[eta_sorted_hit_indices] = np.arange(eta_sorted_hit_indices.size, dtype=np.int64)

    num_particle_rows = max(int(association_indptr.shape[0]) - 1, 0)
    num_particles = min(int(particle_valid.shape[0]), num_particle_rows)

    spans = []
    for particle_idx in range(num_particles):
        if not particle_valid[particle_idx]:
            continue

        start = int(association_indptr[particle_idx])
        end = int(association_indptr[particle_idx + 1])
        hit_indices = association_indices[start:end]
        hit_indices = hit_indices[(hit_indices >= 0) & (hit_indices < eta.shape[0])]
        if hit_indices.size == 0:
            continue

        ranks = eta_rank[hit_indices]
        ranks = ranks[ranks >= 0]
        if ranks.size == 0:
            continue

        spans.append(int(ranks.max() - ranks.min()))

    return np.asarray(spans, dtype=np.int64), int(eta_sorted_hit_indices.size)


def plot_hist_and_cdf(spans, sample_id, hit_label, num_valid_hits):
    if spans.size == 0:
        msg = "No particle hit spans to plot for this event."
        raise RuntimeError(msg)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
    ax_hist, ax_cdf = axes

    max_span = int(spans.max())
    if max_span + 1 <= HIST_BINS:
        hist_bins = np.arange(-0.5, max_span + 1.5, 1.0)
    else:
        hist_bins = np.linspace(0.0, float(max_span), HIST_BINS + 1)

    ax_hist.hist(spans, bins=hist_bins, color="tab:blue", alpha=0.75)
    ax_hist.set_xlabel("Eta-ordered index span (max_idx - min_idx)")
    ax_hist.set_ylabel("Particle count")
    ax_hist.set_title("Span Histogram")
    ax_hist.grid(alpha=0.2, linestyle="--")

    sorted_spans = np.sort(spans)
    cdf = np.arange(1, sorted_spans.size + 1, dtype=np.float64) / sorted_spans.size
    ax_cdf.step(sorted_spans, cdf, where="post", color="tab:orange", linewidth=1.5)
    ax_cdf.set_xlabel("Eta-ordered index span (max_idx - min_idx)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.set_title("Span CDF")
    ax_cdf.grid(alpha=0.2, linestyle="--")

    fig.suptitle(
        f"ColliderML event {sample_id} | {hit_label}={num_valid_hits} | particles={spans.size}",
        fontsize=12,
    )
    return fig


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

    eta, hit_valid, association_indptr, association_indices, hit_label = get_hit_eta_and_associations(inputs, targets)
    particle_valid = targets["particle_valid"].to(dtype=bool).cpu().numpy()

    if HIT_COLLECTION == "calo":
        calo_detector = inputs["calohit_detector"].cpu().numpy().astype(np.int64, copy=False)
        ecal_mask = np.isin(calo_detector, ColliderMLDataset.CALO_ECAL_DETECTOR_IDS)
        hcal_mask = np.isin(calo_detector, ColliderMLDataset.CALO_HCAL_DETECTOR_IDS)
        subsets = [
            ("combined", hit_valid, "calo hits (combined)"),
            ("ecal", hit_valid & ecal_mask, "calo hits (ECAL only)"),
            ("hcal", hit_valid & hcal_mask, "calo hits (HCAL only)"),
        ]
    else:
        subsets = [("tracker", hit_valid, hit_label)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    num_saved = 0
    for subset_name, subset_valid_mask, subset_label in subsets:
        spans, num_valid_hits = compute_particle_eta_spans(
            eta=eta,
            hit_valid=subset_valid_mask,
            association_indptr=association_indptr,
            association_indices=association_indices,
            particle_valid=particle_valid,
        )

        if spans.size == 0:
            print(f"Skipping {subset_name}: no particle spans after subset selection.")
            continue

        fig = plot_hist_and_cdf(
            spans=spans,
            sample_id=sample_id,
            hit_label=subset_label,
            num_valid_hits=num_valid_hits,
        )
        output_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_{subset_name}{OUTPUT_PATH.suffix}")
        fig.savefig(output_path, dpi=DPI)
        plt.close(fig)
        num_saved += 1
        print(f"Saved eta index span histogram/CDF ({subset_name}) to: {output_path.resolve()}")

    if num_saved == 0:
        msg = "No non-empty hit subsets found for this event."
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
