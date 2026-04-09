# ruff: noqa: E501

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from hepattn.experiments.itk.data.data import ITkDataset
from hepattn.utils.histogram import BinomialHistogram
from hepattn.utils.plotting import plot_hist_to_ax, setup_plotting
from hepattn.utils.stats import sigmoid

setup_plotting(fontsize=8)


def main():
    # Arguments for the evaluation
    config_path = Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/itk/configs/filtering_pixel.yaml")
    recon_max_eta = 4.0
    recon_min_pt = 1.0
    recon_min_num_pixel = 3.0

    # Now create the dataset
    config = yaml.safe_load(config_path.read_text())["data"]
    inputs = config["inputs"]

    targets = config["targets"]
    targets["particle"] = ["pt", "eta", "phi"]
    targets["particle_pixel"] = []

    dataset = ITkDataset(
        dirpath=config["test_dir"],
        inputs=inputs,
        targets=targets,
        num_events=-1,
        hit_regions=config["hit_regions"],
        particle_min_pt=recon_min_pt,
        particle_max_abs_eta=recon_max_eta,
        particle_min_num_hits={"pixel": recon_min_num_pixel},
        event_max_num_particles=10000,
    )

    hit_eval_path = "/share/rcifdata/maxhart/hepattn/logs/ITk_filtering_pixel_region135_3pix_eta4_900mev_PE_20250629-T133325/ckpts/epoch=099-val_loss=0.43550_test_eval.h5"

    dump_path = Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/itk/eval_dump")

    # Define bins and create BinomialHistograms for particle retention rate
    particle_bins = {"pt": np.linspace(0.5, 10.0, 32), "eta": np.linspace(-4, 4, 32), "phi": np.linspace(-np.pi, np.pi, 32)}
    retention_hists = {field: BinomialHistogram(bins) for field, bins in particle_bins.items()}

    num_hits_pre = []
    num_recon_parts_pre = []

    working_points = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    wp_num_hits_post = {wp: [] for wp in working_points}
    wp_num_recon_parts_post = {wp: [] for wp in working_points}

    hit = "pixel"
    nominal_wp = 0.025

    for idx in tqdm(range(10)):
        sample_id = dataset.sample_ids[idx]
        inputs, targets = dataset.load_event(sample_id)

        with h5py.File(hit_eval_path, "r") as hit_eval_file:
            hit_logits = hit_eval_file[f"{sample_id}/outputs/final/{hit}_filter/{hit}_logit"][0]

        event_name = dataset.sample_ids_to_event_names[sample_id]
        dump_data_df = pd.DataFrame({
            "hit_id": inputs["pixel_hit_id"],
            "logit": hit_logits,
        })
        dump_data_df.to_csv(dump_path / f"{event_name}.csv", index=False)

        # Particles which are deemed reconstructable pre-filter
        particle_recon_pre = targets["particle_valid"]
        particle_hit_valid_pre = targets["particle_pixel_valid"]

        particle_hit_valid_pre = particle_hit_valid_pre[particle_recon_pre]
        particle_recon_pre = particle_recon_pre[particle_recon_pre]

        num_hits_pre.append(particle_hit_valid_pre.shape[-1])
        num_recon_parts_pre.append(particle_recon_pre.sum())

        # Mark hits which pass the filter at nominal working point
        hit_filter_pred = sigmoid(hit_logits) >= nominal_wp
        particle_hit_valid_post = particle_hit_valid_pre & hit_filter_pred[None, :]
        particle_recon_post = particle_hit_valid_post.sum(-1) >= 3

        # Fill retention histograms
        for field, bins in particle_bins.items():
            particle_field = targets[f"particle_{field}"][targets["particle_valid"]]
            retention_hists[field].fill(particle_field, numerator=particle_recon_post, denominator=particle_recon_pre)

        # Calculate metrics for different working points
        for working_point in working_points:
            hit_filter_pred = sigmoid(hit_logits) >= working_point
            particle_hit_valid_post = particle_hit_valid_pre & hit_filter_pred[None, :]
            particle_recon_post = particle_hit_valid_post.sum(-1) >= 3

            wp_num_hits_post[working_point].append(hit_filter_pred.sum())
            wp_num_recon_parts_post[working_point].append(particle_recon_post.sum())

    plot_save_dir = Path(__file__).resolve().parent / Path("evalplots")

    # Working point scan plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    for wp in working_points:
        frac_recon_parts_retained = np.array(wp_num_recon_parts_post[wp]) / np.array(num_recon_parts_pre)
        num_hits_post = wp_num_hits_post[wp]
        ax.errorbar(
            np.mean(num_hits_post),
            np.mean(frac_recon_parts_retained),
            yerr=np.std(frac_recon_parts_retained),
            xerr=np.std(num_hits_post),
            label=wp,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Hits Retained")
    ax.set_ylabel("Fraction of Reconstructable Particles Retained")

    ax.axvline(np.mean(num_hits_pre), color="gray", linestyle="-", label="No Filtering")
    ax.axvline(np.mean(num_hits_pre) - np.std(num_hits_pre), color="gray", linestyle="--")
    ax.axvline(np.mean(num_hits_pre) + np.std(num_hits_pre), color="gray", linestyle="--")

    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    ax.set_xticks([5e4, 7.5e4, 1e5, 1.25e5, 1.5e5, 2.0e5, 2.5e5])

    fig.savefig(plot_save_dir / Path("wp_scan.png"))

    # Per-field retention plots
    field_labels = {"pt": "Truth Particle $p_T$ [GeV]", "eta": r"Truth Particle $\eta$", "phi": r"Truth Particle $\phi$"}

    for field, bins in particle_bins.items():
        eff, eff_errors = retention_hists[field].ratio()

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 2)

        plot_hist_to_ax(ax, eff, bins, eff_errors)

        ax.set_xlabel(field_labels[field])
        ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
        ax.set_ylim(0.99, 1.01)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")

        fig.savefig(plot_save_dir / Path(f"particle_recon_{field}.png"))


if __name__ == "__main__":
    main()
