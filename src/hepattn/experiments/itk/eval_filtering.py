from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import binned_statistic
from tqdm import tqdm

from hepattn.experiments.itk.data import ITkDataModule

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["figure.constrained_layout.use"] = True


def plot_hist_to_ax(ax, values, bins, value_errors=None, color="cornflowerblue"):
    for bin_idx in range(len(bins) - 1):
        px = np.array([bins[bin_idx], bins[bin_idx + 1]])
        py = np.array([values[bin_idx], values[bin_idx]])

        ax.plot(px, py, color=color, linewidth=1.0)

        if value_errors is not None:
            pe = np.array([value_errors[bin_idx], value_errors[bin_idx]])
            ax.fill_between(px, py - pe, py + pe, color=color, alpha=0.1, ec="none")


def frequentist_binomial_error(k, n):
    return np.sqrt((k / n) * (1 - (k / n)) / n)


def bayesian_binomial_error(k, n):
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) / (n + 2)) ** 2)


def main():
    config_path = Path("/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/itk/configs/filtering_pixel.yaml")
    config = yaml.safe_load(config_path.read_text())["data"]
    config["num_workers"] = 0

    datamodule = ITkDataModule(**config)
    datamodule.setup(stage="test")

    dataloader = datamodule.test_dataloader()
    dataset = dataloader.dataset

    dataset.append_hit_eval_output = True
    dataset.inputs["pixel"].append("filter_logit")
    dataset.targets["particle"] = ["pt", "eta", "phi"]
    dataset.targets["particle_pixel"] = []

    # Define bins for particle retention rate under the nominal working point
    particle_bins = {"pt": np.linspace(0.5, 10.0, 32), "eta": np.linspace(-4, 4, 32), "phi": np.linspace(-np.pi, np.pi, 32)}
    particle_pre_counts = {field: np.zeros(len(particle_bins[field]) - 1) for field in particle_bins}
    particle_post_counts = {field: np.zeros(len(particle_bins[field]) - 1) for field in particle_bins}

    num_hits_pre = []
    num_recon_parts_pre = []

    working_points = [0.01, 0.05, 0.075, 0.1, 0.125]
    wp_num_hits_post = {wp: [] for wp in working_points}
    wp_num_recon_parts_post = {wp: [] for wp in working_points}

    # Iterate over the events
    for i in tqdm(range(10)):
        # Load the data from the event
        inputs, targets = dataset[i]

        # Particles which are deemed reconstructable pre-filter
        particle_recon_pre = targets["particle_valid"][0]
        particle_hit_valid_pre = targets["particle_pixel_valid"][0]

        # Drop any padding
        particle_hit_valid_pre = particle_hit_valid_pre[particle_recon_pre]
        particle_recon_pre = particle_recon_pre[particle_recon_pre]

        # Get the hit filter logits
        hit_logits = inputs["pixel_filter_logit"][0]

        # Record number of reconstructable particles and hits before filtering
        num_hits_pre.append(particle_hit_valid_pre.shape[-1])
        num_recon_parts_pre.append(particle_recon_pre.sum())

        # Mark hits which pass the filter
        hit_filter_pred = hit_logits.sigmoid() >= 0.05

        # The post filter mask is just the pre filter mask, but with filtered hits removed
        particle_hit_valid_post = particle_hit_valid_pre & hit_filter_pred[None, :]
        particle_recon_post = particle_hit_valid_post.sum(-1) >= 3

        # Fill the particle histograms
        for field, bins in particle_bins.items():
            particle_field = targets[f"particle_{field}"][0][targets["particle_valid"][0]]

            pre_count, _, _ = binned_statistic(particle_field, particle_recon_post.float(), statistic="sum", bins=bins)
            post_count, _, _ = binned_statistic(particle_field, particle_recon_post.float(), statistic="count", bins=bins)

            particle_pre_counts[field] += pre_count
            particle_post_counts[field] += post_count

        # Now calculate metrics for different working points
        for working_point in working_points:
            # Mark hits which pass the filter under this new working point
            hit_filter_pred = inputs["pixel_filter_logit"][0].sigmoid() >= working_point
            particle_hit_valid_post = particle_hit_valid_pre & hit_filter_pred[None, :]
            particle_recon_post = particle_hit_valid_post.sum(-1) >= 3

            wp_num_hits_post[working_point].append(hit_filter_pred.sum())
            wp_num_recon_parts_post[working_point].append(particle_recon_post.sum())

    plot_save_dir = Path(__file__).resolve().parent / Path("evalplots")

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    for wp in working_points:
        frac_recon_parts_retained = np.array(wp_num_recon_parts_post[wp]) / np.array(num_recon_parts_pre)
        ax.scatter(np.array(wp_num_hits_post[wp]), frac_recon_parts_retained, label=wp, alpha=0.5)

    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Number of Hits Retained")
    ax.set_ylabel("Fraction of Reconstructable Particles Retained")

    fig.savefig(plot_save_dir / Path("wp_scan.png"))

    pre_count = particle_pre_counts["pt"]
    post_count = particle_post_counts["pt"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["pt"], eff_errors)

    ax.set_xlabel("Truth Particle $p_T$ [GeV]")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.97, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_pt.png"))

    pre_count = particle_pre_counts["eta"]
    post_count = particle_post_counts["eta"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["eta"], eff_errors)

    ax.set_xlabel(r"Truth Particle $\eta$")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.96, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_eta.png"))

    pre_count = particle_pre_counts["phi"]
    post_count = particle_post_counts["phi"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["phi"], eff_errors)

    ax.set_xlabel(r"Truth Particle $\phi$")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.97, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_phi.png"))


if __name__ == "__main__":
    main()
