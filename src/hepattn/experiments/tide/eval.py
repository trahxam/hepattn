from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from atlasify import atlasify
from scipy.optimize import linear_sum_assignment
from scipy.stats import binned_statistic
from tqdm import tqdm

from hepattn.utils.plotting import plot_hist_to_ax
from hepattn.utils.stats import bayesian_binomial_error

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
plt.rcParams["figure.constrained_layout.use"] = True

plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 18, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 18})


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    eval_path = Path(
        "/share/rcifdata/maxhart/hepattn/logs/TIDE_32trk_F32_fixed_nodrop_20250926-T150450/ckpts/epoch=003-train_loss=0.54949_prepped_eval.h5"
    )

    pred_names = ["sudo", "sisp", "reco", "pred"]
    colors = {
        "sudo": "black",
        "sisp": "red",
        "reco": "blue",
        "pred": "purple",
    }
    name_aliases = {
        "sudo": "Pseudo Tracks",
        "sisp": "SiSp Tracks",
        "reco": "Baseline Tracks",
        "pred": "MaskFormer Tracks",
    }

    true_qtys = [
        ("pt", r"Particle $p_\mathrm{T}$ [GeV]", "log", np.geomspace(2, 3.5e3, 32)),
        ("bhad_pt", r"Particle b-hadron $p_\mathrm{T}$ [GeV]", "log", np.geomspace(150, 5e3, 32)),
        ("eta", r"Particle $\eta$", "linear", np.linspace(-2.25, 2.25, 32)),
        ("phi", r"Particle $\phi$", "linear", np.linspace(-np.pi, np.pi, 32)),
        ("deta", r"Particle - RoI Axis $\Delta \eta$", "linear", np.linspace(-0.05, 0.05, 32)),
        ("dphi", r"Particle - RoI Axis $\Delta \phi$", "linear", np.linspace(-0.05, 0.05, 32)),
    ]

    pred_qtys = [
        ("phi", r"RoI $\phi$", "linear", np.linspace(-np.pi, np.pi, 32)),
        ("energy", r"RoI Energy [GeV]", "log", np.geomspace(150, 5e3, 32)),
    ]

    true_all_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in true_qtys} for pred_name in pred_names}
    true_eff_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in true_qtys} for pred_name in pred_names}

    pred_all_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in pred_qtys} for pred_name in pred_names}
    pred_pur_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in pred_qtys} for pred_name in pred_names}
    pred_dup_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in pred_qtys} for pred_name in pred_names}
    pred_fak_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in pred_qtys} for pred_name in pred_names}

    num_pix = {pred_name: np.zeros(shape=(16, 16)) for pred_name in pred_names}

    with h5py.File(eval_path) as file:
        for i, sample_id in tqdm(enumerate(file.keys())):
            if i == 10000:
                break

            targets = file[sample_id]["targets"]
            outputs = file[sample_id]["outputs"]["final"]

            true_valid = targets["sudo_valid"][0]
            true_pix_valid = targets["sudo_pix_valid"][0]
            true_sct_valid = targets["sudo_sct_valid"][0]

            true_pix_valid = true_pix_valid & true_valid[..., None]
            true_sct_valid = true_sct_valid & true_valid[..., None]

            pix_valid = targets["pix_valid"][0]

            preds = {}
            for pred_name in ["sudo", "sisp", "reco"]:
                preds[f"{pred_name}_valid"] = targets[f"{pred_name}_valid"][0]
                preds[f"{pred_name}_pix_valid"] = targets[f"{pred_name}_pix_valid"][0]
                preds[f"{pred_name}_sct_valid"] = targets[f"{pred_name}_sct_valid"][0]

            preds["pred_valid"] = sigmoid(outputs["pred_valid"]["pred_logit"][0]) >= 0.25
            preds["pred_pix_valid"] = sigmoid(outputs["pred_pix_assignment"]["pred_pix_logit"][0]) >= 0.5
            preds["pred_sct_valid"] = sigmoid(outputs["pred_sct_assignment"]["pred_sct_logit"][0]) >= 0.5

            preds["pred_pix_valid"] = preds["pred_pix_valid"] & preds["pred_valid"][..., None]
            preds["pred_sct_valid"] = preds["pred_sct_valid"] & preds["pred_valid"][..., None]

            for pred_name in pred_names:
                pred_valid = preds[f"{pred_name}_valid"]
                pred_pix_valid = preds[f"{pred_name}_pix_valid"]
                pred_sct_valid = preds[f"{pred_name}_sct_valid"]

                pix_tp = np.einsum("nc,mc->nm", true_pix_valid, pred_pix_valid)
                pix_fp = np.einsum("nc,mc->nm", true_pix_valid, (~pred_pix_valid))
                pix_fn = np.einsum("nc,mc->nm", (~true_pix_valid), pred_pix_valid)

                sct_tp = np.einsum("nc,mc->nm", true_sct_valid, pred_sct_valid)
                sct_fp = np.einsum("nc,mc->nm", true_sct_valid, (~pred_sct_valid))
                sct_fn = np.einsum("nc,mc->nm", (~true_sct_valid), pred_sct_valid)

                pix_true_num = true_pix_valid.sum(-1)
                pix_pred_num = pred_pix_valid.sum(-1)

                sct_true_num = true_sct_valid.sum(-1)
                sct_pred_num = pred_sct_valid.sum(-1)  # noqa:

                eps = 1e-6
                metric = "tmp"
                score_threshold = 0.75

                matching_score = (2 * pix_tp + sct_tp) / (2 * (pix_tp + pix_fp + pix_fn) + sct_tp + sct_fp + sct_fn + eps)

                # Using the masks we calculate the desired score - the eps term prevents any division by zero
                if metric == "tmp":
                    scores = (2 * pix_tp + sct_tp) / (2 * (pix_tp + pix_fp) + sct_tp + sct_fp + eps)
                elif metric == "iou":
                    scores = (2 * pix_tp + sct_tp) / (2 * (pix_tp + pix_fp + pix_fn) + sct_tp + sct_fp + sct_fn + eps)
                elif metric == "dice":
                    scores = 2 * (pix_tp + sct_tp) / (2 * pix_tp + pix_fp + pix_fn + 2 * sct_tp + sct_fp + sct_fn + eps)

                scores[~true_valid, :] = 0.0
                scores[:, ~pred_valid] = 0.0

                true_idx, pred_idx = linear_sum_assignment(matching_score, maximize=True)

                pred_valid = pred_valid[pred_idx]
                pred_pix_valid = pred_pix_valid[pred_idx]
                pred_sct_valid = pred_sct_valid[pred_idx]

                # The scores under the optimal true-pred pair assignment
                paired_scores = scores[true_idx, pred_idx]

                # Also keep the scores between every true-pred pair
                scores = scores[:, pred_idx]

                paired_matches = paired_scores >= score_threshold
                matches = scores >= score_threshold

                # A true particle is efficient its assigned/paired pred is a match
                true_is_eff = paired_matches[true_valid]

                # A pred object is pure its paired particle is a match
                pred_is_pur = paired_matches[pred_valid]

                # A pred object is a duplicate if it matches some particle, but not with its paired particle
                pred_is_dup = (~paired_matches & matches.any(-2))[pred_valid]

                # Pred object is fake if it no particles match with it
                pred_is_fak = (~matches.any(-2))[pred_valid]

                for qty_name, _, _, bins in true_qtys:
                    qty = targets[f"sudo_{qty_name}"][0][true_valid]

                    num_all, _, _ = binned_statistic(qty, true_is_eff, statistic="count", bins=bins)
                    num_eff, _, _ = binned_statistic(qty, true_is_eff, statistic="sum", bins=bins)

                    true_all_bins[pred_name][qty_name] += num_all
                    true_eff_bins[pred_name][qty_name] += num_eff

                for qty_name, _, _, bins in pred_qtys:
                    qty = np.full_like(pred_is_pur.astype(np.float32), targets[f"roi_{qty_name}"][0])

                    # Hack to convert to GeV
                    if qty_name == "energy":
                        qty = qty / 1000.0

                    if len(qty) == 0:
                        continue

                    num_all, _, _ = binned_statistic(qty, pred_is_pur, statistic="count", bins=bins)
                    num_pur, _, _ = binned_statistic(qty, pred_is_pur, statistic="sum", bins=bins)
                    num_dup, _, _ = binned_statistic(qty, pred_is_dup, statistic="sum", bins=bins)
                    num_fak, _, _ = binned_statistic(qty, pred_is_fak, statistic="sum", bins=bins)

                    pred_all_bins[pred_name][qty_name] += num_all
                    pred_pur_bins[pred_name][qty_name] += num_pur
                    pred_dup_bins[pred_name][qty_name] += num_dup
                    pred_fak_bins[pred_name][qty_name] += num_fak

                pix_true_num = true_pix_valid.sum(-2)[pix_valid]
                pix_pred_num = pred_pix_valid.sum(-2)[pix_valid]

                num_pix[pred_name] += np.histogram2d(
                    pix_true_num,
                    pix_pred_num,
                    bins=np.arange(17) - 0.5,
                )[0]

    legend_font_size = 14
    sub_font_size = 16
    metric_aliases = {"tmp": "TMP", "iou": "IoU"}
    sub_label = (
        r"$\sqrt{s} = 13\,\mathrm{TeV},\; Z'\!\rightarrow q\bar{q}$" + "\n" + rf"$\mathrm{{{metric_aliases[metric]}}} \geq {score_threshold:.2f}$"
    )

    for qty_name, qty_label, scale, bins in true_qtys:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        for pred_name in pred_names:
            k = true_eff_bins[pred_name][qty_name]
            n = true_all_bins[pred_name][qty_name]
            label = name_aliases[pred_name] if pred_name != "sudo" else None
            plot_hist_to_ax(ax, k / n, bins, value_errors=bayesian_binomial_error(k, n), label=label, color=colors[pred_name])

        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel("Particle Efficiency")
        ax.legend(fontsize=legend_font_size)
        atlasify("Simulation Internal", sub_label, sub_font_size=sub_font_size)

        fig.tight_layout()
        fig.savefig(f"/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/{qty_name}_eff.png")

    for qty_name, qty_label, scale, bins in pred_qtys:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        for pred_name in pred_names:
            n = pred_all_bins[pred_name][qty_name]
            k = pred_pur_bins[pred_name][qty_name]
            label = name_aliases[pred_name] if pred_name != "sudo" else None
            plot_hist_to_ax(ax, k / n, bins, value_errors=bayesian_binomial_error(k, n), label=label, color=colors[pred_name])

        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel("Track Purity")
        ax.legend(fontsize=legend_font_size)
        atlasify("Simulation Internal", sub_label, sub_font_size=sub_font_size)

        fig.tight_layout()
        fig.savefig(f"/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/{qty_name}_pur.png")

    for qty_name, qty_label, scale, bins in pred_qtys:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        for pred_name in pred_names:
            n = pred_all_bins[pred_name][qty_name]
            k = pred_fak_bins[pred_name][qty_name]
            label = name_aliases[pred_name] if pred_name != "sudo" else None
            plot_hist_to_ax(ax, k / n, bins, value_errors=bayesian_binomial_error(k, n), label=label, color=colors[pred_name])

        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel("Track Fake Rate")
        ax.legend(fontsize=legend_font_size)
        atlasify("Simulation Internal", sub_label, sub_font_size=sub_font_size)

        fig.tight_layout()
        fig.savefig(f"/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/{qty_name}_fak.png")

    for qty_name, qty_label, scale, bins in pred_qtys:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        for pred_name in pred_names:
            n = pred_all_bins[pred_name][qty_name]
            k = pred_dup_bins[pred_name][qty_name]
            label = name_aliases[pred_name] if pred_name != "sudo" else None
            plot_hist_to_ax(ax, k / n, bins, value_errors=bayesian_binomial_error(k, n), label=label, color=colors[pred_name])

        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel("Track Duplicate Rate")
        ax.legend(fontsize=legend_font_size)
        atlasify("Simulation Internal", sub_label, sub_font_size=sub_font_size)

        fig.tight_layout()
        fig.savefig(f"/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/{qty_name}_dup.png")

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3.2), layout="constrained")

        vmin, vmax = 0.0, 1.0
        im = None

        for i, pred_name in enumerate(["sisp", "reco", "pred"]):
            hist_count = num_pix[pred_name]
            denom = num_pix[pred_name].sum(-1, keepdims=True)
            denom = np.where(denom == 0, 1, denom)  # avoid divide-by-zero
            hist_frac = (hist_count / denom).T

            im = ax[i].imshow(hist_frac, vmin=vmin, vmax=vmax)

            # annotate cells > 0.1
            nrows, ncols = hist_frac.shape
            for r in range(nrows):
                for c in range(ncols):
                    val = float(hist_frac[r, c])
                    if val > 0.01:
                        ax[i].text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=4, color="black")

            ax[i].set_xlabel("Number of Particles\non Pixel Hit", fontsize=10)

            # ax[i].set_title(name_aliases[pred_name], fontsize=12)
            ax[i].set_xticks(np.arange(ncols))
            ax[i].set_yticks(np.arange(nrows))
            ax[i].set_xticklabels(list(range(ncols)), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
            ax[i].set_yticklabels(list(range(nrows)), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
            ax[i].text(0.01, 0.01, name_aliases[pred_name], transform=ax[i].transAxes, color="white", ha="left", va="bottom", fontsize=10)

        ax[0].set_ylabel("Number of Tracks\non Pixel Hit", fontsize=10)

        plt.rcParams["font.family"] = "Nimbus Sans"

        fig.text(0.05, 0.95, "ATLAS", fontweight="bold", style="italic", ha="left", va="top", fontsize=12)
        fig.text(0.135, 0.95, "Simulation Internal", ha="left", va="top", fontsize=12)
        fig.text(0.92, 0.95, r"$\sqrt{s} = 13\,\mathrm{TeV},\; Z'\!\rightarrow q\bar{q}$", ha="right", va="top", fontsize=10)

        # one shared colorbar on the rightmost axes
        cbar = fig.colorbar(im, ax=ax[-1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Fraction Correct", fontsize=12)

        fig.savefig("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/pixel_sharing.png")


if __name__ == "__main__":
    main()
