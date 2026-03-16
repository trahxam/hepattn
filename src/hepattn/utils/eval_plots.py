from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from sklearn.metrics import confusion_matrix

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)


def as_numpy_dict(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: _to_numpy(values) for name, values in data.items()}


def make_mean_rmse_label(prefix: str | None, mean: float, rmse: float) -> str:
    label_prefix = "" if prefix is None else f"{prefix}\n"
    return label_prefix + rf"Mean = {mean:.2f}" + "\n" + rf"RMS =  {rmse:.2f}"


def calc_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def fold_overflow(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    lo = float(bins[0])
    hi = float(bins[-1])
    hi_inner = np.nextafter(hi, -np.inf)
    return np.clip(values, lo, hi_inner)


def prepare_position_eval_data(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    np_data = as_numpy_dict(data)

    if "particle_position" not in np_data:
        return np_data
    if "particle_position_preds" not in np_data:
        return np_data
    if "particle_position_pred_errs" not in np_data:
        return np_data

    true_pos = np_data["particle_position"]
    pred_pos = np_data["particle_position_preds"]
    pred_err = np_data["particle_position_pred_errs"]

    np_data["pos_res"] = pred_pos - true_pos
    np_data["err_res"] = pred_err - np_data["pos_res"]

    safe_pred_err = np.where(np.abs(pred_err) < 1e-12, np.nan, pred_err)
    np_data["pos_pull"] = np_data["pos_res"] / safe_pred_err

    return np_data


def plot_res_pull_1d_hists(data: Mapping[str, Any], title: str | None = None) -> Figure:
    """Produce 1D residual and pull histograms for x and y coordinates.

    Raises:
        KeyError: If the residual and pull inputs are missing.
    """
    prepared = prepare_position_eval_data(data)

    if "pos_res" not in prepared or "pos_pull" not in prepared:
        raise KeyError("Missing position tensors for residual/pull plotting.")

    coord_name_coord_idx = {"x": 0, "y": 1}
    hist_args = {"histtype": "step", "linewidth": 1.0, "density": True}

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(10, 4)

    if title is not None:
        fig.suptitle(title)

    for coord_name, coord_idx in coord_name_coord_idx.items():
        residuals = prepared["pos_res"][:, :, coord_idx].reshape(-1)
        residuals = residuals[np.isfinite(residuals)]
        residual_bins = np.linspace(-2, 2, 32)
        residual_label = make_mean_rmse_label(None, float(np.mean(residuals)), calc_rms(residuals))
        ax[coord_idx, 0].hist(
            fold_overflow(residuals, residual_bins),
            label=residual_label,
            bins=residual_bins,
            ec="cornflowerblue",
            **hist_args,
        )
        ax[coord_idx, 0].set_xlabel(rf"Residual ${coord_name}$", fontsize=10)
        ax[coord_idx, 0].set_ylabel("Density")
        ax[coord_idx, 0].legend(fontsize=6, loc="upper right")

        pulls = prepared["pos_pull"][:, :, coord_idx].reshape(-1)
        pulls = pulls[np.isfinite(pulls)]
        pull_bins = np.linspace(-3, 3, 32)
        pull_label = make_mean_rmse_label(None, float(np.mean(pulls)), calc_rms(pulls))
        normal_dist = np.exp(-0.5 * pull_bins * pull_bins) / np.sqrt(2 * np.pi)
        ax[coord_idx, 1].plot(pull_bins, normal_dist, color="lightgray", linestyle="--")
        ax[coord_idx, 1].hist(
            fold_overflow(pulls, pull_bins),
            label=pull_label,
            bins=pull_bins,
            ec="cornflowerblue",
            **hist_args,
        )
        ax[coord_idx, 1].set_xlabel(rf"Pull ${coord_name}$", fontsize=10)
        ax[coord_idx, 1].legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    return fig


def _extract_confusion_labels(np_data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if "true_multiplicity" in np_data and "pred_multiplicity" in np_data:
        true_counts = np.atleast_1d(np.squeeze(np_data["true_multiplicity"]).astype(np.int32))
        pred_counts = np.atleast_1d(np.squeeze(np_data["pred_multiplicity"]).astype(np.int32))
        return true_counts, pred_counts

    if "cluster_multiplicity" not in np_data:
        raise KeyError("Missing cluster_multiplicity or true_multiplicity for confusion-matrix plotting.")

    true_counts = np.atleast_1d(np.squeeze(np_data["cluster_multiplicity"]).astype(np.int32))
    if "cluster_multiplicity_preds" in np_data:
        pred_counts = np.atleast_1d(np.squeeze(np_data["cluster_multiplicity_preds"]).astype(np.int32))
        return true_counts, pred_counts

    if "cluster_multiplicity_logits" in np_data:
        pred_counts = np.atleast_1d((np.argmax(np_data["cluster_multiplicity_logits"], axis=-1) + 1).astype(np.int32))
        return true_counts, pred_counts

    raise KeyError("Missing cluster_multiplicity_preds, cluster_multiplicity_logits, or pred_multiplicity for confusion-matrix plotting.")


def _plot_confusion_matrix_values(conf_mat: np.ndarray, labels: np.ndarray, title: str | None = None) -> Figure:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(4, 4)

    if title is not None:
        fig.suptitle(title)

    image = ax.imshow(conf_mat, vmin=0.0, vmax=100.0)

    for row_idx in range(conf_mat.shape[0]):
        for col_idx in range(conf_mat.shape[1]):
            value = conf_mat[row_idx, col_idx]
            if np.isfinite(value) and value > 0:
                ax.text(
                    x=col_idx,
                    y=row_idx,
                    s=f"{value:.2f}",
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white" if value < 50.0 else "black",
                    path_effects=[patheffects.withStroke(linewidth=0.0, foreground="white")],
                )

    ax.set_xticks(np.arange(labels.size), [str(label) for label in labels])
    ax.set_yticks(np.arange(labels.size), [str(label) for label in labels])
    ax.set_xlabel("True Particle Count", fontsize=10)
    ax.set_ylabel("Predicted Particle Count", fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction (%)")

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    data: Mapping[str, Any],
    title: str | None = None,
    labels: Sequence[int] | np.ndarray | None = None,
) -> Figure:
    """Plot confusion matrix with x=true, y=predicted, normalized over true labels."""
    np_data = as_numpy_dict(data)
    true_counts, pred_counts = _extract_confusion_labels(np_data)

    label_values = (
        np.array(sorted(np.unique(np.concatenate((true_counts, pred_counts)))))
        if labels is None
        else np.asarray(labels, dtype=np.int32)
    )
    conf_mat = confusion_matrix(pred_counts, true_counts, labels=label_values, normalize="pred")
    conf_mat = 100.0 * conf_mat
    return _plot_confusion_matrix_values(conf_mat, label_values, title=title)


def plot_confusion_matrix_from_counts(
    counts_by_true_pred: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    title: str | None = None,
) -> Figure:
    """Plot confusion matrix from counts indexed by [true_label, pred_label].

    Raises:
        ValueError: If the provided count matrix does not match the label count.
    """
    label_values = np.asarray(labels, dtype=np.int32)
    count_values = np.asarray(counts_by_true_pred, dtype=float)

    expected_shape = (label_values.size, label_values.size)
    if count_values.shape != expected_shape:
        raise ValueError(f"Expected counts shape {expected_shape}, got {count_values.shape}.")

    true_totals = count_values.sum(axis=1)
    conf_mat = np.zeros((label_values.size, label_values.size), dtype=float)
    np.divide(
        count_values.T,
        true_totals[np.newaxis, :],
        out=conf_mat,
        where=true_totals[np.newaxis, :] > 0.0,
    )
    conf_mat = 100.0 * conf_mat
    return _plot_confusion_matrix_values(conf_mat, label_values, title=title)


def select_validation_metric_figures(val_data: Mapping[str, Any]) -> dict[str, Figure]:
    """Build only the validation metric figures needed for the active task(s)."""
    figures: dict[str, Figure] = {}
    np_data = as_numpy_dict(val_data)

    has_confusion_fields = (
        ("cluster_multiplicity" in np_data and "cluster_multiplicity_logits" in np_data)
        or ("cluster_multiplicity" in np_data and "cluster_multiplicity_preds" in np_data)
        or ("true_multiplicity" in np_data and "pred_multiplicity" in np_data)
    )
    if has_confusion_fields:
        figures["confusion_matrix"] = plot_confusion_matrix(np_data)

    has_position_fields = (
        "particle_position" in np_data
        and "particle_position_preds" in np_data
        and "particle_position_pred_errs" in np_data
    )
    if has_position_fields:
        figures["position_error_histograms"] = plot_res_pull_1d_hists(np_data)

    return figures
