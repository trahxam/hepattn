import matplotlib.pyplot as plt
import numpy as np

from hepattn.utils.eval_plots import (
    plot_confusion_matrix_from_counts,
    select_validation_metric_figures,
)


def test_plot_confusion_matrix_from_counts_normalizes_over_true_labels():
    counts_by_true_pred = np.array(
        [
            [0, 0, 0],
            [0, 2, 1],
            [0, 1, 3],
        ],
        dtype=float,
    )

    fig = plot_confusion_matrix_from_counts(counts_by_true_pred=counts_by_true_pred, labels=[0, 1, 2])
    ax = fig.axes[0]
    text_by_position = {tuple(map(int, text.get_position())): text.get_text() for text in ax.texts}

    assert ax.get_xlabel() == "True Particle Count"
    assert ax.get_ylabel() == "Predicted Particle Count"
    assert text_by_position[1, 1] == "66.67"
    assert text_by_position[1, 2] == "33.33"
    assert text_by_position[2, 1] == "25.00"
    assert text_by_position[2, 2] == "75.00"

    plt.close(fig)


def test_select_validation_metric_figures_accepts_explicit_multiplicity_arrays():
    val_data = {
        "true_multiplicity": np.array([1, 1, 2, 2], dtype=np.int32),
        "pred_multiplicity": np.array([1, 2, 2, 2], dtype=np.int32),
    }

    figures = select_validation_metric_figures(val_data)

    assert set(figures) == {"confusion_matrix"}
    fig = figures["confusion_matrix"]
    ax = fig.axes[0]
    assert ax.get_xlabel() == "True Particle Count"
    assert ax.get_ylabel() == "Predicted Particle Count"

    plt.close(fig)
