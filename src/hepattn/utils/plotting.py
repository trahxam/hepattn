import matplotlib.pyplot as plt
import numpy as np


def setup_plotting(usetex: bool = False, dpi: int = 300, fontsize: int = 10):
    """Set common matplotlib defaults for publication-quality plots."""
    plt.rcParams["text.usetex"] = usetex
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.constrained_layout.use"] = True


def plot_hist_to_ax(ax, values, bins, value_errors=None, color="cornflowerblue", vertical_lines=False, label=None, linestyle="-"):
    for bin_idx in range(len(bins) - 1):
        px = np.array([bins[bin_idx], bins[bin_idx + 1]])
        py = np.array([values[bin_idx], values[bin_idx]])

        if color is None:
            color = "cornflowerblue"

        if linestyle is None:
            linestyle = "-"

        ax.plot(px, py, color=color, linewidth=1.0, label=label if bin_idx == 0 else None, linestyle=linestyle)

        if value_errors is not None:
            pe = np.array([value_errors[bin_idx], value_errors[bin_idx]])
            ax.fill_between(px, py - pe, py + pe, color=color, alpha=0.1, ec="none")

        if vertical_lines and bin_idx < len(bins) - 2:
            px = np.array([bins[bin_idx + 1], bins[bin_idx + 1]])
            py = np.array([values[bin_idx], values[bin_idx + 1]])
            ax.plot(px, py, color=color, linewidth=1.0)
