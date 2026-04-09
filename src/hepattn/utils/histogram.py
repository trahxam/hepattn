import numpy as np
from scipy.stats import binned_statistic

from hepattn.utils.stats import bayesian_binomial_error, combine_mean_std


class CountingHistogram:
    """Simple counting histogram over fixed bin edges.

    Attributes:
        bins (np.ndarray): Monotonically increasing bin edges of shape (nbins + 1,).
        counts (np.ndarray): Bin counts of shape (nbins,), dtype float32.
    """

    def __init__(self, bins: np.ndarray):
        self.bins = np.asarray(bins)
        self.counts = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def fill(self, values: np.ndarray) -> None:
        """Accumulate counts from values into the histogram bins."""
        values = np.asarray(values)
        if values.size == 0:
            return
        counts, _, _ = binned_statistic(values, values, statistic="count", bins=self.bins)
        self.counts += counts.astype(np.float32)


class BinomialHistogram:
    """Accumulates binomial (k, n) tallies per bin for computing ratios like
    efficiency, purity, fake rate, or retention rate.

    Usage:
        hist = BinomialHistogram(bins=np.linspace(0, 10, 32))

        for event in events:
            hist.fill(values=pt, numerator=is_matched, denominator=is_valid)

        ratio, errors = hist.ratio()
        plot_hist_to_ax(ax, ratio, hist.bins, value_errors=errors)

    Attributes:
        bins (np.ndarray): Bin edges of shape (nbins + 1,).
        k (np.ndarray): Accumulated numerators per bin.
        n (np.ndarray): Accumulated denominators per bin.
    """

    def __init__(self, bins: np.ndarray):
        self.bins = np.asarray(bins)
        self.k = np.zeros(len(self.bins) - 1, dtype=np.float32)
        self.n = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def fill(self, values: np.ndarray, numerator: np.ndarray, denominator: np.ndarray | None = None) -> None:
        """Accumulate k and n into bins.

        Args:
            values: The binning variable (e.g., pt, eta). Shape (N,).
            numerator: Boolean or float array for the numerator count (k). Shape (N,).
            denominator: Boolean or float array for the denominator count (n). Shape (N,).
                If None, counts all entries (equivalent to np.ones_like(numerator)).
        """
        values = np.asarray(values, dtype=np.float32)
        numerator = np.asarray(numerator, dtype=np.float32)

        if values.size == 0:
            return

        # Clip to bin range so out-of-range values land in edge bins
        values = np.clip(values, self.bins[0], self.bins[-1])

        k_binned, _, _ = binned_statistic(values, numerator, statistic="sum", bins=self.bins)
        self.k += k_binned.astype(np.float32)

        if denominator is None:
            n_binned, _, _ = binned_statistic(values, numerator, statistic="count", bins=self.bins)
        else:
            denominator = np.asarray(denominator, dtype=np.float32)
            n_binned, _, _ = binned_statistic(values, denominator, statistic="sum", bins=self.bins)
        self.n += n_binned.astype(np.float32)

    def ratio(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the binomial ratio k/n and Bayesian binomial errors.

        Returns:
            Tuple of (ratio, errors), each of shape (nbins,).
            Bins with n=0 will have ratio=nan and errors=nan.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.k / self.n
        errors = bayesian_binomial_error(self.k, self.n)
        return r, errors


class GaussianHistogram:
    """Maintains per-bin Gaussian summary stats (n, mean, std) and merges across fills.

    Attributes:
        bins (np.ndarray): Bin edges for the field.
        n (np.ndarray): Number of entries per bin (float32).
        mu (np.ndarray): Mean value per bin (float32).
        sigma (np.ndarray): Standard deviation per bin (float32).
    """

    def __init__(self, bins: np.ndarray):
        self.bins = np.asarray(bins)
        self.n = np.zeros(len(self.bins) - 1, dtype=np.float32)
        self.mu = np.zeros(len(self.bins) - 1, dtype=np.float32)
        self.sigma = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def fill(self, values: np.ndarray, weights: np.ndarray) -> None:
        """Accumulate Gaussian summary statistics per bin.

        Args:
            values: The binning variable. Shape (N,).
            weights: The values to summarize (compute mean/std of). Shape (N,).
        """
        values = np.asarray(values, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)

        if values.size == 0:
            return

        n_new, _, _ = binned_statistic(values, weights, statistic="count", bins=self.bins)
        mu_new, _, _ = binned_statistic(values, weights, statistic="mean", bins=self.bins)
        sig_new, _, _ = binned_statistic(values, weights, statistic="std", bins=self.bins)

        mu_comb, sig_comb, n_comb = combine_mean_std(self.mu, self.sigma, self.n, mu_new, sig_new, n_new)

        self.n = n_comb.astype(np.float32)
        self.mu = mu_comb.astype(np.float32)
        self.sigma = sig_comb.astype(np.float32)
