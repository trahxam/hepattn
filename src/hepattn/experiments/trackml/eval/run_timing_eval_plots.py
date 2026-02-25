import pathlib

import numpy as np
from matplotlib import pyplot as plt

# ----------------------------------------------------
# Plotting setup
# ----------------------------------------------------

plt.rcParams["figure.dpi"] = 400
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True

training_colours = {
    "Baseline": "mediumvioletred",
    "kMax-DeepLab": "mediumseagreen",
    "kMax-DeepLab-Sparse": "cornflowerblue",
}

training_dirs = {
    "Baseline": "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/trackml/logs/TRK-v6_20260105-T160031/times",
    "kMax-DeepLab": "",
    "kMax-DeepLab-Sparse": "/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/trackml/logs/TRK-Sparse-Win256_20260111-T203804/times",
}

out_dir = pathlib.Path("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/trackml/eval/plots/")
out_dir.mkdir(exist_ok=True)


def _resolve_times_dir(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.is_file():
        return p.parent
    if p.name == "times":
        return p
    return p / "times"


def _load_timing(times_dir: pathlib.Path):
    times_path = next(times_dir.glob("*_times.npy"))
    dims_path = next(times_dir.glob("*_dims.npy"))
    times = np.load(times_path)
    dims = np.asarray(np.load(dims_path, allow_pickle=True))

    hits = dims.astype(float)

    peak_alloc_path = next(times_dir.glob("*_peak_allocated_bytes.npy"), None)
    peak_rsvd_path = next(times_dir.glob("*_peak_reserved_bytes.npy"), None)
    peak_alloc = np.load(peak_alloc_path) if peak_alloc_path and peak_alloc_path.exists() else None
    peak_rsvd = np.load(peak_rsvd_path) if peak_rsvd_path and peak_rsvd_path.exists() else None

    n = min(len(hits), len(times))
    hits = hits[:n]
    times = times[:n]
    if peak_alloc is not None:
        peak_alloc = peak_alloc[:n]
    if peak_rsvd is not None:
        peak_rsvd = peak_rsvd[:n]

    return hits, times, peak_alloc, peak_rsvd


# ----------------------------------------------------
# Load data
# ----------------------------------------------------

timing_results = {}
for name, time_path in training_dirs.items():
    times_dir = _resolve_times_dir(time_path)
    timing_results[name] = _load_timing(times_dir)

# ----------------------------------------------------
# Time vs hits plot
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for name, (hits, times, _peak_alloc, _peak_rsvd) in timing_results.items():
    colour = training_colours.get(name, "black")
    ax.scatter(hits, times, color=colour, marker="+", alpha=0.3, label=name)

ax.set_ylabel("Inference Time [ms]")
ax.set_xlabel("Number of Hits in Event")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend(frameon=False)
fig.savefig(out_dir / "trackml_inference_time_vs_hits.png")

# Extrapolated time vs hits plot
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for name, (hits, times, _peak_alloc, _peak_rsvd) in timing_results.items():
    colour = training_colours.get(name, "black")
    ax.scatter(hits, times, color=colour, marker="+", alpha=0.3, label=name)
    if len(hits) > 1:
        coeffs = np.polyfit(hits, times, 1)
        x_line = np.array([hits.max(), 1_000_000.0])
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, linestyle="--", color=colour, alpha=0.7)
        for x_mark in (100_000.0, 500_000.0, 1_000_000.0):
            y_mark = coeffs[0] * x_mark + coeffs[1]
            label = f"{int(x_mark/1000)}k\n{y_mark:.1f} ms"
            ax.text(x_mark, y_mark, label, color=colour, fontsize=8, ha="left", va="bottom")

ax.set_ylabel("Inference Time [ms]")
ax.set_xlabel("Number of Hits in Event")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend(frameon=False)
fig.savefig(out_dir / "trackml_inference_time_vs_hits_extrapolated.png")

# ----------------------------------------------------
# Peak memory vs hits plot
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
has_mem = False
for name, (hits, _times, peak_alloc, _peak_rsvd) in timing_results.items():
    if peak_alloc is None:
        continue
    has_mem = True
    peak_mb = peak_alloc / (1024**2)
    colour = training_colours.get(name, "black")
    ax.scatter(hits, peak_mb, color=colour, marker="+", alpha=0.3, label=name)

if has_mem:
    ax.set_ylabel("Peak GPU Allocated [MB]")
    ax.set_xlabel("Number of Hits in Event")
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "trackml_peak_allocated_vs_hits.png")

if has_mem:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name, (hits, _times, peak_alloc, _peak_rsvd) in timing_results.items():
        if peak_alloc is None:
            continue
        peak_mb = peak_alloc / (1024**2)
        colour = training_colours.get(name, "black")
        ax.scatter(hits, peak_mb, color=colour, marker="+", alpha=0.3, label=name)
        if len(hits) > 1:
            coeffs = np.polyfit(hits, peak_mb, 1)
            x_line = np.array([hits.max(), 1_000_000.0])
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, linestyle="--", color=colour, alpha=0.7)
            for x_mark in (100_000.0, 500_000.0, 1_000_000.0):
                y_mark = coeffs[0] * x_mark + coeffs[1]
                label = f"{int(x_mark/1000)}k\n{y_mark:.1f} MB"
                ax.text(x_mark, y_mark, label, color=colour, fontsize=8, ha="left", va="bottom")

    ax.set_ylabel("Peak GPU Allocated [MB]")
    ax.set_xlabel("Number of Hits in Event")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "trackml_peak_allocated_vs_hits_extrapolated.png")

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
has_mem = False
for name, (hits, _times, _peak_alloc, peak_rsvd) in timing_results.items():
    if peak_rsvd is None:
        continue
    has_mem = True
    peak_mb = peak_rsvd / (1024**2)
    colour = training_colours.get(name, "black")
    ax.scatter(hits, peak_mb, color=colour, marker="+", alpha=0.3, label=name)

if has_mem:
    ax.set_ylabel("Peak GPU Reserved [MB]")
    ax.set_xlabel("Number of Hits in Event")
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "trackml_peak_reserved_vs_hits.png")

if has_mem:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name, (hits, _times, _peak_alloc, peak_rsvd) in timing_results.items():
        if peak_rsvd is None:
            continue
        peak_mb = peak_rsvd / (1024**2)
        colour = training_colours.get(name, "black")
        ax.scatter(hits, peak_mb, color=colour, marker="+", alpha=0.3, label=name)
        if len(hits) > 1:
            coeffs = np.polyfit(hits, peak_mb, 1)
            x_line = np.array([hits.max(), 1_000_000.0])
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, linestyle="--", color=colour, alpha=0.7)
            for x_mark in (100_000.0, 500_000.0, 1_000_000.0):
                y_mark = coeffs[0] * x_mark + coeffs[1]
                label = f"{int(x_mark/1000)}k\n{y_mark:.1f} MB"
                ax.text(x_mark, y_mark, label, color=colour, fontsize=8, ha="left", va="bottom")

    ax.set_ylabel("Peak GPU Reserved [MB]")
    ax.set_xlabel("Number of Hits in Event")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "trackml_peak_reserved_vs_hits_extrapolated.png")
