import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({"figure.dpi": 400})


def plot_top_events(
    inputs,
    targets,
    nplots=36,   # total panels INCLUDING legend panel
    ncols=6,
    seed=123,
    max_special_labels=4,
    special_pids=(6, -6, 24, -24),
    pt_fmt="{:.0f}",
    fontsize=4,
):
    rng = np.random.default_rng(seed)

    pid_label = {6: "t", -6: "t̄", 24: "W+", -24: "W−"}
    pid_color = {6: "blue", -6: "orange", 24: "blue", -24: "orange"}
    pid_style = {6: "--", -6: "--", 24: ":", -24: ":"}

    # legend consumes ax[0]
    nevents = nplots - 1
    nrows = int(np.ceil(nplots / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 7))
    ax = np.array(ax).reshape(-1)

    # ---- legend panel ----
    ax[0].axis("off")
    ax[0].legend(
        handles=[
            Line2D([0], [0], color="C0", lw=1.5, alpha=0.7, label="Top 1 jets"),
            Line2D([0], [0], color="C1", lw=1.5, alpha=0.7, label="Top 2 jets"),
            Line2D([0], [0], color="gray", lw=1.5, alpha=0.6, label="Other jets"),
            Line2D([0], [0], color="blue",   linestyle="--", lw=1, label="t"),
            Line2D([0], [0], color="orange", linestyle="--", lw=1, label="t̄"),
            Line2D([0], [0], color="blue",   linestyle=":",  lw=1, label="W+"),
            Line2D([0], [0], color="orange", linestyle=":",  lw=1, label="W−"),
            Line2D([0], [0], color="gray",   linestyle="-",  lw=1, label="other particles"),
        ],
        loc="center",
        fontsize=6,
        frameon=False,
        handlelength=2.2,
        borderaxespad=0.2,
    )

    def t2n(x):
        return x.detach().cpu().numpy()

    def draw_rays(a, px, py, color, alpha=0.7, ls="-"):
        for x, y in zip(px, py):
            a.plot([0.0, float(x)], [0.0, float(y)], color=color, alpha=alpha, linestyle=ls)

    B = int(inputs["jet_pt"].shape[0])

    # choose nevents distinct events from the batch
    idxs = rng.choice(B, size=nevents, replace=False)

    for k, b in enumerate(idxs):
        a = ax[1 + k]

        nj = int(inputs["event_num_jets"][b].item())
        jets_px = t2n(inputs["px"][b, :nj])
        jets_py = t2n(inputs["py"][b, :nj])

        topmask = t2n(targets["top_jet_valid"][b, :2, :nj]).astype(bool)  # always two tops

        npart = int(targets.get("particle_num", np.array([0]))[b].item()) if "particle_num" in targets else 0
        par_px = t2n(targets["particle_px"][b, :npart]) if npart else np.empty(0)
        par_py = t2n(targets["particle_py"][b, :npart]) if npart else np.empty(0)
        par_pt = t2n(targets["particle_pt"][b, :npart]) if npart else np.empty(0)
        par_id = t2n(targets["particle_pid"][b, :npart]).astype(int) if npart else np.empty(0, dtype=int)

        # symmetric limits from jets + particles
        rmax = 0.0
        if jets_px.size:
            rmax = max(rmax, float(np.max(np.hypot(jets_px, jets_py))))
        if par_px.size:
            rmax = max(rmax, float(np.max(np.hypot(par_px, par_py))))
        R = 1.15 * rmax if rmax > 0 else 1.0

        # ---- jets ----
        for top_i, c in enumerate(("C0", "C1")):
            jj = np.where(topmask[top_i])[0]
            draw_rays(a, jets_px[jj], jets_py[jj], c, alpha=0.7)

        non_top = ~(topmask[0] | topmask[1])
        jj = np.where(non_top)[0]
        draw_rays(a, jets_px[jj], jets_py[jj], "gray", alpha=0.6)

        # ---- particles ----
        for x, y, pid in zip(par_px, par_py, par_id):
            draw_rays(
                a, [x], [y],
                pid_color.get(int(pid), "gray"),
                alpha=0.5,
                ls=pid_style.get(int(pid), "-"),
            )

        # ---- labels (top pt among special pids) ----
        if par_id.size:
            special = np.where(np.isin(par_id, np.array(special_pids)))[0]
            if special.size:
                order = special[np.argsort(par_pt[special])[::-1]][:max_special_labels]
                for j in order:
                    pid = int(par_id[j])
                    name = pid_label.get(pid, f"pid {pid}")
                    a.annotate(
                        f"{name} {pt_fmt.format(float(par_pt[j]))}",
                        xy=(float(par_px[j]), float(par_py[j])),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=fontsize,
                        ha="left",
                        va="bottom",
                        clip_on=True,
                    )

        a.set_aspect("equal", adjustable="box")
        a.set_xlim(-R, R)
        a.set_ylim(-R, R)
        a.axis("off")

    # any remaining panels off
    for kk in range(1 + nevents, len(ax)):
        ax[kk].axis("off")

    fig.tight_layout(w_pad=0.0, h_pad=0.0)
    return fig


