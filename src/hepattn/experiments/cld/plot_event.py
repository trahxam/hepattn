import matplotlib.pyplot as plt
import torch

plt.rcParams["figure.dpi"] = 300


def plot_cld_event_reconstruction(inputs, reconstruction, axes_spec, object_name="particle", batch_idx=0):
    num_axes = len(axes_spec)

    fig, ax = plt.subplots(1, num_axes)
    fig.set_size_inches(8 * num_axes, 8)

    ax = [ax] if num_axes == 1 else ax.flatten()

    colormap = plt.cm.tab10
    cycler = [colormap(i) for i in range(colormap.N)]

    sihit_names = ["vtb", "vte", "itb", "ite", "otb", "ote", "sihit", "vtxd", "trkr"]

    ecal_names = [
        "ecb",
        "ece",
        "ecal",
    ]

    hcal_names = [
        "hcb",
        "hce",
        "hcal",
    ]

    for ax_idx, ax_spec in enumerate(axes_spec):
        # if "sample_id" in reconstruction:
        #    event_id = reconstruction[f"sample_id"][batch_idx]
        #    ax[ax_idx].set_title(f"Event {event_id}")

        for input_name in ax_spec["input_names"]:
            x = inputs[f"{input_name}_{ax_spec['x']}"][batch_idx]
            y = inputs[f"{input_name}_{ax_spec['y']}"][batch_idx]

            ax[ax_idx].scatter(x, y, alpha=0.25, s=1.0, color="black")

            num_particles = reconstruction[f"{object_name}_{input_name}_valid"][batch_idx].shape[-2]

            for object_idx in range(num_particles):
                # Only plot the masks for slots that are marked valid
                if not reconstruction[f"{object_name}_valid"][batch_idx][object_idx]:
                    continue

                color = cycler[object_idx % len(cycler)]
                mask = reconstruction[f"{object_name}_{input_name}_valid"][batch_idx][object_idx]

                # Tracker hit
                if input_name in sihit_names:
                    # Used for sorting the hits in time when we want to plot them in order in the tracker
                    idx = torch.argsort(inputs[f"{input_name}_time"][batch_idx][mask], dim=-1)

                    ax[ax_idx].plot(x[mask][idx], y[mask][idx], color=color, marker="o", alpha=0.75, linewidth=1.0, ms=2.0)

                    # px = reconstruction[f"{object_name}_{input_name}_{ax_spec['px']}"][batch_idx][object_idx]
                    # py = reconstruction[f"{object_name}_{input_name}_{ax_spec['py']}"][batch_idx][object_idx]

                    # for i in range(len(x[mask][idx])):
                    #    mag = 5.0 * torch.sqrt(px[mask][idx][i]** 2 + py[mask][idx][i]**2)
                    #    ax[ax_idx].arrow(x[mask][idx][i], y[mask][idx][i], dx=px[mask][idx][i] / mag,
                    #                      dy=py[mask][idx][i] / mag, color=color, alpha=0.8, head_width=0.04)

                # ECAL hit
                elif input_name in ecal_names:
                    ax[ax_idx].scatter(x[mask], y[mask], color=color, marker=".", alpha=0.5, s=1.0)

                # HCAL hit
                elif input_name in hcal_names:
                    ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="s", alpha=0.5, s=4.0)

                # Muon hit
                elif input_name == "muon":
                    ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="x", alpha=0.75, s=4.0)

                ax[ax_idx].set_xlabel(ax_spec["x"])
                ax[ax_idx].set_ylabel(ax_spec["y"])
                ax[ax_idx].set_aspect("equal", "box")

    return fig
