import matplotlib.pyplot as plt


def plot_cld_event_reconstruction(inputs, reconstruction, axes_spec):
    num_axes = len(axes_spec)

    fig, ax = plt.subplots(1, num_axes)
    fig.set_size_inches(8 * num_axes, 8)

    if num_axes == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    colormap = plt.cm.tab10
    cycler = [colormap(i) for i in range(colormap.N)]

    batch_idx = 0
    num_particles = reconstruction["particle_valid"].sum()

    for ax_idx, ax_spec in enumerate(axes_spec):
        for input_name in ax_spec["input_names"]:
            x = inputs[f"{input_name}_{ax_spec['x']}"][batch_idx]
            y = inputs[f"{input_name}_{ax_spec['y']}"][batch_idx]
                       
            ax[ax_idx].scatter(x, y, s=1.0, color="black")
            
            for mcparticle_idx in range(num_particles):
                color = cycler[mcparticle_idx % len(cycler)]
                mask = reconstruction[f"particle_{input_name}_mask"][batch_idx][mcparticle_idx]

                # Tracker hit
                if input_name[1] == "t":
                    ax[ax_idx].plot(x[mask], y[mask], color=color, marker="o", alpha=0.5, linewidth=1.0)

                # Calo hit
                elif input_name[1] == "c":
                    ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="s", alpha=0.2)
                
                # Muon hit
                elif input_name == "muon":
                    ax[ax_idx].scatter(x[mask], y[mask], color=color, marker="+", alpha=0.8)
                
                ax[ax_idx].set_xlabel(ax_spec["x"])
                ax[ax_idx].set_ylabel(ax_spec["y"])
                ax[ax_idx].set_aspect("equal", "box")
    
    return fig