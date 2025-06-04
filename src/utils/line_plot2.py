import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


def enhanced_line_plot(
    time_,
    x_axis,
    y_axis,
    data,
    models,
    data_labels,
    timeStp,
    yStep,
    model_dirname,
    fontsize=16,
    labelsize=14,
):
    """
    Enhanced line plot function with improved visualization features.

    Parameters:
    -----------
    time_ : array-like
        Time values
    x_axis : array-like
        X-axis values
    y_axis : array-like
        Y-axis values
    data : dict
        Dictionary containing the data for each variable
    models : list
        List of model names
    line_styles : list
        List of line styles for each model
    data_labels : list
        List of data labels to plot
    colors : list
        List of colors for each model
    timeStp : list
        List of time step indices
    yStep : list
        List of y-axis indices
    model_dirname : str
        Directory to save the figures
    fontsize : int, optional
        Font size for titles and labels
    labelsize : int, optional
        Font size for tick labels
    """
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "lightgray"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 120

    line_widths = [1.5, 3, 5]
    alphas = [0.9, 0.6, 0.3]
    line_styles = ["solid", "dashed", "dashdot"]
    colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
    ]

    bbox_to_anchors = [(0.5, 0.10) for _ in data_labels]
    fig_sizes = [(12, 10) for _ in data_labels]

    c = 2
    for data_label, bbox_to_anchor, fig_size in zip(
        data_labels, bbox_to_anchors, fig_sizes
    ):
        global_min = float("inf")
        global_max = float("-inf")

        for z in data[data_label]:
            for t in timeStp:
                for y in yStep:
                    values = z[:, y, :][t, :]
                    global_min = min(global_min, np.min(values))
                    global_max = max(global_max, np.max(values))

        y_padding = 0.1 * (global_max - global_min)
        y_min = global_min - y_padding
        y_max = global_max + y_padding
        fig = plt.figure(figsize=fig_size)

        n_plots = len(timeStp) * len(yStep)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        grid = ImageGrid(
            fig,
            111,
            direction="row",
            nrows_ncols=(n_rows, n_cols),
            label_mode="1",
            axes_pad=0.4,
            share_all=True,
        )

        lines = []
        labels = []
        index = 0
        for t in timeStp:
            for y in yStep:
                if index >= len(grid):
                    break

                ax = grid[index]
                index += 1

                for z, color, label, line_style, lw, alpha in zip(
                    data[data_label], colors, models, line_styles, line_widths, alphas
                ):
                    if data_label == "p":
                        y_value = z[:, y, :][t, :] / c
                    else:
                        y_value = z[:, y, :][t, :]
                    x_value = x_axis

                    line = ax.plot(
                        x_value,
                        y_value,
                        color=color,
                        linewidth=lw,
                        alpha=alpha,
                        label=label,
                        linestyle=line_style,
                        marker="",
                        zorder=3,
                    )

                    if len(lines) < len(models):
                        lines.append(line[0])
                        labels.append(label)

                ax.set_title(
                    f"$t = {time_[t] * 10:.2f}$ , $y = {y_axis[y]:.2f}$",
                    fontsize=fontsize,
                    pad=10,
                )

                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))

                if data_label == "p":
                    ax.set_ylim([y_min / c, y_max / c])
                else:
                    ax.set_ylim([y_min, y_max])

                if index % n_cols == 1:
                    if data_label in ["p"]:
                        ax.set_ylabel(
                            rf"${data_label}/c$→",
                            fontsize=fontsize,
                            labelpad=7,
                        )
                    else:
                        ax.set_ylabel(
                            rf"${data_label}$→",
                            fontsize=fontsize,
                            labelpad=7,
                        )

                if index > (n_rows - 1) * n_cols:
                    ax.set_xlabel("x→", fontsize=fontsize)

                ax.tick_params(labelsize=labelsize)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                    spine.set_linewidth(0.8)
                    spine.set_color("black")

        legend = fig.legend(
            lines,
            labels,
            loc="lower center",
            # bbox_to_anchor=bbox_to_anchor,
            # bbox_to_anchor=(0.5, -0.01),  # Place legend below subplots
            ncol=min(len(models), 4),
            fontsize=10,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            shadow=False,
            borderpad=1,
            handlelength=3,
            handletextpad=0.4,
            columnspacing=1.5,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_linewidth(1)

        # Add variable name as a super title
        # plt.suptitle(f"{data_label} Comparison", fontsize=fontsize + 4, y=0.98)

        # legend_bottom = {
        #     "u": 0.0000007,
        #     "v": 0.05,
        #     "p": 0.05,
        # }
        # # Adjust layout and save
        # # plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        # plt.subplots_adjust(
        #     bottom=legend_bottom[data_label]
        # )  # Add space for the legend

        file_name = os.path.join(model_dirname, f"line_plot_{data_label}.png")
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
