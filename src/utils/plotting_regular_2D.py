import os
from tkinter import font

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import ImageGrid


def draw_contourf_regular_2D(
    tf,
    xf,
    yf,
    data,
    titles,
    nrows_ncols,
    values,
    xref,
    yref,
    model_dirname,
    img_width,
    img_height,
    ticks,
    fontsize,
    labelsize,
    axes_pad,
):

    yf = yref * xf
    xf = xref * yf
    # minmax = [xf.min(), xf.max(), yf.min(), yf.max()]

    for timeStp in values:
        file = os.path.join(model_dirname, "tricontourf_" + str(timeStp) + ".png")
        fig, grid, pcfsets, kwargs = grid_contour_plots_regular(
            data,
            nrows_ncols,
            titles,
            xf,
            yf,
            timeStp,
            file,
            img_width,
            img_height,
            ticks,
            fontsize=fontsize,
            labelsize=labelsize,
            axes_pad=axes_pad,
        )


def grid_contour_plots_regular(
    data,
    nrows_ncols,
    titles,
    x,
    y,
    time_step,
    dirname,
    img_width,
    img_height,
    ticks,
    fontsize,
    labelsize,
    axes_pad,
):

    # CREATE FIGURE AND AXIS
    fig = plt.figure()

    grid = ImageGrid(
        fig,
        111,
        direction="row",
        nrows_ncols=nrows_ncols,
        label_mode="1",
        axes_pad=axes_pad,
        share_all=False,
        cbar_mode="each",
        cbar_location="right",
        cbar_size="5%",
        cbar_pad=0.0,
    )

    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []

    for d in data:
        min_ = np.min(d[time_step, :])
        max_ = np.max(d[time_step, :])
        if min_ == max_ == 0:
            min_ += -1e-16
            max_ += 1e-6
        minmax_list.append([min_, max_])
        kwargs_list.append(
            dict(
                levels=np.linspace(minmax_list[-1][0], minmax_list[-1][1], 60),
                cmap="coolwarm",
                vmin=minmax_list[-1][0],
                vmax=minmax_list[-1][1],
            )
        )

    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(
        grid, data, kwargs_list, minmax_list, titles
    ):
        pcf = [ax.contourf(x, y, z[time_step, :, :], **kwargs)]
        pcfsets.append(pcf)
        cb = ax.cax.colorbar(
            pcf[0], ticks=np.linspace(minmax[0], minmax[1], ticks), format="%.1e"
        )

        cb.ax.tick_params(labelsize=fontsize)
        ax.cax.tick_params(labelsize=fontsize)

        ax.set_title(title, fontsize=fontsize, pad=7)
        ax.locator_params(axis="x", nbins=3)  # y number of ticks 3
        ax.locator_params(axis="y", nbins=3)
        ax.set_ylabel("y→", labelpad=14, fontsize=fontsize, rotation="vertical")
        ax.set_xlabel("x→", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        # ax.set_xlim(x.min(), x.max())
        # ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")

    fig.set_size_inches(img_width, img_height)
    # fig.subplots_adjust(
    #     left=0, bottom=0.0, right=4, top=0.1, wspace=5.0, hspace=0.0
    # )

    plt.tight_layout()
    plt.savefig(dirname, dpi=300, bbox_inches="tight")
    plt.close("all")
    return fig, grid, pcfsets, kwargs_list


def plot_time_profile_regular_data_IBM(
    dirname, steps, txy, exact, pred, part, img_width, img_height
):

    [tstep, xstep, ystep] = steps
    [x, y, time_] = txy

    # minVal = min(exact.min(), pred.min())
    # maxVal = max(exact.max(), pred.max())

    x = x.reshape(tstep, xstep, ystep)[0, :, 0]
    y = y.reshape(tstep, xstep, ystep)[0, 0, :]
    time_ = time_.reshape(tstep, xstep, ystep)[:, 0, 0]

    exact = exact.reshape(tstep, xstep, ystep)
    pred = pred.reshape(tstep, xstep, ystep)

    timeStp = [35, 55, 80]
    yStep = [35, 55, 80]

    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "lightgray"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5
    # Draw boundary
    plt.rcParams["axes.edgecolor"] = "lightgray"
    plt.rcParams["axes.linewidth"] = 0.5

    fig = plt.figure(figsize=(img_width, img_height))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(3, 3),
        axes_pad=0.45,
        share_all=True,
        cbar_mode=None,
    )

    for i in range(len(yStep)):
        for j in range(len(timeStp)):
            ax = grid[i * len(timeStp) + j]
            exact_values = exact[:, yStep[i], :][timeStp[j], :]
            pred_values = pred[:, yStep[i], :][timeStp[j], :]

            if part in ["F_x", "F_y"]:
                exact_values /= 10
                pred_values /= 10

            ax.plot(
                x,
                exact_values,
                color="#FFA500",
                linewidth=3,
                alpha=0.7,
                label="Exact",
            )
            ax.plot(
                x,
                pred_values,
                color="#1E90FF",
                linestyle="--",
                dashes=(3, 3),  # Equal length of dashes and gaps
                linewidth=1,
                label="Prediction",
            )

            fontsize = 8

            # Set y-axis to logarithmic scale

            # Adjusting x and y limits
            ax.set_xlim([x.min() - 0.05, x.max() + 0.05])
            y_min = min(exact_values.min(), pred_values.min())
            y_max = max(exact_values.max(), pred_values.max())

            ax.set_ylim([y_min - 0.1, y_max + 0.21])

            # Optional: set y-limits only if needed
            # ax.set_ylim([-0.5, 0.5])

            ax.set_title(
                "$t = %.2f$ , y =  %.2f" % (time_[timeStp[j]], y[yStep[i]]),
                fontsize=fontsize,
            )

            if i == 2 and j == 0:
                ax.set_xlabel("$x$→", fontsize=fontsize)
                ylabel = rf"${part}$→"
                if part in ["F_x", "F_y"]:
                    ylabel += "/10"
                ax.set_ylabel(ylabel, labelpad=14, fontsize=fontsize)
                ax.locator_params(axis="x", nbins=3)  # y number of ticks 3
                ax.locator_params(axis="y", nbins=3)
                ax.tick_params(labelsize=fontsize)

            if i == 2 and j == 1:
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=2,
                    frameon=False,
                )

    fig.set_size_inches(img_width, img_height, forward=True)
    # plt.tight_layout()
    plt.savefig(
        os.path.join(dirname, "time_profile_" + part + ".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        "all",
    )
