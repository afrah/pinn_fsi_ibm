import os

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid


def draw_contourf_regular_2D(
    tf,
    xf,
    yf,
    data,
    nrows_ncols,
    time_steps,
    file_name,
    img_width=35,
    img_height=32,
    axes_pad=1.0,
    fontsize=16,
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
        cbar_pad=0.03,
    )

    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []

    for d, time_step in zip(data, time_steps):
        min_ = np.min(d[time_step, :, :])
        max_ = np.max(d[time_step, :, :])
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
    for ax, z, kwargs, minmax, time_step in zip(
        grid, data, kwargs_list, minmax_list, time_steps
    ):
        pcf = [ax.contourf(xf, yf, z[time_step, :, :], **kwargs)]
        cb = ax.cax.colorbar(
            pcf[0], ticks=np.linspace(minmax[0], minmax[1], 3), format="%.1e"
        )

        cb.ax.tick_params(labelsize=22)
        ax.cax.tick_params(labelsize=22)
        ax.locator_params(axis="x", nbins=3)  # y number of ticks 3
        ax.locator_params(axis="y", nbins=3)
        ax.set_ylabel("y→", labelpad=14, fontsize=fontsize, rotation="vertical")
        ax.set_xlabel("x→", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        pcfsets.append(pcf)
        ax.set_aspect("equal")

    fig.set_size_inches(img_width, img_height, True)
    plt.tight_layout()
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close(
        "all",
    )
    return fig, grid, pcfsets, kwargs_list
