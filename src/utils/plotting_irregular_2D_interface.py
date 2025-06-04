import os
import sys
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.tri as tri


## these functions for Lagragian  interface plotting


def update_contourf_irregular_2D(frame_number, xs, ys, data, axsets, pcfsets, kwargs):
    list_of_collections = []
    for x, y, d, ax, pcfset, kw in zip(xs, ys, data, axsets, pcfsets, kwargs):
        pcfset = ax.scatter(x, y, c=d[frame_number, :], **kw)  # shading='gouraud' ,
        list_of_collections.append(pcfset)
    return list_of_collections


def grid_contour_plots_irregular_2D(
    data,
    nrows_ncols,
    titles,
    x,
    y,
    timeStp,
    dirname,
    ticks,
    fontsize,
    labelsize,
    axes_pad,
):

    npts = 200
    fig = plt.figure()
    xi = np.linspace(x.min(), x.max(), npts)
    yi = np.linspace(y.min(), y.max(), npts)

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
        minmax_list.append([np.min(d[timeStp, :]), np.max(d[timeStp, :])])
        kwargs_list.append(
            dict(cmap="coolwarm", vmin=minmax_list[-1][0], vmax=minmax_list[-1][1])
        )
    # CREATE PLOTS
    pcfsets = []
    index = 1
    axsets = []
    # CREATE PLOTS
    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(
        grid, data, kwargs_list, minmax_list, titles
    ):
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z[timeStp, :])
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        pcf = [
            ax.pcolormesh(xi, yi, zi, shading="gouraud", **kwargs)
        ]  # shading='gouraud' ,
        pcfsets.append(pcf)
        cb = ax.cax.colorbar(
            pcf[0], ticks=np.linspace(minmax[0], minmax[1], ticks), format="%.3e"
        )
        cb.ax.tick_params(labelsize=labelsize)
        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=7)
        ax.set_ylabel("y", labelpad=labelsize, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.tick_params(labelsize=labelsize)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")
        axsets.append(ax)

    fig.set_size_inches(15, 15, True)
    # fig.subplots_adjust(left=0.7, bottom=0, right=2.2, top=1, wspace=None, hspace=None)
    plt.savefig(dirname, dpi=300, bbox_inches="tight")
    plt.close(
        "all",
    )
    return fig, axsets, pcfsets, kwargs_list


#########################################################


def draw_contourf_irregular_2D(
    time,
    xfa,
    yfa,
    data,
    titles,
    model_dirname,
    nrows_ncols,
    values,
    ticks=5,
    fontsize=10,
    labelsize=7,
    axes_pad=1.4,
):
    """
    draw results
    """
    # minmax = [xfa.min(), xfa.max(), yfa.min(), yfa.max()]
    (nrows, ncols) = nrows_ncols

    for index in values:
        file = os.path.join(model_dirname, f"interface_tricontourf_{index}.png")
        fig, axsets, pcfsets, kwargs_list = grid_contour_plots_irregular_2D(
            data,
            nrows_ncols,
            titles,
            xfa,
            yfa,
            index,
            file,
            ticks,
            fontsize=fontsize,
            labelsize=labelsize,
            axes_pad=axes_pad,
        )


### this code for animation
# ani = FuncAnimation(
#     fig,
#     update_contourf_irregular_2D,
#     frames=len(time),
#     fargs=(
#         [xfa] * np.prod((nrows, ncols)),
#         [yfa] * np.prod((nrows, ncols)),
#         data,
#         axsets,
#         pcfsets,
#         kwargs_list,
#     ),
#     interval=50,
#     blit=False,
#     repeat=False,
#     save_count=sys.maxsize,
# )

# # FFwriter = mpa.FFMpegWriter(fps=30, codec="libx264")
# # writergif = PillowWriter(fps=30)
# ani.save(os.path.join(dirname, "result.gif"))
# plt.close(
#     "all",
# )


#############################################################################################


def combine_contourf_fluid_interface_2D(
    fluid_data,
    interface_data,
    axis,
    time_,
    model_dirname,
    titles,
    nrows_ncols,
    time_values,
    ref,
    img_size,
    ticks=0,
    fontsize=0,
    labelsize=0,
    axes_pad=0,
):

    for time_step in time_values:
        dirname = os.path.join(
            model_dirname, "combined_tricontourf_" + str(time_step) + ".png"
        )
        fig, grid, pcfsets, kwargs = combined_fluid_interface_plot(
            fluid_data,
            interface_data,
            axis,
            ref,
            time_step,
            titles,
            dirname,
            img_size,
            nrows_ncols,
            ticks,
            fontsize=fontsize,
            labelsize=labelsize,
            axes_pad=axes_pad,
        )


def combined_fluid_interface_plot(
    fluid_data,
    interface_data,
    axis,
    ref,
    time_step,
    titles,
    dirname,
    img_size,
    nrows_ncols,
    ticks=5,
    fontsize=10,
    labelsize=8,
    axes_pad=0.1,
):

    # Set up the figure and grid
    fig = plt.figure()
    [fluid_x, fluid_y, interface_x, interface_y] = axis

    npts = 200
    xi = np.linspace(interface_x.min(), interface_x.max(), npts)
    yi = np.linspace(interface_y.min(), interface_y.max(), npts)

    # Creating the grid layout for subplots
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
    fluid_minmax_list = []
    fluid_kwargs_list = []

    interface_minmax_list = []
    interface_kwargs_list = []
    for d in fluid_data:
        min_ = np.min(d[time_step, :])
        max_ = np.max(d[time_step, :])
        if min_ == max_ == 0:
            min_ += 1e-16
            max_ += 1e-6
        fluid_minmax_list.append([min_, max_])
        fluid_kwargs_list.append(
            dict(
                levels=np.linspace(
                    fluid_minmax_list[-1][0], fluid_minmax_list[-1][1], 60
                ),
                cmap="coolwarm",
                vmin=fluid_minmax_list[-1][0],
                vmax=fluid_minmax_list[-1][1],
            )
        )

    for d in interface_data:
        interface_minmax_list.append([np.min(d[time_step, :]), np.max(d[time_step, :])])
        interface_kwargs_list.append(
            dict(
                cmap="coolwarm",
                vmin=interface_minmax_list[-1][0],
                vmax=interface_minmax_list[-1][1],
            )
        )

    # CREATE PLOTS
    pcfsets = []
    for (
        ax,
        fluid_z,
        interface_z,
        fluid_kwargs,
        interface_kwargs,
        fluid_minmax,
        interface_minmax,
        title,
    ) in zip(
        grid,
        fluid_data,
        interface_data,
        fluid_kwargs_list,
        interface_kwargs_list,
        fluid_minmax_list,
        interface_minmax_list,
        titles,
    ):

        print("title", title)
        print(
            "np.min([fluid_minmax[0],interface_minmax[0]])",
            fluid_minmax[0],
            fluid_minmax[1],
        )
        pcf1 = [ax.contourf(fluid_x, fluid_y, fluid_z[time_step, :, :], **fluid_kwargs)]

        triang = tri.Triangulation(interface_x, interface_y)
        interpolator = tri.LinearTriInterpolator(triang, interface_z[time_step, :])
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        pcf = [ax.pcolormesh(xi, yi, zi, shading="gouraud", **interface_kwargs)]

        pcfsets.append(pcf1)
        cb = ax.cax.colorbar(
            pcf1[0],
            ticks=np.linspace(fluid_minmax[0], fluid_minmax[1], ticks),
            format="%.3e",
        )

        # print(np.max([fluid_minmax[1],interface_minmax[1]]))

        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=7)
        ax.set_ylabel("y", labelpad=labelsize, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x", fontsize=fontsize)
        ax.tick_params(labelsize=labelsize)
        ax.set_aspect("equal")

    fig.set_size_inches(img_size[0], img_size[1], True)
    fig.subplots_adjust(
        left=1.3, bottom=0.1, right=1.5, top=0.12, wspace=None, hspace=None
    )
    plt.tight_layout()
    plt.savefig(dirname, dpi=300, bbox_inches="tight")
    plt.close(
        "all",
    )
    return fig, grid, pcfsets, fluid_kwargs_list
