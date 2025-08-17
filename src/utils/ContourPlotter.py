import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from typing import List, Tuple

from src.utils.cmap import orange_cmap2


SOLUTION_MAP = "rainbow"
ERROR_MAP = orange_cmap2


class ContourPlotter:
    def __init__(self, fontsize: int = 7, labelsize: int = 7, axes_pad: float = 0.5):
        self.fontsize = fontsize
        self.labelsize = labelsize
        self.axes_pad = axes_pad

    def draw_contourf_regular_2D(
        self,
        tf: np.ndarray,
        xf: np.ndarray,
        yf: np.ndarray,
        data: List[np.ndarray],
        titles: List[str],
        nrows_ncols: Tuple[int, int],
        time_steps: List[int],
        xref: float = 1.0,
        yref: float = 1.0,
        model_dirname: str = "./",
        img_width: int = 10,
        img_height: int = 10,
        ticks: int = 3,
    ) -> None:
        """
        Draw 2D contour plots for cavity flow results.
        Each row (u, v, p) has independent scaling for solutions and errors.
        """

        x_grid = xf.flatten()  # Convert to 1D arrays first
        y_grid = yf.flatten()
        X, Y = np.meshgrid(x_grid, y_grid)

        for time_step in time_steps:
            filename = os.path.join(model_dirname, f"tricontourf_{time_step}.pdf")
            self._create_contour_plot(
                data=data,
                nrows_ncols=nrows_ncols,
                titles=titles,
                X=X,
                Y=Y,
                time_step=time_step,
                filename=filename,
                img_width=img_width,
                img_height=img_height,
                ticks=ticks,
            )

    def _create_contour_plot(
        self,
        data: List[np.ndarray],
        nrows_ncols: Tuple[int, int],
        titles: List[str],
        X: np.ndarray,
        Y: np.ndarray,
        time_step: int,
        filename: str,
        img_width: int,
        img_height: int,
        ticks: int,
    ) -> None:
        fig = plt.figure()
        grid = ImageGrid(
            fig,
            111,
            direction="row",
            nrows_ncols=nrows_ncols,
            label_mode="1",
            axes_pad=self.axes_pad,
            share_all=False,
            cbar_mode="each",
            cbar_location="right",
            cbar_size="5%",
            cbar_pad=0.02,
        )

        n_rows, n_cols = nrows_ncols
        plots_per_row = n_cols

        # Process each row (u, v, p) separately
        plot_params = []
        for row in range(n_rows):
            # Get data for current row (u, v, or p)
            row_start = row * plots_per_row
            row_end = row_start + plots_per_row
            row_data = data[row_start:row_end]
            row_titles = titles[row_start:row_end]

            # Split into solutions (exact + predictions) and errors
            row_solutions = []
            row_errors = []
            for d, title in zip(row_data, row_titles):
                # Extract 2D slice for current time step
                d_slice = d[time_step] if d.ndim == 3 else d
                if d_slice.ndim == 1:
                    d_slice = d_slice.reshape(X.shape)

                if "error" in title.lower():
                    row_errors.append(d_slice)
                    # print("inside if error title: ", title)
                else:
                    row_solutions.append(d_slice)
                    # print("inside else title: ", title)

            # Calculate min/max for solutions and errors in this row
            solution_min = min(np.min(d) for d in row_solutions)
            solution_max = max(np.max(d) for d in row_solutions)

            if row_errors:
                error_min = 0.0  # smin(np.min(d) for d in row_errors)
                error_max = max(np.max(d) for d in row_errors)

            # Create parameters for each plot in current row
            for i, (d, title) in enumerate(zip(row_data, row_titles)):
                if "error" in title.lower():
                    cmap = ERROR_MAP
                    vmin, vmax = error_min, error_max
                    # print("error title", title)
                else:
                    cmap = SOLUTION_MAP
                    vmin, vmax = solution_min, solution_max
                    # print("solution title", title)

                # Handle zero-valued data
                if vmin == vmax == 0:
                    vmin += -1e-16
                    vmax += 1e-6

                plot_params.append(
                    {
                        "minmax": [vmin, vmax],
                        "kwargs": {
                            "levels": np.linspace(vmin, vmax, 50),
                            "cmap": cmap,
                            "vmin": vmin,
                            "vmax": vmax,
                        },
                    }
                )

        # Create individual plots
        for idx, (ax, Z, params, title) in enumerate(
            zip(grid, data, plot_params, titles)
        ):
            ax.set_aspect("equal", adjustable="box")
            # Create contour plot
            pcf = ax.contourf(X, Y, Z[time_step, :, :], **params["kwargs"])

            cb = ax.cax.colorbar(
                pcf,
                ticks=np.linspace(params["minmax"][0], params["minmax"][1], ticks),
                format="%.1e",
            )
            cb.ax.tick_params(labelsize=self.labelsize)

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_title(title, fontsize=self.fontsize, pad=7)

            row = idx // plots_per_row
            col = idx % plots_per_row

            if row == n_rows - 1 and col == 0:
                ax.set_xticks(np.linspace(X.min(), X.max(), 3))
                ax.set_yticks(np.linspace(Y.min(), Y.max(), 4))
                ax.set_xticklabels(
                    [f"{x:.1f}" for x in np.linspace(X.min(), X.max(), 3)]
                )
                ax.set_yticklabels(
                    [f"{y:.1f}" for y in np.linspace(Y.min(), Y.max(), 4)]
                )
                ax.set_xlabel(r"$x_1$→", fontsize=self.fontsize)
                ax.set_ylabel(r"$x_2$→", fontsize=self.fontsize)
                ax.tick_params(axis="both", which="major", labelsize=self.labelsize)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

        self._finalize_figure(fig, filename, img_width, img_height)

    def _finalize_figure(
        self,
        fig: plt.Figure,
        path: str,
        img_width: int,
        img_height: int,
    ) -> None:
        fig.set_size_inches(img_width, img_height, True)
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.15,
            top=0.92,
            wspace=0.5,
        )
        # plt.tight_layout()
        plt.savefig(
            path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        plt.show()
        plt.close("all")
        
        

    def draw_contourf_time_series(
        self,
        tf: np.ndarray,
        xf: np.ndarray,
        yf: np.ndarray,
        data: List[np.ndarray],
        titles: List[str],
        nrows_ncols: Tuple[int, int],
        time_steps: List[int],
        model_dirname: str = "./",
        img_width: int = 300,
        img_height: int = 60,
        ticks: int = 3,
        variable_name: str = "u",
        solution_type: str = "exact",
        draw_axis_ticks: bool = False
    ) -> None:
        x_grid = xf.flatten()
        y_grid = yf.flatten()
        X, Y = np.meshgrid(x_grid, y_grid)
        
        filename = os.path.join(model_dirname, f"time_series_{variable_name}_{solution_type}.pdf")
        self._create_time_series_plot(
            data=data,
            nrows_ncols=nrows_ncols,
            titles=titles,
            X=X,
            Y=Y,
            time_steps=time_steps,
            filename=filename,
            img_width=img_width,
            img_height=img_height,
            ticks=ticks,
            solution_type=solution_type,
            draw_axis_ticks=draw_axis_ticks
        )

    def _create_time_series_plot(
        self,
        data: List[np.ndarray],
        nrows_ncols: Tuple[int, int],
        titles: List[str],
        X: np.ndarray,
        Y: np.ndarray,
        time_steps: List[int],
        filename: str,
        img_width: int,
        img_height: int,
        ticks: int,
        solution_type: str,
        draw_axis_ticks: bool = False
    ) -> None:
        from src.utils.cmap import orange_cmap2
        
        SOLUTION_MAP = "rainbow"
        ERROR_MAP = orange_cmap2
        
        fig = plt.figure()
        grid = ImageGrid(
            fig,
            111,
            direction="row",
            nrows_ncols=nrows_ncols,
            label_mode="1",
            axes_pad=self.axes_pad,
            share_all=False,
        )

        n_rows, n_cols = nrows_ncols
        
        # Calculate global min/max across all time steps for consistent scaling
        all_data_slices = []
        for i, (data_array, time_step) in enumerate(zip(data, time_steps)):
            data_slice = data_array[time_step] if data_array.ndim == 3 else data_array
            if data_slice.ndim == 1:
                data_slice = data_slice.reshape(X.shape)
            all_data_slices.append(data_slice)
        
        # Use different scaling for errors vs solutions
        if solution_type == "error":
            global_min = 0.0
            global_max = max(np.max(d) for d in all_data_slices)
            cmap = ERROR_MAP
        else:
            global_min = min(np.min(d) for d in all_data_slices)
            global_max = max(np.max(d) for d in all_data_slices)
            cmap = SOLUTION_MAP
        
        # Handle zero-valued data
        if global_min == global_max == 0:
            global_min += -1e-16
            global_max += 1e-6
        
        # Create individual plots
        for idx, (ax, data_array, title, time_step) in enumerate(
            zip(grid, data, titles, time_steps)
        ):
            ax.set_aspect("equal", adjustable="box")
            
            # Get data slice for this time step
            data_slice = data_array[time_step] if data_array.ndim == 3 else data_array
            if data_slice.ndim == 1:
                data_slice = data_slice.reshape(X.shape)
            
            # Create contour plot
            pcf = ax.contourf(
                X, Y, data_slice,
                levels=np.linspace(global_min, global_max, 50),
                cmap=cmap,
                vmin=global_min,
                vmax=global_max
            )

            # # Add colorbar
            # cb = ax.cax.colorbar(
            #     pcf,
            #     ticks=np.linspace(global_min, global_max, ticks),
            #     format="%.1e",
            # )
            # cb.ax.tick_params(labelsize=self.labelsize)

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Set title
            ax.set_title(title, fontsize=self.fontsize, pad=7)

            # Only show axis labels on the leftmost plot
            if idx == 0 and draw_axis_ticks:
                ax.set_xticks(np.linspace(X.min(), X.max(), 3))
                ax.set_yticks(np.linspace(Y.min(), Y.max(), 4))
                ax.set_xticklabels(
                    [f"{x:.1f}" for x in np.linspace(X.min(), X.max(), 3)]
                )
                ax.set_yticklabels(
                    [f"{y:.1f}" for y in np.linspace(Y.min(), Y.max(), 4)]
                )
                ax.set_xlabel(r"$x_1$→", fontsize=self.fontsize)
                ax.set_ylabel(r"$x_2$→", fontsize=self.fontsize)
                ax.tick_params(axis="both", which="major", labelsize=self.labelsize)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

        self._finalize_figure(fig, filename, img_width, img_height)
