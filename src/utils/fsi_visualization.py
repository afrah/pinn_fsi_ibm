import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import glob
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle


def create_frames(
    df,
    output_dir="animations",
    sample_rate=2,
    cavity_height=6.0,
    plate_height=1.0,
    plate_width=0.06,
    plate_position=10.0,
    add_interface_boundary=False,
):
    """Create animations of pressure and velocity components with sampling to reduce size"""

    if isinstance(df, dict):
        # Convert PyTorch tensors to NumPy arrays
        converted_dict = {}
        for k, v in df.items():
            # Check if it's a tensor
            if hasattr(v, "detach") and hasattr(v, "numpy"):
                # It's a tensor, convert to numpy and flatten
                converted_dict[k] = v.detach().numpy().flatten()
            elif isinstance(v, np.ndarray):
                # It's already a numpy array, just flatten
                converted_dict[k] = v.flatten()
            else:
                # It's something else, keep as is
                converted_dict[k] = v

        # Create the DataFrame from the converted dictionary
        df = pd.DataFrame(converted_dict)

    os.makedirs(output_dir, exist_ok=True)

    # Get all unique time steps
    time_steps = sorted(df["time"].unique())
    # print(f"Found {len(time_steps)} time steps from {time_steps[0]} to {time_steps[-1]}")

    sampled_time_steps = time_steps[::sample_rate]

    # Make sure to include the last timestep (steady state) if it's not already included
    if time_steps[-1] not in sampled_time_steps:
        sampled_time_steps = np.append(sampled_time_steps, time_steps[-1])
        # print(f"Added last timestep (steady state) at t={time_steps[-1]}")

    # print(f"Sampled {len(sampled_time_steps)} time steps for animation")

    fields = [
        ("pressure", "pressure"),
        ("velocity_magnitude", "velocity_magnitude"),
        ("u_x", "u_x"),
        ("v_y", "v_y"),
    ]

    # print("Columns at the begining :", df.columns)
    available_fields = []
    for field, title in fields:
        if field in df.columns:
            available_fields.append((field, title))
            # print(f"Found field: {field}")
        else:
            print(f"Field not found: {field}")

    # Create a series of frames for each field (instead of animation)
    for field_name, field_title in available_fields:
        # print(f"Creating frames for {field_title}...")

        frames_dir = os.path.join(output_dir, field_name)
        os.makedirs(frames_dir, exist_ok=True)

        # Find global min and max values for consistent color scale
        field_min = df[field_name].min()
        field_max = df[field_name].max()
        # print(f"  Value range: {field_min} to {field_max}")

        for i, time in enumerate(sampled_time_steps):
            print(
                f"  Processing time step {time:.2f} ({i + 1}/{len(sampled_time_steps)})"
            )

            time_data = df[df["time"] == time]

            # Skip if no data for this time step
            if len(time_data) == 0:
                print(f"  No data for time step {time:.2f}, skipping")
                continue

            x = time_data["x"].values
            y = time_data["y"].values

            # Create figure with proper dimensions
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))

            tri = Triangulation(x, y)

            z = time_data[field_name].values
            if not np.isfinite(z).all():
                print(
                    f"    Warning: Found non-finite values in {field_name} at time {time}. Replacing them."
                )
                z = np.nan_to_num(z, nan=field_min)  # Replace NaN with min value

            # Set min and max values for contour plot
            min_val = field_min
            max_val = field_max

            # Make sure min and max aren't the same
            if np.isclose(min_val, max_val):
                min_val = min_val - 1e-6 if min_val != 0 else -1e-6
                max_val = max_val + 1e-6 if max_val != 0 else 1e-6

            # Create levels for contour
            levels = np.linspace(min_val, max_val, 60)

            # Create a custom colormap that highlights variations near zero
            if min_val < 0 and max_val > 0:
                # Create a diverging colormap centered at zero
                divnorm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
                contour = ax.tricontourf(
                    tri, z, levels=levels, cmap="jet", norm=divnorm
                )
            else:
                # Use standard jet colormap for all-positive or all-negative data
                contour = ax.tricontourf(
                    tri, z, levels=levels, cmap="jet", vmin=min_val, vmax=max_val
                )

            # Find and highlight the fluid-structure interface
            if add_interface_boundary:
                # Method 1: Use is_interface column if it exists
                if "is_interface" in time_data.columns:
                    # Get all nodes where is_interface is True
                    interface_data = time_data[time_data["is_interface"] == True]

                    if len(interface_data) > 0:
                        # print(f"  Found {len(interface_data)} interface nodes")
                        # Split the interface points into two groups: x = 10.0 and x != 10.0
                        interface_boundary = interface_data[
                            (interface_data["x"] == 9.76631641)
                            | (interface_data["x"] == 10.2853775)
                            | (interface_data["y"] == 1.0008378)
                        ]
                        # print(f"  Found {len(interface_boundary)} interface nodes at x = 10.0")
                        # Highlight the remaining interface points with small white dots
                        ax.scatter(
                            interface_data["x"],
                            interface_data["y"],
                            color="white",
                            alpha=0.8,
                            zorder=10,
                            s=3,
                        )
                        # # Highlight the interface points at x = 10.0 with small black dots
                        # ax.scatter(interface_boundary['x'], interface_boundary['y'],
                        #         color='black', s=1, alpha=0.8, zorder=20)
                else:
                    print(
                        "Warning: 'is_interface' column not found in DataFrame. Using manual method to identify interface."
                    )

            # Set labels
            ax.set_xlabel(r"$x→$", fontsize=18, color="grey")
            ax.set_ylabel(r"$y→$", fontsize=18, color="grey")
            ax.set_title(f"{field_title} - Time: {time * 10:.2f}s")

            # Set tick parameters
            ax.tick_params(axis="both", which="major", labelsize=14, colors="grey")

            # Add colorbar with 4 ticks, reduced gap
            cbar = fig.colorbar(
                contour,
                ax=ax,
                ticks=np.linspace(min_val, max_val, 3),
                fraction=0.18,
                pad=0.02,
            )  # Small padding between plot and colorbar
            # Format colorbar ticks
            cbar.ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.FormatStrFormatter("%.2f")
            )
            cbar.ax.tick_params(labelsize=14, colors="grey")
            cbar.outline.set_visible(False)  # Remove outer border from colorbar

            # Set aspect ratio to equal for proper scaling
            ax.set_aspect("equal")

            # Set axis limits to match the cavity dimensions (similar to Ansys)
            # This keeps the visualization consistent with the problem description
            ax.set_xlim(
                min(x), max(x)
            )  # Based on cavity width from the problem description
            ax.set_ylim(min(y), max(y))

            # Save frame
            filename = os.path.join(frames_dir, f"time_{time:.2f}.png")
            plt.savefig(filename, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        print(f"Saved frames to {frames_dir}")

    print("All frames created successfully!")
    print(f"Individual frames are saved in subdirectories of {output_dir}")


def create_animations_from_existing_frames(
    frames_dirs, output_dir="animations_combined"
):
    """
    Create animations from existing PNG frames in multiple directories without using ffmpeg

    Parameters:
    -----------
    frames_dirs : list of str
        List of directories containing the PNG frames
    output_dir : str
        Directory to save the animations
    """
    os.makedirs(output_dir, exist_ok=True)

    for frames_dir in frames_dirs:
        field_name = os.path.basename(frames_dir)

        png_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

        if not png_files:
            print(f"No PNG files found in {frames_dir}")
            continue

        print(f"Found {len(png_files)} frames for {field_name}")

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_tight_layout(True)
        ax.axis("off")

        def update(frame):
            ax.clear()
            ax.axis("off")

            img = plt.imread(png_files[frame])
            ax.imshow(img)

            return ax

        # Create animation
        print(f"Creating animation for {field_name}...")
        animation = FuncAnimation(
            fig,
            update,
            frames=len(png_files),
            blit=False,
            interval=200,  # 200ms between frames
        )

        gif_path = os.path.join(output_dir, f"{field_name}_animation.gif")

        print(f"Saving animation to {gif_path}...")
        animation.save(
            gif_path,
            writer=PillowWriter(fps=5),
            dpi=100,  # Lower DPI since we're reusing existing images
            savefig_kwargs={"pad_inches": 0},
            progress_callback=lambda i, n: print(f"Saving frame {i + 1}/{n}", end="\r"),
        )
        print(f"\nAnimation saved to {gif_path}")
        plt.close(fig)
