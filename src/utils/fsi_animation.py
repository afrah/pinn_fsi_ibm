import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle
import os
from scipy.spatial import cKDTree
import shutil
import imageio


COLOR_SOLID = "#a0a0a0"
COLOR_FLUID = "#5e3c99"
COLOR_INTERFACE = "#daa520"


def create_alternative_animation(
    fluid_data,
    solid_data,
    interface_data=None,
    output_file="fsi_animation_alt.gif",
    fps=10,
    dpi=150,
    speedup=1,
):
    """
    Alternative implementation that uses individual frames instead of FuncAnimation

    Parameters:
    -----------
    fluid_data : pandas.DataFrame
        Combined fluid data
    solid_data : pandas.DataFrame
        Combined solid data
    interface_data : pandas.DataFrame, optional
        Pre-identified interface points
    output_file : str
        Path to save the animation
    fps : int
        Frames per second
    dpi : int
        Resolution
    speedup : int
        Factor to speed up animation
    """
    # Get unique timesteps
    fluid_times = np.sort(fluid_data["time"].unique())
    solid_times = np.sort(solid_data["time"].unique())

    # Find common timesteps
    common_times = np.intersect1d(fluid_times, solid_times)

    # Apply speedup by selecting subset of timesteps
    timesteps = common_times[::speedup]

    print(f"Creating animation with {len(timesteps)} frames (alternative method)")

    # Create directory for individual frames
    frames_dir = "animation_frames"
    os.makedirs(frames_dir, exist_ok=True)

    frame_files = []

    # Create individual frames
    for i, t in enumerate(timesteps):
        print(f"Processing frame {i + 1}/{len(timesteps)}, time = {t:.2f}")

        # Create a new figure for each frame
        fig, ax = plt.subplots(figsize=(4, 4))

        # Filter data for current timestep
        fluid_t = fluid_data[fluid_data["time"] == t]
        solid_t = solid_data[solid_data["time"] == t]

        # Plot fluid points with velocity vectors
        if len(fluid_t) > 0:
            # Subsample fluid points for clarity
            sample_size = min(500, len(fluid_t))
            fluid_sample = fluid_t.sample(sample_size)

            # Plot velocity vectors
            ax.quiver(
                fluid_sample["x"],
                fluid_sample["y"],
                fluid_sample["u"],
                fluid_sample["v"],
                color=COLOR_FLUID,
                alpha=0.6,
                label="Fluid Velocity",
                scale_units="xy",
                angles="xy",
                # scale=1,
                # width=0.0012,
                headwidth=4,
                headlength=6,
            )

        # Plot solid points
        ax.scatter(
            solid_t["x"], solid_t["y"], s=1, color=COLOR_SOLID, alpha=0.7, label="Solid"
        )

        # Plot interface points
        if interface_data is not None and t in interface_data["time"].values:
            interface_t = interface_data[
                (interface_data["time"] == t) & (interface_data["is_interface"])
            ]
            ax.scatter(
                interface_t["x"],
                interface_t["y"],
                s=1,
                color=COLOR_INTERFACE,
                alpha=0.9,
                label="Interface",
            )

        if t == timesteps[0]:
            ax.legend(loc="upper right")
            ax.set_xlabel("x→", fontsize=12)
            ax.set_ylabel("y→", fontsize=12)

        # Set plot limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # Add title with current time
        time_step = t * 10
        ax.set_title(f"t = {time_step:.1f}s", fontsize=14)

        # Add axis labels and legend

        # ax.legend(loc="upper right")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save frame
        frame_file = os.path.join(frames_dir, f"frame_{time_step:.1f}.png")
        plt.savefig(frame_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        frame_files.append(frame_file)

    try:
        print(f"Creating GIF from {len(frame_files)} frames...")

        with imageio.get_writer(output_file, mode="I", fps=fps, loop=0) as writer:
            for frame_file in frame_files:
                image = imageio.imread(frame_file)
                writer.append_data(image)

        print(f"Animation saved successfully to {output_file}!")

    except ImportError:
        print(f"Individual frames are saved in {frames_dir}")

    return output_file


def main():
    # Load the combined fluid and solid data
    fluid_file = "./data/processed_dataset/old/fluid.csv"
    solid_file = "./data/processed_dataset/old/solid.csv"
    interface_file = (
        "./data/processed_dataset/old/interface.csv"  # From previous script
    )

    # Create output directory - use relative path with ./ prefix
    animations_dir = "./animations"
    os.makedirs(animations_dir, exist_ok=True)

    if not os.path.exists(fluid_file) or not os.path.exists(solid_file):
        print(
            f"Error: Data files not found. Make sure {fluid_file} and {solid_file} exist."
        )
        return

    print(f"Loading fluid data from {fluid_file}...")
    fluid_data = pd.read_csv(fluid_file)

    print(f"Loading solid data from {solid_file}...")
    solid_data = pd.read_csv(solid_file)

    # Load interface data if available
    interface_data = None
    if os.path.exists(interface_file):
        print(f"Loading interface data from {interface_file}...")
        interface_data = pd.read_csv(interface_file)
    else:
        print(
            "Interface data file not found. Will identify interface points on the fly."
        )

    # Try alternative method instead
    print("\nUsing alternative animation method...")
    create_alternative_animation(
        fluid_data,
        solid_data,
        interface_data,
        output_file=f"{animations_dir}/fsi_animation_alt.gif",
        fps=10,
        dpi=100,
        speedup=2,
    )

    # # Create simple animation - most reliable method
    # print("\nCreating simple animation...")
    # create_simple_animation(
    #     fluid_data,
    #     solid_data,
    #     interface_data,
    #     output_file=f"{animations_dir}/fsi_simple.gif",
    #     fps=5,
    #     dpi=100,
    #     speedup=2,
    # )

    print("Animation creation completed!")

    # shutil.rmtree("./animation_frames")
    # shutil.rmtree("./simple_frames")


if __name__ == "__main__":
    main()
