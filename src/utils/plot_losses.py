import os
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from src.utils.colors import model_color


def exponential_moving_average(data, alpha=0.1):
    """Applies Exponential Moving Average (EMA) smoothing to the data."""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def smooth_loss(data, alpha=0.1, window_length=51, polyorder=3):
    """Applies a combination of EMA and Savitzky-Golay smoothing to the data."""
    ema_data = exponential_moving_average(data, alpha=alpha)
    if len(ema_data) < window_length:  # Avoid issues with short data
        window_length = len(ema_data) - 1 if len(ema_data) % 2 == 0 else len(ema_data)
    smoothed_data = savgol_filter(
        ema_data, window_length=window_length, polyorder=polyorder
    )
    return smoothed_data


def spline_smoothing(data, s=0.5):
    """Applies spline smoothing to the data."""
    x = np.arange(len(data))
    spline = UnivariateSpline(x, data, s=s)
    return spline(x)


def smoothed_min(data):
    # Calculate rolling minimum
    window_size = 1000
    min_values = np.minimum.accumulate(data)
    for i in range(window_size, len(data)):
        min_values[i] = min(data[i - window_size + 1 : i + 1])

    # Smooth the minimum values
    window_length = 51  # must be odd and less than data size
    poly_order = 3  # must be less than window_length
    smoothed_min = savgol_filter(min_values, window_length, poly_order)
    return smoothed_min


def plot_loss_history(data_list, save_path=None, y_max=10, y_min=0, figsize=(12, 8)):
    """
    Plot multiple training loss curves in a single figure with logarithmic y-axis.

    Args:
        data_list (list): List of dictionaries containing loss data and plotting parameters
            Each dictionary should have:
                - data: loss values
                - color: line color
                - name: label for the legend
                - alpha: transparency level
                - window: smoothing window size
                - show_avg: whether to show average line
                - show_lower: whether to show lower envelope
        save_path (str, optional): Path to save the figure. Defaults to None.
        y_max (float, optional): Maximum y-axis value. Defaults to None.
        y_min (float, optional): Minimum y-axis value. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 8).
    """
    # Set style to default with white background
    plt.style.use("default")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each loss curve
    for i, entry in enumerate(data_list):
        data = entry["data"]
        name = entry["name"]
        color = entry["color"]
        # Get other parameters or use defaults
        alpha = entry.get("alpha", 1.0)
        window = entry.get("window", 51)
        show_avg = entry.get("show_avg", False)
        show_lower = entry.get("show_lower", False)

        if len(data) > 0:
            print(f"final loss {name}: {data[-1]:.2e}")
        else:
            print(f"Warning: Loss history for {name} is empty.")
            continue

        smooth_alpha = 0.1
        polyorder = 1
        smoothed_data = smooth_loss(data, smooth_alpha, window, polyorder)

        # Plot the smoothed data
        ax.plot(
            np.arange(len(data)),
            smoothed_data,
            label=name,
            color=color,
            alpha=alpha,
            linewidth=1.5,
        )

        # Optionally show average and lower envelope
        if show_avg:
            ax.plot(
                np.arange(len(smoothed_data)),
                smoothed_data,
                color=color,
                linewidth=0.5,
                alpha=0.7,
            )
        if show_lower:
            smoothed_lower = smoothed_min(data)
            ax.plot(
                np.arange(len(smoothed_lower)),
                smoothed_lower,
                color=color,
                linewidth=1.0,
                alpha=0.7,
            )

    # Set up the plot styling
    ax.set_yscale("log")
    ax.set_xlabel("Epochs →", fontsize=12)
    ax.set_ylabel("Loss (log) →", fontsize=12)

    # Add vertical grid lines at regular intervals
    ax.grid(True, which="major", axis="both", linestyle="-", alpha=0.2)
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.1)

    # Set axis limits if provided
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    # Add legend
    legend = ax.legend(loc="upper right", framealpha=0.9)

    # Add thin border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("lightgray")
        spine.set_linewidth(0.5)

    # Save figure if path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_M1_loss_history(
    loss_history, save_path="./", y_max=10, y_min=0, figsize=(12, 8)
):
    bc_loss = np.sum(
        [
            loss_history["left"],
            loss_history["right"],
            loss_history["bottom"],
            loss_history["up"],
        ],
        axis=0,
    )

    # data =  np.sum([loss_history["solid"],  loss_history["fluid_points"]], axis = 0)
    data_list = [
        {
            "data": bc_loss,
            "color": model_color["bc_loss"],
            "name": "BC",
            "alpha": 0.9,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
        },
        {
            "data": loss_history["solid"],
            "color": model_color["data"],
            "name": "Data",
            "alpha": 0.9,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
        },
        {
            "data": loss_history["fluid_points"],
            "color": model_color["data"],
            "name": "Fluid points",
            "alpha": 0.9,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
        },
        {
            "data": loss_history["fluid"],
            "color": model_color["physics"],
            "name": "Physics",
            "alpha": 0.9,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
        },
        {
            "data": loss_history["interface"],
            "color": model_color["interface"],
            "name": "FSI",
            "alpha": 0.9,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
        },
        {
            "data": loss_history["initial"],
            "color": model_color["initial"],
            "name": "Initial",
            "alpha": 0.9,
            "window": 100,
            "show_avg": False,
            "show_lower": False,
        },
        # {
        #     "data": loss_history["fluid_total"],
        #     "color": model_color["total"],
        #     "name": "Fluid total",
        #     "alpha": 0.9,
        #     "window": 100,
        #     "show_avg": False,
        #     "show_lower": False,
        # },
    ]

    plot_loss_history(
        data_list,
        save_path,
        y_max=y_max,
    )
