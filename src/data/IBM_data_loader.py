import pickle
import torch
import numpy as np
import scipy
import scipy.io
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd


def generate_sobol_sequence(low, high, n):
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    
    input_tb = soboleng.draw(n)
    input_np = input_tb.numpy()
    result = np.floor(low + (high - low) * input_np).flatten()
    result = result.astype(int)
    return result

class FluidData:
    def __init__(self, fluid, fluid_points, boundaries, initial, device):
        # Convert to tensors
        self.txy_fluid = torch.tensor(fluid[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_fluid = torch.tensor(fluid[:, 3:8], dtype=torch.float32).to(device)

        self.txy_fluid_points = torch.tensor(
            fluid_points[:, 0:3], dtype=torch.float32
        ).to(device)
        self.uvp_fluid_points = torch.tensor(
            fluid_points[:, 3:8], dtype=torch.float32
        ).to(device)

        self.txy_left = torch.tensor(
            boundaries["left"][:, 0:3], dtype=torch.float32
        ).to(device)
        self.uvp_left = torch.tensor(
            boundaries["left"][:, 3:8], dtype=torch.float32
        ).to(device)

        self.txy_right = torch.tensor(
            boundaries["right"][:, 0:3], dtype=torch.float32
        ).to(device)
        self.uvp_right = torch.tensor(
            boundaries["right"][:, 3:8], dtype=torch.float32
        ).to(device)

        self.txy_bottom = torch.tensor(
            boundaries["bottom"][:, 0:3], dtype=torch.float32
        ).to(device)
        self.uvp_bottom = torch.tensor(
            boundaries["bottom"][:, 3:8], dtype=torch.float32
        ).to(device)

        self.txy_up = torch.tensor(boundaries["up"][:, 0:3], dtype=torch.float32).to(
            device
        )
        self.uvp_up = torch.tensor(boundaries["up"][:, 3:8], dtype=torch.float32).to(
            device
        )

        self.txy_initial = torch.tensor(initial[:, 0:3], dtype=torch.float32).to(device)
        self.uvp_initial = torch.tensor(initial[:, 3:8], dtype=torch.float32).to(device)

        # Calculate mean and std for fluid domain
        self.mean_x = torch.tensor(np.mean(fluid, axis=0), dtype=torch.float32).to(
            device
        )
        self.std_x = torch.tensor(np.std(fluid, axis=0), dtype=torch.float32).to(device)

        # Debugging purpose
        print(f"FluidData: {self.txy_fluid.shape=},  {self.mean_x=}, {self.std_x=}")


class SolidData:
    def __init__(self, solid_points, device):
        # Convert to tensors
        self.txy_solid_points = torch.tensor(
            solid_points[:, 0:3], dtype=torch.float32
        ).to(device)
        self.uvp_solid_points = torch.tensor(
            solid_points[:, 3:8], dtype=torch.float32
        ).to(device)

        # Calculate mean and std for solid domain
        self.mean_x = torch.tensor(
            np.mean(solid_points, axis=0), dtype=torch.float32
        ).to(device)
        self.std_x = torch.tensor(np.std(solid_points, axis=0), dtype=torch.float32).to(
            device
        )

        # Debugging purpose
        print(
            f"SolidData: {self.txy_solid_points.shape=},  {self.mean_x=}, {self.std_x=}"
        )


class FluidSolidInterfaceData:
    def __init__(self, interface, initial_interface, device):
        # Convert to tensors
        self.txy_interface = torch.tensor(interface[:, 0:3], dtype=torch.float32).to(
            device
        )
        self.uvp_interface = torch.tensor(interface[:, 3:8], dtype=torch.float32).to(
            device
        )

        self.txy_initial_interface = torch.tensor(
            initial_interface[:, 0:3], dtype=torch.float32
        ).to(device)
        self.uvp_initial_interface = torch.tensor(
            initial_interface[:, 3:8], dtype=torch.float32
        ).to(device)

        # Calculate mean and std for interface domain
        self.mean_x = torch.tensor(np.mean(interface, axis=0), dtype=torch.float32).to(
            device
        )
        self.std_x = torch.tensor(np.std(interface, axis=0), dtype=torch.float32).to(
            device
        )

        # Debugging purpose
        print(
            f"FluidSolidInterfaceData: {self.txy_interface.shape=},  {self.mean_x=}, {self.std_x=}"
        )


class IBM_data_loader:
    def __init__(self, data_file, device="cpu"):
        [
            fluid,
            fluid_points,
            solid_interface,
            solid_points,
            left,
            right,
            bottom,
            up,
            initial,
            initial_interface,
        ] = process_file(
            data_file,
            training_selection_method,
        )

        # Organize boundary conditions
        boundaries = {"left": left, "right": right, "bottom": bottom, "up": up}

        # Create instances of each data class
        self.fluid_data = FluidData(fluid, fluid_points, boundaries, initial, device)
        self.solid_data = SolidData(solid_points, device)
        self.interface_data = FluidSolidInterfaceData(
            solid_interface, initial_interface, device
        )

    def __getitem__(self):
        return self.fluid_data, self.solid_data, self.interface_data


def remove_initial(data, axis=0):
    initial_idx = np.where(data[:, axis] == data[:, axis].min())[0]
    initial = data[initial_idx, :]
    # data = np.delete(data, initial_idx, axis)
    return data, initial


def subsample(data, proportion, method, seed):
    np.random.seed(seed)
    if method == "Sobol":
        indices = generate_sobol_sequence(
            0, data.shape[0], int(data.shape[0] * proportion)
        )
    else:
        indices = np.random.choice(
            data.shape[0], int(data.shape[0] * proportion), replace=False
        )
    return data[indices, :]


def prepare_training_data(
    dataset_type,
    fluid_sampling_ratio,
    interface_sampling_ratio,
    solid_sampling_ratio,
    left_sampling_ratio,
    right_sampling_ratio,
    bottom_sampling_ratio,
    top_sampling_ratio,
    initial_sampling_ratio,
    training_selection_method,
    device,
    save_dir,
):
    
    # file_path = "./data/Fluid_trainingData.mat"

    SEED = 1234
    if dataset_type == "old":
        processed_dataset_path = "./data/Fluid_trainingData.mat"
        Fluid_data = scipy.io.loadmat(processed_dataset_path)

        # Load data
        fluid = Fluid_data["Fluid_training"]
        interface = pd.read_csv("data/training_dataset/old/all_interface_points_with_normals.csv").to_numpy()#Fluid_data["Solid_interface"]
        solid = Fluid_data["Solid_points"]
 
    fluid = np.concatenate([solid, fluid], 0)
    fluid_points = np.concatenate([solid, fluid], 0)
    # Remove initial points
    fluid_points, fluid_initial1 = remove_initial(fluid_points)
    solid, solid_initial1 = remove_initial(solid)
    # interface, initial_interface = remove_initial(interface)
    initial = np.concatenate([fluid_initial1, solid_initial1], 0)

    # Extract boundary data
    left = fluid[np.where(fluid[:, 1] == fluid[:, 1].min())[0], :]
    right = fluid[np.where(fluid[:, 1] == fluid[:, 1].max())[0], :]
    bottom = fluid[np.where(fluid[:, 2] == fluid[:, 2].min())[0], :]
    up = fluid[np.where(fluid[:, 2] == fluid[:, 2].max())[0], :]

    # Subsample data
    fluid = subsample(fluid, fluid_sampling_ratio, training_selection_method, SEED)

    fluid_points = subsample(
        fluid_points, fluid_sampling_ratio / 3, training_selection_method, SEED
    )
    interface = subsample(
        interface, interface_sampling_ratio, training_selection_method, SEED
    )

    solid = subsample(solid, solid_sampling_ratio, training_selection_method, SEED)
    left = subsample(left, left_sampling_ratio, training_selection_method, SEED)
    right = subsample(right, right_sampling_ratio, training_selection_method, SEED)
    bottom = subsample(bottom, bottom_sampling_ratio, training_selection_method, SEED)
    up = subsample(up, top_sampling_ratio, training_selection_method, SEED)
    initial = subsample(
        initial, initial_sampling_ratio, training_selection_method, SEED
    )
    tensor_data = {
        "fluid": torch.tensor(fluid, dtype=torch.float32, device=device),
        "fluid_points": torch.tensor(fluid_points, dtype=torch.float32, device=device),
        "interface": torch.tensor(interface, dtype=torch.float32, device=device),
        "solid": torch.tensor(solid, dtype=torch.float32, device=device),
        "left": torch.tensor(left, dtype=torch.float32, device=device),
        "right": torch.tensor(right, dtype=torch.float32, device=device),
        "bottom": torch.tensor(bottom, dtype=torch.float32, device=device),
        "up": torch.tensor(up, dtype=torch.float32, device=device),
        "initial": torch.tensor(initial, dtype=torch.float32, device=device),
    }

    os.makedirs(save_dir, exist_ok=True)

    for key, tensor in tensor_data.items():
        tensor_path = os.path.join(save_dir, f"{key}_tensor.pt")
        torch.save(tensor, tensor_path)
        print(f"Saved {key} tensor with shape {tensor.shape} to {tensor_path}")

    return tensor_data


def load_fluid_testing_dataset(dataset_type):
    # file_path = "/home/ubuntu/afrah/datasets/IBM/IB_PINN3.mat"
    old_dataset_path = "./data/IB_PINN3.mat"
    new_dataset_path = "data/processed_dataset/new/combined_fluid_data.csv"

    if dataset_type == "old":
        pkl_path = "data/testing_dataset/old/fluid_testing_tensor.pkl"
        try:
            with open(pkl_path, "rb") as f:
                tensor_data = pickle.load(f)
                return tensor_data
        except FileNotFoundError:
            print(f"File {pkl_path} not found. Loading from dataset...")

        data = h5py.File(old_dataset_path, "r")  # load dataset from matlab
        fluid = np.transpose(
            data["Fluid"], axes=range(len(data.get("Fluid").shape) - 1, -1, -1)
        ).astype(np.float32)

    else:
        pkl_path = "data/testing_dataset/new/fluid_testing_tensor.pkl"
        try:
            with open(pkl_path, "rb") as f:
                tensor_data = pickle.load(f)
                return tensor_data
        except FileNotFoundError:
            print(f"File {pkl_path} not found. Loading from dataset...")

        fluid = pd.read_csv(new_dataset_path).to_numpy()

    tensor_data = torch.tensor(fluid, dtype=torch.float32)
    print(f"Saving tensor to {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(tensor_data, f)

    return tensor_data


def load_solid_testing_dataset(file_path, output_dir="./data/testing_dataset"):
    # file_path = "/home/ubuntu/afrah/datasets/IBM/IB_PINN3.mat"

    try:
        with open(file_path, "rb") as f:
            tensor_data = pickle.load(f)
            return tensor_data
    except FileNotFoundError:
        print(f"File {file_path} not found. Loading from dataset...")

    data = h5py.File(file_path, "r")
    Solid = np.transpose(
        data["Solid"], axes=range(len(data.get("Solid").shape) - 1, -1, -1)
    ).astype(np.float32)
    tensor_data = torch.tensor(Solid, dtype=torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "solid_testing_tensor.pkl")

    print(f"Saving tensor to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(tensor_data, f)

    return tensor_data


def load_training_dataset(
    dataset_path="./data/training_dataset",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load pre-saved training dataset tensors from a specified path.

    Args:
        dataset_path (str): Path to the directory containing saved tensor files
        device (str): Device to load tensors to (cuda or cpu)

    Returns:
        dict: A dictionary of loaded tensor data with keys matching the saved tensor files
    """
    # Ensure the path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # List all .pt files in the directory
    tensor_files = [f for f in os.listdir(dataset_path) if f.endswith("_tensor.pt")]

    # Load tensors
    tensor_data = {}
    for file in tensor_files:
        key = file.replace("_tensor.pt", "")  # Extract key from filename
        tensor_path = os.path.join(dataset_path, file)

        try:
            # Load tensor and move to specified device
            tensor = torch.load(tensor_path, map_location=device)
            tensor_data[key] = tensor
            print(f"Loaded {key} tensor from {tensor_path} with shape {tensor.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return None

    # Verify that tensors were loaded
    if not tensor_data:
        print(f"No tensor files found in {dataset_path}")
        return None

    print(f"Loaded training dataset from {dataset_path} successfully!")
    return tensor_data


def visualize_tensor_datasets(tensor_data, save_dir=None):
    """
    Create scatter plots for different tensor datasets at a specific time step.
    Combines fluid, initial, top, left, right, bottom, and up into one figure,
    leaves interface and solid as separate plots.

    Args:
        tensor_data (dict): Dictionary of tensor datasets
        save_dir (str, optional): Directory to save the plots
    """

    # Create a figure with subplots - now we only need 3 plots
    plt.figure(figsize=(15, 5))

    # First plot: Combined datasets with different colors
    plt.subplot(1, 3, 1)

    # List of datasets to combine
    combine_keys = ["fluid", "initial", "left", "right", "bottom", "up"]
    colors = ["blue", "green", "red", "black", "orange", "cyan"]
    markers = ["*", "+", "o", "x", "D", "^"]
    for i, key in enumerate(combine_keys):
        if key in tensor_data:
            data = tensor_data[key]
            x_data = data[:, 0].cpu().numpy()

            # Use y-coordinate for most datasets
            y_data = (
                data[:, 1].cpu().numpy()
                if key not in ["bottom", "up"]
                else data[:, 2].cpu().numpy()
            )

            # Plot with a different color and add to legend
            plt.scatter(
                x_data,
                y_data,
                color=colors[i],
                label=key,
                marker=markers[i],
                alpha=0.7,
                s=1,
            )

    plt.title("Fluid Datasets with Boundaries")
    plt.xlabel(r"$x→$")
    plt.ylabel(r"$y→$")
    plt.legend()

    # Second plot: Interface (unchanged)
    plt.subplot(1, 3, 2)
    if "interface" in tensor_data:
        data = tensor_data["interface"]
        plt.scatter(
            data[:, 0].cpu().numpy(),
            data[:, 1].cpu().numpy(),
            label="interface",
            alpha=0.7,
            s=1,
            color="blue",
            marker="*",
        )
        plt.title("Interface")

    # Third plot: Solid (unchanged)
    plt.subplot(1, 3, 3)
    if "solid" in tensor_data:
        data = tensor_data["solid"]
        plt.scatter(
            data[:, 0].cpu().numpy(),
            data[:, 1].cpu().numpy(),
            label="solid",
            alpha=0.7,
            s=1,
            color="blue",
            marker="*",
        )
        plt.title("Solid")

    plt.tight_layout()

    if save_dir is None:
        # Create save directory if it doesn't exist
        save_dir = "./data/training_dataset/plots"
        os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(save_dir, f"tensor_datasets_scatter.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved tensor datasets scatter plot to {plot_path}")
    plt.close()