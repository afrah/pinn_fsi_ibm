import pickle
import torch
import numpy as np
import scipy
import scipy.io

import h5py

dist = "Sobol"


# fluidP = 1
# solid_interfaceP = 1
# solidP = 1
# leftP = 1
# rightP = 1
# bottomP = 1
# upP = 1
# initialP = 1


fluidP = 0.005
solid_interfaceP = 0.05
solidP = 0.04
leftP = 0.1
rightP = 0.1
bottomP = 0.1
upP = 0.1
initialP = 0.05


def generate_sobol_sequence(low, high, n):
    soboleng = torch.quasirandom.SobolEngine(dimension=1)
    bounds = [low, high]

    input_tb = soboleng.draw(n)
    result = np.floor((bounds[0] + (bounds[1] - bounds[0]) * input_tb))
    result = [int(i) for i in result]
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
    def __init__(self, data_file, device):
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
            dist,
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


def load_data(Fluid_data, key):
    return Fluid_data[key]


def remove_initial(data, axis=0):
    initial_idx = np.where(data[:, axis] == data[:, axis].min())[0]
    initial = data[initial_idx, :]
    data = np.delete(data, initial_idx, axis)
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


def process_file(path, dist):
    """
    Process the MATLAB file to extract and subsample fluid and solid data.

    Parameters:
    path (str): Path to the MATLAB file.
    FluidP (float): Proportion of fluid data to retain.
    solid_interfaceP (float): Proportion of solid interface data to retain.
    solidP (float): Proportion of solid points data to retain.
    leftP (float): Proportion of left boundary data to retain.
    rightP (float): Proportion of right boundary data to retain.
    bottomP (float): Proportion of bottom boundary data to retain.
    upP (float): Proportion of up boundary data to retain.
    pInitial (float): Proportion of initial data to retain.
    dist (str): Distribution type for subsampling ('Sobol' or 'random').

    Returns:
    list: Processed data including fluid, solid interface, solid points, and boundary conditions.
    """
    # file_path = "/home/ubuntu/afrah/datasets/IBM/Fluid_trainingData.mat"

    SEED = 1234
    Fluid_data = scipy.io.loadmat(path)

    # Load data
    fluid = load_data(Fluid_data, "Fluid_training")
    interface = load_data(Fluid_data, "Solid_interface")
    solid = load_data(Fluid_data, "Solid_points")

    # fluid = fluid[(fluid[:, -1] == 0) & (fluid[:, -2] == 0)]

    fluid = np.concatenate([solid, fluid], 0)
    fluid_points = np.concatenate([solid, fluid], 0)
    # Remove initial points
    fluid_points, fluid_initial1 = remove_initial(fluid_points)
    solid, solid_initial1 = remove_initial(solid)
    # interface, initial_interface = remove_initial(interface)
    initial_interface = interface
    initial = np.concatenate([fluid_initial1, solid_initial1], 0)

    # Extract boundary data
    left = fluid[np.where(fluid[:, 1] == fluid[:, 1].min())[0], :]
    right = fluid[np.where(fluid[:, 1] == fluid[:, 1].max())[0], :]
    bottom = fluid[np.where(fluid[:, 2] == fluid[:, 2].min())[0], :]
    up = fluid[np.where(fluid[:, 2] == fluid[:, 2].max())[0], :]

    # Subsample data
    fluid = subsample(fluid, fluidP, dist, SEED)
    fluid_points = subsample(fluid_points, fluidP, dist, SEED)
    interface = subsample(interface, solid_interfaceP, dist, SEED)
    solid = subsample(solid, solidP, dist, SEED)
    left = subsample(left, leftP, dist, SEED)
    right = subsample(right, rightP, dist, SEED)
    bottom = subsample(bottom, bottomP, dist, SEED)
    up = subsample(up, upP, dist, SEED)
    initial = subsample(initial, initialP, dist, SEED)

    # fluid = fluid[np.argsort(fluid[:, 0])]
    # fluid_points = fluid_points[np.argsort(fluid_points[:, 0])]
    # interface = interface[np.argsort(interface[:, 0])]
    # solid = solid[np.argsort(solid[:, 0])]
    # left = left[np.argsort(left[:, 0])]
    # right = right[np.argsort(right[:, 0])]
    # bottom = bottom[np.argsort(bottom[:, 0])]
    # up = up[np.argsort(up[:, 0])]

    return [
        fluid,
        fluid_points,
        interface,
        solid,
        left,
        right,
        bottom,
        up,
        initial,
        initial_interface,
    ]


def generate_fluid_testing_dataset(file_path):
    # file_path = "/home/ubuntu/afrah/datasets/IBM/IB_PINN3.mat"

    data = h5py.File(file_path, "r")  # load dataset from matlab
    Fluid = np.transpose(
        data["Fluid"], axes=range(len(data.get("Fluid").shape) - 1, -1, -1)
    ).astype(np.float32)
    tfa = Fluid[:, 0].flatten()[:, None]  # test_data[0]
    xfa = Fluid[:, 1].flatten()[:, None]  # test_data[1]
    yfa = Fluid[:, 2].flatten()[:, None]  # test_data[2]
    ufa = Fluid[:, 3].flatten()[:, None]  # test_data[3]
    vfa = Fluid[:, 4].flatten()[:, None]  # test_data[4]
    pfa = Fluid[:, 5].flatten()[:, None]  # test_data[5]
    fxfa = Fluid[:, 6].flatten()[:, None]  # test_data[5]
    fyfa = Fluid[:, 7].flatten()[:, None]  # test_data[5]

    # with open("IBM_Fluid_Testing.pickle", "wb") as file:
    #     pickle.dump([tfa, xfa, yfa, ufa, vfa, pfa, fxfa, fyfa], file)
    # data.close()

    return [tfa, xfa, yfa, ufa, vfa, pfa, fxfa, fyfa]


def generate_solid_testing_dataset(file_path):
    # file_path = "/home/ubuntu/afrah/datasets/IBM/IB_PINN3.mat"

    data = h5py.File(file_path, "r")  # load dataset from matlab
    Solid = np.transpose(
        data["Solid"], axes=range(len(data.get("Solid").shape) - 1, -1, -1)
    ).astype(np.float32)
    tfa = Solid[:, 0].flatten()[:, None]  # test_data[0]
    xfa = Solid[:, 1].flatten()[:, None]  # test_data[1]
    yfa = Solid[:, 2].flatten()[:, None]  # test_data[2]
    ufa = Solid[:, 3].flatten()[:, None]  # test_data[3]
    vfa = Solid[:, 4].flatten()[:, None]  # test_data[4]
    pfa = Solid[:, 5].flatten()[:, None]  # test_data[5]
    fxfa = Solid[:, 6].flatten()[:, None]  # test_data[5]
    fyfa = Solid[:, 7].flatten()[:, None]  # test_data[5]

    with open("IBM_Solid_Testing.pickle", "wb") as file:
        pickle.dump([tfa, xfa, yfa, ufa, vfa, pfa, fxfa, fyfa], file)

    return [tfa, xfa, yfa, ufa, vfa, pfa, fxfa, fyfa]
