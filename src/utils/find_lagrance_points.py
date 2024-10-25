import sys
import numpy as np
import scipy.io


## Start: Importing local packages. As I don't know how to run it as module
## with torchrun  (e.g., python -m trainer.Coronary_ddp_trainer)

from src.utils.euler_lagrange_force_diff import find_sum_gaussian_time_tensors

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# file_path = "/home/vlq26735/saiful/afrah/datasets/IBM/IB_PINN3.mat"

# h=  0.498715
# k=  0.6851750000000001
# rx=  0.209685
# ry=  0.21008499999999997
DATASET_PATH = "./data/Fluid_trainingData.mat"

Fluid_data = scipy.io.loadmat(DATASET_PATH)

# Load data
fluid = Fluid_data["Fluid_training"]
interface = Fluid_data["Solid_interface"]
solid = Fluid_data["Solid_points"]

fluid_solid = np.concatenate((fluid, solid), axis=0)
fluid_solid = fluid_solid[np.argsort(fluid_solid[:, 0])]
fluid_solid_size = fluid_solid.shape[0]
interface_size = interface.shape[0]

fluid_solid = np.hstack((fluid_solid, np.arange(fluid_solid_size).reshape(-1, 1)))

interface = interface[np.argsort(interface[:, 0])]
interface = np.hstack((interface, np.arange(interface_size).reshape(-1, 1)))


tstep = 101

fluid_n_data = int(fluid_solid_size / tstep)
interface_n_data = int(interface_size / tstep)

fluid_times = fluid_solid[:, 0].reshape(tstep, fluid_n_data)

fluid_x = fluid_solid[:, 1].reshape(tstep, fluid_n_data)
fluid_y = fluid_solid[:, 2].reshape(tstep, fluid_n_data)
fluid_u = fluid_solid[:, 3].reshape(tstep, fluid_n_data)
fluid_v = fluid_solid[:, 4].reshape(tstep, fluid_n_data)
fluid_p = fluid_solid[:, 5].reshape(tstep, fluid_n_data)
fluid_fx = fluid_solid[:, 6].reshape(tstep, fluid_n_data)
fluid_fy = fluid_solid[:, 7].reshape(tstep, fluid_n_data)
fluid_index = fluid_solid[:, 8].reshape(tstep, fluid_n_data)


interface_times = interface[:, 0].reshape(tstep, interface_n_data)
interface_x = interface[:, 1].reshape(tstep, interface_n_data)
interface_y = interface[:, 2].reshape(tstep, interface_n_data)
interface_u = interface[:, 3].reshape(tstep, interface_n_data)
interface_v = interface[:, 4].reshape(tstep, interface_n_data)
interface_p = interface[:, 5].reshape(tstep, interface_n_data)
interface_fx = interface[:, 6].reshape(tstep, interface_n_data)
interface_fy = interface[:, 7].reshape(tstep, interface_n_data)
interface_index = interface[:, 8].reshape(tstep, interface_n_data)

import json

# Assuming lagrance_pints_list is defined in the loop as in your code
lagrance_pints_list = []

# Example loop (replace this with your actual data processing loop)
for index in range(tstep):
    int_times = interface_times[index, :]
    int_x = interface_x[index, :]
    int_y = interface_y[index, :]
    int_u = interface_u[index, :]
    int_v = interface_v[index, :]
    int_p = interface_p[index, :]
    int_fx = interface_fx[index, :]
    int_fy = interface_fy[index, :]
    int_index = interface_index[index, :]
    fs_time = fluid_times[index, :]
    fs_x = fluid_x[index, :]
    fs_y = fluid_y[index, :]
    fs_u = fluid_u[index, :]
    fs_v = fluid_v[index, :]
    fs_p = fluid_p[index, :]
    fs_fx = fluid_fx[index, :]
    fs_fy = fluid_fy[index, :]
    fs_index = fluid_index[index, :]

    for interfacePt_index in range(len(int_times)):
        inter_pt_index = int_index[interfacePt_index]
        euler_neighbors = find_sum_gaussian_time_tensors(
            int_x[interfacePt_index],
            int_y[interfacePt_index],
            fs_time,
            fs_x,
            fs_y,
            fs_u,
            fs_v,
            fs_p,
            fs_fx,
            fs_fy,
            fs_index,
            sigma=1 / tstep,
        )

        neighbor = {
            "time_step": index,
            "lagrace_point_index": int(inter_pt_index),
            "lagrace_point": [
                (int_times[interfacePt_index]),
                (int_x[interfacePt_index]),
                (int_y[interfacePt_index]),
                (int_u[interfacePt_index]),
                (int_v[interfacePt_index]),
                (int_p[interfacePt_index]),
                (int_fx[interfacePt_index]),
                (int_fy[interfacePt_index]),
            ],
            "euler_neighbors": euler_neighbors,
        }

        lagrance_pints_list.append(neighbor)

# Save lagrance_pints_list to JSON after processing
with open(f"lagrance_points.json", "w") as json_file:
    json.dump(lagrance_pints_list, json_file, indent=2)
