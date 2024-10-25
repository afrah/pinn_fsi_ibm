from math import e

import numpy as np
import json
import torch
import pickle


def gaussian_kernel_2d(lx, ly, ex, ey, sigma):
    """2D Gaussian kernel with normalization using tensors."""
    weight_x = (1 / (2 * np.pi * sigma**2)) * np.exp(-((ex - lx) ** 2) / (2 * sigma**2))
    weight_y = (1 / (2 * np.pi * sigma**2)) * np.exp(-((ey - ly) ** 2) / (2 * sigma**2))
    return weight_x, weight_y


def time_smoothing_kernel(t_x, t_y, sigma_time):
    """Time smoothing kernel based on time differences using tensors."""
    return np.exp(-((t_x - t_y) ** 2) / (2 * sigma_time**2))


def find_sum_gaussian_time_tensors(
    lagrancian_pt_x,
    lagrancian_pt_y,
    eulerian_pts_time,
    eulerian_pts_x,
    eulerian_pts_y,
    eulerian_pts_u,
    eulerian_pts_v,
    eulerian_pts_p,
    eulerian_pts_fx,
    eulerian_pts_fy,
    fs_index,
    sigma=0.2,
    sigma_time=0.1,
):

    # For each Lagrangian point, calculate the weighted sum of Eulerian forces
    neighbors_list = []
    tstep = 101
    for index in range(len(eulerian_pts_x)):
        # Spatial Gaussian kernel for each Eulerian-Lagrangian pair
        eulerian_pt_t = eulerian_pts_time[index]
        eulerian_pt_x = eulerian_pts_x[index]
        eulerian_pt_y = eulerian_pts_y[index]
        eulerian_pt_u = eulerian_pts_u[index]
        eulerian_pt_v = eulerian_pts_v[index]
        eulerian_pt_p = eulerian_pts_p[index]
        eulerian_pt_fx = eulerian_pts_fx[index]
        eulerian_pt_fy = eulerian_pts_fy[index]

        pint_index = fs_index[index]
        spatial_weight_x, spatial_weight_y = gaussian_kernel_2d(
            lagrancian_pt_x,
            lagrancian_pt_y,
            eulerian_pt_x,
            eulerian_pt_x,
            sigma=1.0 / tstep,
        )

        weights = (spatial_weight_x**2 + spatial_weight_y**2) ** 0.5
        if weights > 1.0e-6:
            neighbor = {
                "euler_point_index": pint_index,
                "eulerian_point": [
                    eulerian_pt_t,
                    eulerian_pt_x,
                    eulerian_pt_y,
                    eulerian_pt_u,
                    eulerian_pt_v,
                    eulerian_pt_p,
                    eulerian_pt_fx,
                    eulerian_pt_fy,
                ],
                "spatial_weight": [spatial_weight_x, spatial_weight_y],
            }
            neighbors_list.append(neighbor)

    return neighbors_list


def get_lagracian_eulerian_pt(lagrance_points_data, device):
    # Load the JSON file

    # Initialize lists for Lagrangian points and Eulerian points
    all_lagrangian_points = []
    all_eulerian_points = []

    # Select every 5th element from lagrance_points_data
    selected_lagrance_points_data = lagrance_points_data[::400]

    # Iterate over each Lagrangian point entry in the loaded data
    for entry_index, entry in enumerate(selected_lagrance_points_data):
        # Append Lagrangian point to the list
        all_lagrangian_points.append(
            {
                "time_step": entry["time_step"],
                "lagrace_point_index": entry["lagrace_point_index"],
                "lagrace_point": entry["lagrace_point"],
            }
        )

        # Append each Eulerian neighbor to the Eulerian points list with reference to the Lagrangian point and entry index
        for neighbor in entry["euler_neighbors"]:
            all_eulerian_points.append(
                {
                    "entry_index": entry_index,  # Index of the Lagrangian entry in the original list
                    "lagrace_point_index": entry[
                        "lagrace_point_index"
                    ],  # Index of the Lagrangian point
                    "euler_point_index": neighbor["euler_point_index"],
                    "eulerian_point": neighbor["eulerian_point"],
                    "spatial_weight": neighbor["spatial_weight"],
                }
            )
    lagrace_points = [entry["lagrace_point"] for entry in all_lagrangian_points]
    eulerian_points = [entry["eulerian_point"] for entry in all_eulerian_points]
    spatial_weights = [entry["spatial_weight"] for entry in all_eulerian_points]

    lagrace_points = np.array(lagrace_points)
    eulerian_points = np.array(eulerian_points)
    spatial_weights = np.array(spatial_weights)

    lagrace_points = torch.tensor(lagrace_points, device=device, dtype=torch.float32)
    eulerian_points = torch.tensor(eulerian_points, device=device, dtype=torch.float32)
    spatial_weights = torch.tensor(spatial_weights, device=device, dtype=torch.float32)
    with open(
        "/home/ubuntu/afrah/code/pinn_fsi_ibm/data/lagrace_points.pkl", "wb"
    ) as f:
        pickle.dump(lagrace_points, f)

    with open(
        "/home/ubuntu/afrah/code/pinn_fsi_ibm/data/eulerian_points.pkl", "wb"
    ) as f:
        pickle.dump(eulerian_points, f)

    with open(
        "/home/ubuntu/afrah/code/pinn_fsi_ibm/data/spatial_weights.pkl", "wb"
    ) as f:
        pickle.dump(spatial_weights, f)

    return lagrace_points, eulerian_points, spatial_weights
