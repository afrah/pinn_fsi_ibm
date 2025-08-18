import sys
import os
import torch
import numpy as np
import torch.nn as nn
import scipy
import pandas as pd
import matplotlib.pyplot as plt


from src.utils.utils import lp_error
from src.utils.logger import Logging
from src.utils.colors import model_color
from src.utils.draw_contour_plts import (
    plot_time_profile_regular_data_IBM_uvp,
    draw_contour_plts,
    draw_interface_contour_plts,
)
from src.nn.tanh import MLP
from src.nn.bspline import KAN
from src.utils.utils import clear_gpu_memory
from src.data.IBM_data_loader import prepare_training_data, visualize_tensor_datasets
from src.data.IBM_data_loader import load_fluid_testing_dataset
from src.models.m2 import PINNTrainer
from src.utils.plot_losses import plot_M1_loss_history
from src.utils.fsi_visualization import (
    create_frames,
    create_animations_from_existing_frames,
)
from src.data.IBM_data_loader import load_training_dataset, load_fluid_testing_dataset

CHECKPOINT_PATH = "./checkpoints"
logger = Logging(CHECKPOINT_PATH)
model_dirname = logger.get_output_dir()

logger.print(model_dirname)

clear_gpu_memory()
config = {
    "dataset_type": "old",
    "training_selection_method": "Sobol",
    "input_dim": 3,
    "hidden_dim": 350,
    "hidden_layers_dim": 3,
    "fluid_density": 1.0,
    "fluid_viscosity": 0.01,
    "num_epochs": 20000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "data_weight": 2.0,
    "physics_weight": 0.15,
    "boundary_weight": 2.0,
    "fsi_weight": 0.5,
    "initial_weight": 2.0,
    "checkpoint_dir": "./checkpoints",
    "resume": None,
    "print_every": 500,
    "save_every": 1000,
    "fluid_sampling_ratio": 0.005,
    "interface_sampling_ratio": 0.02,
    "solid_sampling_ratio": 0.0,
    "left_sampling_ratio": 0.1,
    "right_sampling_ratio": 0.15,
    "bottom_sampling_ratio": 0.1,
    "top_sampling_ratio": 0.1,
    "initial_sampling_ratio": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "solver": "mlp",
    "model": "m2",
}

logger.print("Config:")
for key, value in config.items():
    logger.print(f"{key}: {value}")

training_data_path = "./data/training_dataset/old"

training_data = load_training_dataset(training_data_path, device=config["device"])

if training_data is None:
    training_data = prepare_training_data(
        config["dataset_type"],
        fluid_sampling_ratio=config["fluid_sampling_ratio"],
        interface_sampling_ratio=config["interface_sampling_ratio"],
        solid_sampling_ratio=config["solid_sampling_ratio"],
        left_sampling_ratio=config["left_sampling_ratio"],
        right_sampling_ratio=config["right_sampling_ratio"],
        bottom_sampling_ratio=config["bottom_sampling_ratio"],
        top_sampling_ratio=config["top_sampling_ratio"],
        initial_sampling_ratio=config["initial_sampling_ratio"],
        training_selection_method=config["training_selection_method"],
        device=config["device"],
        save_dir=training_data_path,
    )


visualize_tensor_datasets(training_data, save_dir=training_data_path)

fluid_network = (
    [config["input_dim"]] + [config["hidden_dim"]] * config["hidden_layers_dim"] + [3]
)
solid_network = (
    [config["input_dim"]] + [config["hidden_dim"]] * config["hidden_layers_dim"] + [3]
)

if config["solver"] == "mlp":
    fluid_model = MLP(network=fluid_network)
    solid_model = MLP(network=solid_network)
else:
    fluid_model = KAN(network=fluid_network)
    solid_model = KAN(network=solid_network)

logger.print(f"Fluid model architecture:")
logger.print(fluid_model)
logger.print(
    f"Number of parameters: {sum(p.numel() for p in fluid_model.parameters())}"
)

logger.print(f"Solid model architecture:")
logger.print(solid_model)
logger.print(
    f"Number of parameters: {sum(p.numel() for p in solid_model.parameters())}"
)

trainer = PINNTrainer(
    fluid_model=fluid_model,
    solid_model=solid_model,
    training_data=training_data,
    learning_rate=config["learning_rate"],
    logger=logger,
    device=config["device"],
    fluid_density=config["fluid_density"],
    fluid_viscosity=config["fluid_viscosity"],
    print_every=config["print_every"],
    save_every=config["save_every"],
    solver=config["solver"],
    model=config["model"],
)


loss_history = trainer.train(
    num_epochs=config["num_epochs"],
    batch_size=config["batch_size"],
    data_weight=config["data_weight"],
    physics_weight=config["physics_weight"],
    boundary_weight=config["boundary_weight"],
    fsi_weight=config["fsi_weight"],
    initial_weight=config["initial_weight"],
)

model_path = os.path.join(trainer.logger.get_output_dir(), "model.pth")
model_state = torch.load(model_path)


if config["solver"] == "mlp":
    fluid_model = MLP(model_state["fluid_network"]).to("cpu")
else:
    fluid_model = KAN(model_state["fluid_network"]).to("cpu")


fluid_model.load_state_dict(model_state["fluid_model_state_dict"])

fluid_model.eval()

logger.print(
    f"Number of parameters: {sum(p.numel() for p in fluid_model.parameters())}"
)

loss_history = model_state["loss_history"]

save_path = os.path.join(logger.get_output_dir(), "loss_history_M1.png")

plot_M1_loss_history(loss_history, save_path, y_max=20, y_min=0, figsize=(12, 8))

animations_reference_dir = os.path.join(logger.get_output_dir(), "animations_reference")


try:
    testing_dataset = load_fluid_testing_dataset(config["dataset_type"])
except Exception as e:
    logger.print(f"Testing dataset not found.")
    raise e

skip = 1
time = testing_dataset[:, 0:1][::skip]
x = testing_dataset[:, 1:2][::skip]
y = testing_dataset[:, 2:3][::skip]
u_ref = testing_dataset[:, 3:4][::skip]
v_ref = testing_dataset[:, 4:5][::skip]
p_ref = testing_dataset[:, 5:6][::skip]


with torch.no_grad():
    outputs = fluid_model(torch.cat([time, x, y], dim=1).squeeze(1))

u_pred = outputs[:, 0:1]
v_pred = outputs[:, 1:2]
p_pred = outputs[:, 2:3]


velocity_magnitude_pred = torch.sqrt(u_pred**2 + v_pred**2)

rel_u_l2_error = (
    torch.sqrt(torch.mean((u_pred - u_ref) ** 2) / torch.mean(u_ref**2)) * 100
)
rel_v_l2_error = (
    torch.sqrt(torch.mean((v_pred - v_ref) ** 2) / torch.mean(v_ref**2)) * 100
)
rel_p_l2_error = (
    torch.sqrt(torch.mean((p_pred - p_ref) ** 2) / torch.mean(p_ref**2)) * 100
)

logger.print(f"Relative L2 error for u: {rel_u_l2_error:.2e} %")
logger.print(f"Relative L2 error for v: {rel_v_l2_error:.2e} %")
logger.print(f"Relative L2 error for p: {rel_p_l2_error:.2e} %")

animations_pred_dir = os.path.join(logger.get_output_dir(), "animations_pred")

prediction_df = pd.DataFrame(
    {
        "time": time.detach().numpy().flatten(),
        "x": x.detach().numpy().flatten(),
        "y": y.detach().numpy().flatten(),
        "pressure": p_pred.detach().numpy().flatten(),
        "u_x": u_pred.detach().numpy().flatten(),
        "v_y": v_pred.detach().numpy().flatten(),
        "velocity_magnitude": np.sqrt(
            u_pred.detach().numpy().flatten() ** 2
            + v_pred.detach().numpy().flatten() ** 2
        ),
    }
)

create_frames(prediction_df, output_dir=animations_pred_dir, sample_rate=10)


testing_df = pd.DataFrame(
    {
        "time": time.detach().numpy().flatten(),
        "x": x.detach().numpy().flatten(),
        "y": y.detach().numpy().flatten(),
        "pressure": p_ref.detach().numpy().flatten(),
        "u_x": u_ref.detach().numpy().flatten(),
        "v_y": v_ref.detach().numpy().flatten(),
        "velocity_magnitude": np.sqrt(
            u_ref.detach().numpy().flatten() ** 2
            + v_ref.detach().numpy().flatten() ** 2
        ),
    }
)


create_frames(testing_df, output_dir=animations_reference_dir, sample_rate=10)

create_animations_from_existing_frames(
    frames_dirs=[
        os.path.join(animations_reference_dir, "pressure"),
        os.path.join(animations_reference_dir, "u_x"),
        os.path.join(animations_reference_dir, "v_y"),
        os.path.join(animations_reference_dir, "velocity_magnitude"),
    ],
    output_dir=os.path.join(animations_reference_dir, "gif"),
)

create_animations_from_existing_frames(
    frames_dirs=[
        os.path.join(animations_pred_dir, "pressure"),
        os.path.join(animations_pred_dir, "u_x"),
        os.path.join(animations_pred_dir, "v_y"),
        os.path.join(animations_pred_dir, "velocity_magnitude"),
    ],
    output_dir=os.path.join(animations_pred_dir, "gif"),
)
