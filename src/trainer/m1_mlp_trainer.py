import torch
import os
import scipy
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.utils.utils import lp_error
from src.utils.logger import Logging
from src.utils.colors import model_color

from src.nn.tanh import MLP
from src.nn.bspline import KAN
from src.utils.utils import clear_gpu_memory
from src.data.IBM_data_loader import prepare_training_data, visualize_tensor_datasets
from src.data.IBM_data_loader import load_fluid_testing_dataset
from src.models.m1_physics import PINNTrainer
from src.utils.plot_losses import plot_M1_loss_history
from src.utils.fsi_visualization import (
    create_frames,
    create_animations_from_existing_frames,
)
from src.data.IBM_data_loader import load_training_dataset
from src.utils.ContourPlotter import ContourPlotter


CHECKPOINT_PATH = "./checkpoints"
logger = Logging(CHECKPOINT_PATH)
model_dirname = logger.get_output_dir()

logger.print(model_dirname)



clear_gpu_memory()
config = {
    "dataset_type": "old",
    "training_selection_method": "Sobol",
    "input_dim": 3,  # (x, y, z, t)
    "hidden_dim": 2,  #######################################
    "hidden_layers_dim": 3,
    "fluid_density": 1.0,
    "fluid_viscosity": 0.01,
    "num_epochs": 4,  #######################################
    "batch_size": 128,
    "learning_rate": 1e-3,
    "data_weight": 2.0,
    "physics_weight": 0.01,
    "boundary_weight": 2.0,
    "fsi_weight": 0.5,
    "initial_weight": 4.0,
    "checkpoint_dir": CHECKPOINT_PATH,
    "resume": None,
    "print_every": 2,  #######################################
    "save_every": 2, #######################################
    "fluid_sampling_ratio": 0.01,
    "interface_sampling_ratio": 0.07,
    "solid_sampling_ratio": 0.01,
    "left_sampling_ratio": 0.1,
    "right_sampling_ratio": 0.15,
    "bottom_sampling_ratio": 0.1,
    "top_sampling_ratio": 0.1,
    "initial_sampling_ratio": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "solver": "mlp",
    "model": "m1",
}


logger.print("Config:")
for key, value in config.items():
    logger.print(f"{key}: {value}")

training_data_path = "./data/training_dataset/old"

training_data = None #load_training_dataset(training_data_path, device=config["device"])

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
if config["solver"] == "mlp":
    fluid_model = MLP(network=fluid_network)
else:
    fluid_model = KAN(fluid_network)

logger.print("Fluid model architecture:")
logger.print(fluid_model)
logger.print(
    f"Number of parameters: {sum(p.numel() for p in fluid_model.parameters())}"
)



trainer = PINNTrainer(
    fluid_model=fluid_model,
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

if model_state["solver"] == "mlp":
    fluid_model = MLP(model_state["fluid_network"]).to(config["device"])
else:
    fluid_model = KAN(model_state["fluid_network"]).to(config["device"])


fluid_model.load_state_dict(model_state["fluid_model_state_dict"])

fluid_model.eval()

logger.print(
    f"Number of parameters: {sum(p.numel() for p in fluid_model.parameters())}"
)



loss_history = model_state["loss_history"]

save_path = os.path.join(logger.get_output_dir(), "loss_history_M1.png")

plot_M1_loss_history(loss_history, save_path, y_max=1000, y_min=0, figsize=(12, 8))




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

processed_dataset_path = "./data/Fluid_trainingData.mat"
Fluid_data = scipy.io.loadmat(processed_dataset_path)

interface = Fluid_data["Solid_interface"]
solid = Fluid_data["Solid_points"]

with torch.no_grad():
    outputs_interface_m1 = np.array(
        fluid_model(
            torch.cat(
                [
                    torch.tensor(solid[:, 0:1], dtype=torch.float32),
                    torch.tensor(solid[:, 1:2], dtype=torch.float32),
                    torch.tensor(solid[:, 2:3], dtype=torch.float32),
                ],
                dim=1,
            ).squeeze(1)
        ).detach().numpy()
    )

u_pred_interface_m1 = outputs_interface_m1[:, 0:1]
v_pred_interface_m1 = outputs_interface_m1[:, 1:2]
p_pred_interface_m1 = outputs_interface_m1[:, 2:3]


logger.print(f"On the interface")

rel_u_l2_error = (
    np.sqrt(
        np.mean((u_pred_interface_m1 - np.array(solid[:, 3:4])) ** 2)
        / np.mean(np.array(interface[:, 3:4]) ** 2)
    )
    * 100
)

rel_v_l2_error = (
    np.sqrt(
        np.mean((v_pred_interface_m1 - np.array(solid[:, 4:5])) ** 2)
        / np.mean(np.array(interface[:, 4:5]) ** 2)
    )
    * 100
)

rel_p_l2_error = (
    np.sqrt(
        np.mean((p_pred_interface_m1 - np.array(solid[:, 5:6])) ** 2)
        / np.mean(np.array(interface[:, 5:6]) ** 2)
    )
    * 100
)


logger.print(f"Relative L2 error for u: {rel_u_l2_error:.2e} %")
logger.print(f"Relative L2 error for v: {rel_v_l2_error:.2e} %")
logger.print(f"Relative L2 error for p: {rel_p_l2_error:.2e} %")



class CavityFlowAnalyzer:
    def __init__(self, logger, device: torch.device):
        self.logger = logger
        self.device = device
        self.results = {}

    def load_data(
        self, data_path: str, tstep: int, xstep: int, ystep: int, skip: int
    ) -> None:
        """Load and preprocess cavity flow data."""
        domain = testing_dataset

        # Reshape and skip data points
        self.time_ = (
            domain[:, 0:1]
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.xfa = (
            domain[:, 1:2]
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.yfa = (
            domain[:, 2:3]
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.ufa = (
            domain[:, 3:4]
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.vfa = (
            domain[:, 4:5]
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.pfa = (
            domain[:, 5:6]
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.u_pred = (
            u_pred
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.v_pred = (
            v_pred
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.p_pred = (
            p_pred
            .reshape(tstep, xstep, ystep)[:, ::skip, ::skip]
            .reshape(-1, 1)
        )
        self.new_shape = (
            domain[:, 0:1].reshape(tstep, xstep, ystep)[:, ::skip, ::skip].shape
        )

    def reshape_results(self) -> Dict[str, np.ndarray]:
        """Reshape all results for visualization."""
        tstep, xstep, ystep = self.new_shape

        # Helper function to convert tensor to numpy if needed
        def to_numpy(data):
            if hasattr(data, 'detach'):  # PyTorch tensor
                return data.detach().cpu().numpy()
            return np.array(data) if not isinstance(data, np.ndarray) else data

        reshaped_data = {
            "tf": to_numpy(self.time_).reshape(tstep, xstep, ystep),
            "xf": to_numpy(self.xfa).reshape(tstep, xstep, ystep),
            "yf": to_numpy(self.yfa).reshape(tstep, xstep, ystep),
            "exact_u": to_numpy(self.ufa).reshape(tstep, xstep, ystep),
            "exact_v": to_numpy(self.vfa).reshape(tstep, xstep, ystep),
            "exact_p": to_numpy(self.pfa).reshape(tstep, xstep, ystep),
            "u_pred": to_numpy(self.u_pred).reshape(tstep, xstep, ystep),
            "v_pred": to_numpy(self.v_pred).reshape(tstep, xstep, ystep),
            "p_pred": to_numpy(self.p_pred).reshape(tstep, xstep, ystep),
            # Calculate errors
            "error_u": to_numpy(np.abs(to_numpy(self.ufa) - to_numpy(self.u_pred))).reshape(tstep, xstep, ystep),
            "error_v": to_numpy(np.abs(to_numpy(self.vfa) - to_numpy(self.v_pred))).reshape(tstep, xstep, ystep),
            "error_p": to_numpy(np.abs(to_numpy(self.pfa) - to_numpy(self.p_pred))).reshape(tstep, xstep, ystep), 
        }
        return reshaped_data

    def plot_time_series_for_variable(
        self, 
        variable_name: str, 
        time_steps: List[int], 
        transpose: bool = True,
        solution_type: str = "exact" , # "exact", "pred", or "error"
        draw_axis_ticks : bool = False
    ):
        reshaped_data = self.reshape_results()
        
        # Get the appropriate data key
        if solution_type == "exact":
            data_key = f"exact_{variable_name}"
            title_prefix = f"Exact solution ${variable_name}(x)$"
        elif solution_type == "pred":
            data_key = f"{variable_name}_pred"
            title_prefix = f"PINN prediction $\\hat{{{variable_name}}}(x)$"
        elif solution_type == "error":
            data_key = f"error_{variable_name}"
            title_prefix = f"PINN error ${variable_name}$"
        else:
            raise ValueError("solution_type must be 'exact', 'pred', or 'error'")
        
        variable_data = reshaped_data[data_key]
        
        if transpose:
            variable_data = variable_data.transpose(0, 2, 1)
        
        visualization_data = [variable_data for _ in time_steps]
        
        titles = [f"$t = {t}$" for t in time_steps]
        nrows_ncols = (1, len(time_steps))
        
        plotter = ContourPlotter(fontsize=8, labelsize=7, axes_pad=0.15)
        
        if transpose:
            xf_data = reshaped_data["yf"][0, 0, :]
            yf_data = reshaped_data["xf"][0, :, 0]
        else:
            xf_data = reshaped_data["xf"][0, :, 0]
            yf_data = reshaped_data["yf"][0, 0, :]
        
        plotter.draw_contourf_time_series(
            reshaped_data["tf"][:, 0, 0],
            xf_data,
            yf_data,
            visualization_data,
            titles=titles,
            nrows_ncols=nrows_ncols,
            time_steps=time_steps,
            model_dirname=self.logger.get_output_dir(),
            img_width=1.5 * len(time_steps),  # Adjust width based on number of time steps
            img_height=6,
            ticks=3,
            variable_name=variable_name,
            solution_type=solution_type,
            draw_axis_ticks= draw_axis_ticks
        )


analyzer = CavityFlowAnalyzer(logger, config["device"])

data_path = "./data/IB_PINN3.mat"
analyzer.load_data(data_path, tstep=101, xstep=102, ystep=102, skip=1)



reshaped_data = analyzer.reshape_results()
titles = [
    "Exact solution $u(x)$",
    "PINN prediction $\\hat{u}(x)$",
    "PINN error",
    "Exact solution $v(x)$",
    "PINN prediction $\\hat{v}(x)$",
    "PINN error",
    "Exact solution $p(x)$",
    "PINN prediction $\\hat{p}(x)$",
    "PINN error",
]
nrows_ncols = (3, 3)
values = [99]
xref = 1
yref = 1
model_dirname = logger.get_output_dir()
img_width = 30
img_height = 6
ticks = 3

visualization_data = [
    reshaped_data["exact_u"].transpose(0, 2, 1),  # swap x and y dimensions
    reshaped_data["u_pred"].transpose(0, 2, 1),
    reshaped_data["error_u"].transpose(0, 2, 1),
    reshaped_data["exact_v"].transpose(0, 2, 1),
    reshaped_data["v_pred"].transpose(0, 2, 1),
    reshaped_data["error_v"].transpose(0, 2, 1),
    reshaped_data["exact_p"].transpose(0, 2, 1),
    reshaped_data["p_pred"].transpose(0, 2, 1),
    reshaped_data["error_p"].transpose(0, 2, 1),
]

plotter = ContourPlotter(fontsize=8, labelsize=7, axes_pad=0.65)

plotter.draw_contourf_regular_2D(
    reshaped_data["tf"][:, 0, 0],
    reshaped_data["yf"][0, 0, :],
    reshaped_data["xf"][0, :, 0],
    visualization_data,
    titles=titles,
    nrows_ncols=nrows_ncols,
    time_steps=values,
    xref=xref,
    yref=yref,
    model_dirname=model_dirname,
    img_width=img_width,
    img_height=img_height,
    ticks=ticks,
)

analyzer = CavityFlowAnalyzer(logger, config["device"])
data_path = "./data/IB_PINN3.mat"
analyzer.load_data(data_path, tstep=101, xstep=102, ystep=102, skip=1)

# Plot time series for u variable (exact solution)
time_steps = [0, 10, 50, 60, 80, 99]  # Specify which time steps to plot
analyzer.plot_time_series_for_variable(
    variable_name="u", 
    time_steps=time_steps,
    transpose=True,
    solution_type="exact",
    draw_axis_ticks = False
)

# Plot time series for u variable (predictions)
analyzer.plot_time_series_for_variable(
    variable_name="u", 
    time_steps=time_steps,
    transpose=True,
    solution_type="pred",
    draw_axis_ticks = False

)

# Plot time series for u variable (errors)
analyzer.plot_time_series_for_variable(
    variable_name="u", 
    time_steps=time_steps,
    transpose=True,
    solution_type="error",
    draw_axis_ticks = True
)

# You can also plot for v and p variables:
analyzer.plot_time_series_for_variable("v", time_steps, transpose=True, solution_type="exact")
analyzer.plot_time_series_for_variable("v", time_steps, transpose=True, solution_type="pred")
analyzer.plot_time_series_for_variable("v", time_steps, transpose=True, solution_type="error")

analyzer.plot_time_series_for_variable("p", time_steps, transpose=True, solution_type="exact")
analyzer.plot_time_series_for_variable("p", time_steps, transpose=True, solution_type="pred")
analyzer.plot_time_series_for_variable("p", time_steps, transpose=True, solution_type="error")

