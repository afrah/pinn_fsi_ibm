import os
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.utils.utils import clear_gpu_memory
from src.nn.pde import navier_stokes_2D_IBM


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Network (PINN) for FSI simulation
    """

    def __init__(
        self,
        fluid_model,
        solid_model,
        training_data,
        learning_rate,
        logger,
        device=None,
        fluid_density=1.0,
        fluid_viscosity=0.2,
        print_every=10,
        save_every=100,
        solver="mlp",
        model="m2",
    ):

        self.device = device
        self.logger = logger
        # Move model to the specified device
        self.fluid_model = fluid_model.to(self.device)
        self.solid_model = solid_model.to(self.device)

        # Store the training data (will move tensors to device when needed)
        self.training_data = training_data

        self.print_every = print_every
        self.save_every = save_every

        self.output_dir = self.logger.get_output_dir()

        self.loss_list = [
            "left",
            "right",
            "bottom",
            "up",
            "solid",
            "fluid_points",
            "fluid",
            "interface",
            "initial",
            "fluid_total",
            "solid_total",
        ]

        self.loss_history = {loss: [] for loss in self.loss_list}

        self.epoch_loss = {
            loss: torch.tensor(0.0, requires_grad=True).to(self.device)
            for loss in self.loss_list
        }

        self.solver = solver
        self.model = model
        self.learning_rate = learning_rate
        self.fluid_optimizer = optim.Adam(
            self.fluid_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        self.solid_optimizer = optim.Adam(
            self.solid_model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        
        if self.solver == "mlp":
            self.fluid_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.fluid_optimizer, "min", patience=5000, factor=0.85
            )
            self.solid_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.solid_optimizer, "min", patience=5000, factor=0.85
            )
        else:
            self.fluid_scheduler = optim.lr_scheduler.StepLR(
                self.fluid_optimizer, step_size=5000, gamma=0.85
            )
            self.solid_scheduler = optim.lr_scheduler.StepLR(
                self.solid_optimizer, step_size=5000, gamma=0.85
            )

    def train(
        self,
        num_epochs=1000,
        batch_size=128,
        data_weight=1.0,
        physics_weight=0.1,
        boundary_weight=1.0,
        fsi_weight=1.0,
        initial_weight=1.0,
    ):
        data_loaders = {}
        self.training_data = {k: v.to(self.device) for k, v in self.training_data.items()}

        try:
            for domain_type, tensor_data in self.training_data.items():
                # tensor_data = tensor_data.to(self.device)

                dataset = TensorDataset(tensor_data)
                current_batch_size = batch_size
                if domain_type != "fluid":
                    current_batch_size = batch_size // 2
                data_loaders[domain_type] = DataLoader(
                    dataset, batch_size=current_batch_size, shuffle=True
                )

            for epoch in range(num_epochs):
                losses_list = {key: 0.0 for key in self.loss_list}

                self.fluid_optimizer.zero_grad()
                self.solid_optimizer.zero_grad()

                for domain_type, loader in data_loaders.items():
                    if domain_type in self.loss_list:
                        domain_batches = 0

                        for batch_idx, (batch_tensor,) in enumerate(loader):
                            # batch_tensor = batch_tensor.to(self.device)
                            time = batch_tensor[:, 0:1]
                            x = batch_tensor[:, 1:2]
                            y = batch_tensor[:, 2:3]

                            inputs = torch.cat([time, x, y], dim=1).squeeze(1)

                            if domain_type in [
                                # "solid",
                                "fluid_points",
                            ]:  # these are non-interface points
                                fluid_outputs = self.fluid_model(inputs)

                                loss = data_weight * torch.mean(
                                    (fluid_outputs[:, 0:1] - batch_tensor[:, 3:4]) ** 2
                                    + (fluid_outputs[:, 1:2] - batch_tensor[:, 4:5])
                                    ** 2
                                    + (fluid_outputs[:, 2:3] - batch_tensor[:, 5:6])
                                    ** 2
                                )
                                losses_list[domain_type] += loss 


                            elif domain_type == "fluid":
                                # NS loss using PDE residuals at non interface points (fluid points)
                                [continuity, f_u, f_v] = navier_stokes_2D_IBM(
                                    self.fluid_model, time, x, y
                                )

                                loss = physics_weight * torch.mean(
                                    continuity**2 + f_u**2 + f_v**2
                                )

                                losses_list[domain_type] += loss

                            elif domain_type == "interface":
                                time.requires_grad_(True)
                                x.requires_grad_(True)
                                y.requires_grad_(True)

                                fluid_outputs = self.fluid_model(
                                    torch.cat([time, x, y], dim=1).squeeze(1)
                                )
                                solid_outputs = self.solid_model(
                                    torch.cat([time, x, y], dim=1).squeeze(1)
                                )
                                p = solid_outputs[:, 2:3]
                                n_x = batch_tensor[:, 8:9]
                                n_y = batch_tensor[:, 9:10]
                                p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[
                                    0
                                ]
                                p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[
                                    0
                                ]
                                p_normal = physics_weight * torch.mean((p_x * n_x + p_y * n_y)** 2)
                                interface_loss1 = physics_weight * torch.mean(
                                    (fluid_outputs[:, 0:1] - solid_outputs[:, 0:1]) ** 2
                                    + (fluid_outputs[:, 1:2] - solid_outputs[:, 1:2])
                                    ** 2
                                    + (fluid_outputs[:, 2:3] - solid_outputs[:, 2:3])
                                    ** 2
                                )

                                interface_loss2 = data_weight * torch.mean(
                                    (solid_outputs[:, 0:1] - batch_tensor[:, 3:4]) ** 2
                                    + (solid_outputs[:, 1:2] - batch_tensor[:, 4:5])
                                    ** 2
                                    + (solid_outputs[:, 2:3] - batch_tensor[:, 5:6])
                                    ** 2
                                )

                                loss = interface_loss1 + interface_loss2+ p_normal

                                losses_list["interface"] += interface_loss1 + p_normal
                                losses_list["solid_total"] += loss

                            elif domain_type in ["left", "right", "up", "bottom"]:
                                fluid_outputs = self.fluid_model(inputs)
                                # left velcoties are nonzero, pressure in nonzero.
                                # right, and pressure are nonzero.
                                # bottom, pressure is zero, velocity is nonzero.
                                # top, pressure is nonzero, velocity is nonzero.
                                loss = boundary_weight * torch.mean(
                                    (fluid_outputs[:, 0:1] - batch_tensor[:, 3:4]) ** 2
                                    + (fluid_outputs[:, 1:2] - batch_tensor[:, 4:5])
                                    ** 2
                                    + (fluid_outputs[:, 2:3] - batch_tensor[:, 5:6])
                                    ** 2
                                )

                                losses_list[domain_type] += loss

                            elif domain_type == "initial":
                                fluid_outputs = self.fluid_model(inputs)
                                loss = initial_weight * torch.mean(
                                    (fluid_outputs[:, 0:1] - batch_tensor[:, 3:4]) ** 2
                                    + (fluid_outputs[:, 1:2] - batch_tensor[:, 4:5])
                                    ** 2
                                    + (fluid_outputs[:, 2:3] - batch_tensor[:, 5:6])
                                    ** 2
                                )

                                losses_list[domain_type] += loss

                            domain_batches += 1

                        # Average the domain loss over batches
                        if domain_batches > 0:
                            losses_list[domain_type] /= domain_batches
                            if domain_type != "solid":
                                losses_list["fluid_total"] += losses_list[domain_type]
                            if domain_type in ["interface"]:
                                losses_list["solid_total"] /= domain_batches

                fluid_total_loss = losses_list["fluid_total"]
                solid_total_loss = losses_list["solid_total"]
                
                fluid_total_loss.backward(retain_graph=True)
                solid_total_loss.backward()
                self.fluid_optimizer.step()
                self.solid_optimizer.step()
                self.fluid_scheduler.step(fluid_total_loss)
                # self.solid_scheduler.step(solid_loss)

                for key in self.loss_list:
                    if key in losses_list:
                        if key not in ["solid", "fluid_points"]:
                            self.loss_history[key].append(losses_list[key].item())
                    else:
                        print(f"Error: Key {key} not found in epoch_losses")

                if (epoch) % self.print_every == 0:
                    if hasattr(self.fluid_scheduler, 'get_last_lr'):
                        lr = self.fluid_scheduler.get_last_lr()[0]
                    else:
                        lr = self.fluid_optimizer.param_groups[0]['lr']
                        
                    self.logger.print(
                        f"Epoch {epoch}/{num_epochs}, "
                        f"Total: {losses_list['fluid_total'].item():.1e}, "
                        f"Data(F&S): {sum(losses_list[b].item() for b in [ 'fluid_points']):.1e}, "
                        f"Physics: {losses_list['fluid'].item():.1e}, "
                        f"Boundary: {sum(losses_list[b].item() for b in ['left', 'right', 'up', 'bottom']):.1e}, "
                        f"FSI: {losses_list['interface'].item():.1e}, "
                        f"Solid: {losses_list['solid_total'].item():.1e}, "
                        f"Initial: {losses_list['initial'].item():.1e}, "
                        f"LR: {lr:.2e}"
                    )

                if (epoch + 1) % self.save_every == 0:
                    self._save_checkpoint(epoch + 1, num_epochs)

            self._save_checkpoint(num_epochs, num_epochs)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("OOM error occurred, attempting to recover...")
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                clear_gpu_memory()
            else:
                raise

        return self.loss_history

    def _save_checkpoint(self, epoch, total_epochs):
        """
        Save checkpoint

        Args:
            epoch: Current epoch number
        """
        model_path = os.path.join(self.output_dir, "model.pth")

        state = {
            "fluid_network": self.fluid_model.network,
            "solid_network": self.solid_model.network,
            "fluid_model_state_dict": self.fluid_model.state_dict(),
            "solid_model_state_dict": self.solid_model.state_dict(),
            "learning_rate": self.learning_rate,
            "fluid_optimizer_state_dict": self.fluid_optimizer.state_dict(),
            "solid_optimizer_state_dict": self.solid_optimizer.state_dict(),
            "fluid_scheduler_state_dict": self.fluid_scheduler.state_dict(),
            # "solid_scheduler_state_dict": self.solid_scheduler.state_dict(),
            "loss_history": self.loss_history,
            "epoch": epoch,
            "model_path": model_path,
            "solver": self.solver,
            "model": self.model,
        }

        torch.save(state, model_path)
        self.logger.print("Final losses:")
        self.logger.print(
            " ".join(
                [
                    "Final %s: %0.3e | " % (key, self.loss_history[key][-1])
                    for key in self.loss_list
                    if key in self.loss_history and self.loss_history[key]
                ]
            )
        )
        self.logger.print(
            f"_save_checkpoint: Epoch {epoch} | Training checkpoint saved at {model_path}"
        )

    def evaluate(self, time, x, y, z):
        outputs = self.fluid_model(torch.cat([time, x, y, z], dim=1).squeeze(1))
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]

        return u, v, p