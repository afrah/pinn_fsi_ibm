import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))


## Start: Importing local packages. As I don't know how to run it as module
## with torchrun  (e.g., python -m trainer.Coronary_ddp_trainer)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nn.pde import navier_stokes_2D_IBM
from src.trainer_M1.base_trainer import BaseTrainer
from src.nn.nn_functions import MSE
from src.utils.max_eigenvlaue_of_hessian import power_iteration

## End: Importing local packages


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_dataloader,
        fluid_model: nn.Module,
        optimizer_fluid: optim.Optimizer,
        fluid_scheduler,
        rank: int,
        config,
    ) -> None:
        super(Trainer, self).__init__(
            train_dataloader,
            fluid_model,
            optimizer_fluid,
            fluid_scheduler,
            rank,
            config,
        )

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)
        if self.rank == 0:
            t1 = time.time()
        # print("inside _run_epoch" , epoch)

        bclosses = self._compute_losses()

        total_loss = sum(
            [
                2.0 * ((bclosses["left"])),
                # 2.0 * ((losses_velocity["right"])),
                # 2.0 * ((losses_velocity["bottom"])),
                # 2.0 * ((losses_velocity["up"])),
                4.0 * ((bclosses["fluid_points"])),
                4.0 * ((bclosses["initial"])),
                0.01 * ((bclosses["fluid"])),
                1.0 * ((bclosses["lint_pts"])),
                1.0 * ((bclosses["int_initial"])),
                1.0 * ((bclosses["vCoupling"])),
            ]
        )

        self.optimizer_fluid.zero_grad()

        if self.rank == 0:
            elapsed_time1 = time.time() - t1

        self.update_epoch_loss(bclosses)

        ### printing
        if self.rank == 0 and epoch % 10000 == 0:

            loss_bc = sum(bclosses[key] for key in self.epoch_loss.keys())
            loss_initial = bclosses["initial"]
            loss_res = bclosses["fluid"]

            self.max_eig_hessian_bc_log.append(
                power_iteration(self.fluid_model, loss_bc)
            )
            self.max_eig_hessian_res_log.append(
                power_iteration(self.fluid_model, loss_res)
            )
            self.max_eig_hessian_ic_log.append(
                power_iteration(self.fluid_model, loss_initial)
            )

        if self.rank == 0:
            t2 = time.time()

        total_loss.backward()
        self.optimizer_fluid.step()
        self.fluid_scheduler.step()

        if self.rank == 0:
            elapsed_time2 = time.time() - t2
            elapsed_time = elapsed_time1 + elapsed_time2

        if self.rank == 0 and epoch % self.config.get("print_every") == 0:
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )

    def _compute_losses(self):
        data_mean = self.train_dataloader.fluid_data.mean_x[:3]
        data_std = self.train_dataloader.fluid_data.std_x[:3]

        # Access the randomly selected minibatch for each tensor
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid.shape[0]
        )
        txy_fluid = self.train_dataloader.fluid_data.txy_fluid[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid_points.shape[0]
        )
        txy_fluid_points = self.train_dataloader.fluid_data.txy_fluid_points[
            batch_indices, :
        ]
        uvp_fluid_points = self.train_dataloader.fluid_data.uvp_fluid_points[
            batch_indices, :
        ]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_left.shape[0]
        )
        txy_left = self.train_dataloader.fluid_data.txy_left[batch_indices, :]
        uvp_left = self.train_dataloader.fluid_data.uvp_left[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_right.shape[0]
        )
        txy_right = self.train_dataloader.fluid_data.txy_right[batch_indices, :]
        uvp_right = self.train_dataloader.fluid_data.uvp_right[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_bottom.shape[0]
        )
        txy_bottom = self.train_dataloader.fluid_data.txy_bottom[batch_indices, :]
        uvp_bottom = self.train_dataloader.fluid_data.uvp_bottom[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_up.shape[0]
        )
        txy_up = self.train_dataloader.fluid_data.txy_up[batch_indices, :]
        uvp_up = self.train_dataloader.fluid_data.uvp_up[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_initial.shape[0]
        )
        txy_initial = self.train_dataloader.fluid_data.txy_initial[batch_indices, :]
        uvp_initial = self.train_dataloader.fluid_data.uvp_initial[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_interface.shape[0]
        )
        txy_interface = self.train_dataloader.interface_data.txy_interface[
            batch_indices, :
        ]
        uvp_interface = self.train_dataloader.interface_data.uvp_interface[
            batch_indices, :
        ]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_initial_interface.shape[0]
        )
        txy_initial_interface = (
            self.train_dataloader.interface_data.txy_initial_interface[batch_indices, :]
        )
        uvp_initial_interface = (
            self.train_dataloader.interface_data.uvp_initial_interface[batch_indices, :]
        )
        # print("self.train_dataloader.fluid_data.txy_fluid shape : " , self.train_dataloader.fluid_data.txy_fluid.shape)
        continuity, f_u, f_v = navier_stokes_2D_IBM(
            txy_fluid,
            self.fluid_model,
            self.fluid_model,
            data_mean,
            data_std,
        )

        lphy = torch.mean(
            torch.square(continuity) + torch.square(f_u) + torch.square(f_v)
        )

        pred_left = self.fluid_model(txy_left, data_mean, data_std)
        lleft = torch.mean(
            torch.square(pred_left[:, 0] - uvp_left[:, 0])
            + torch.square(pred_left[:, 1] - uvp_left[:, 1])
        )

        pred_right = self.fluid_model(txy_right, data_mean, data_std)
        lright = torch.mean(
            torch.square(pred_right[:, 0] - uvp_right[:, 0])
            + torch.square(pred_right[:, 1] - uvp_right[:, 1])
        )

        pred_bottom = self.fluid_model(txy_bottom, data_mean, data_std)
        lbottom = torch.mean(
            torch.square((pred_bottom[:, 0] - uvp_bottom[:, 0]))
            + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 1])))  ## zero  ## zero
        )

        pred_up = self.fluid_model(txy_up, data_mean, data_std)
        lup = torch.mean(
            torch.square(((pred_up[:, 0] - uvp_up[:, 0])))
            + torch.square(((pred_up[:, 1] - uvp_up[:, 1])))  ## one
        )  ## zero

        pred_initial = self.fluid_model(txy_initial, data_mean, data_std)
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_initial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 1])
            + torch.square(pred_initial[:, 2] - uvp_initial[:, 2])
            # + torch.square(pred_initial[:, 3] - uvp_initial[:, 3])
            # + torch.square(pred_initial[:, 4] - uvp_initial[:, 4])
        )
        ## presssure training is necessary

        pred_sensors = self.fluid_model(txy_fluid_points, data_mean, data_std)
        lsensors = torch.mean(
            torch.square(pred_sensors[:, 0] - uvp_fluid_points[:, 0])
            + torch.square(pred_sensors[:, 1] - uvp_fluid_points[:, 1])
            + torch.square(pred_sensors[:, 2] - uvp_fluid_points[:, 2])
            + torch.square(pred_sensors[:, 3] - uvp_fluid_points[:, 3])
            + torch.square(pred_sensors[:, 4] - uvp_fluid_points[:, 4])
        )
        pred_fluid1 = self.fluid_model(
            txy_interface,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        vCoupling2 = torch.mean(
            torch.square(uvp_interface[:, 0] - pred_fluid1[:, 0])
            + torch.square(uvp_interface[:, 1] - pred_fluid1[:, 1])
        )

        pred_initial = self.fluid_model(
            txy_initial_interface,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )
        initial_interface = torch.mean(
            # torch.square(pred_initial[:, 0] - uvp_intitial[:, 0])
            # + torch.square(pred_initial[:, 1] - uvp_intitial[:, 1])
            # + torch.square(pred_initial[:, 2] - uvp_intitial[:, 2])
            torch.square(pred_initial[:, 3] - uvp_initial_interface[:, 3])
            + torch.square(pred_initial[:, 4] - uvp_initial_interface[:, 4])
        )

        pred_interface = self.fluid_model(
            txy_interface,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        linterface = torch.mean(
            torch.square(pred_interface[:, 0] - uvp_interface[:, 0])
            + torch.square(pred_interface[:, 1] - uvp_interface[:, 1])
            + torch.square(pred_interface[:, 2] - uvp_interface[:, 2])
            + torch.square(pred_interface[:, 3] - uvp_interface[:, 3])
            + torch.square(pred_interface[:, 4] - uvp_interface[:, 4])
        )
        return {
            "left": lleft + lright + lup,
            "right": lright,
            "bottom": lbottom,
            "up": lup,
            "fluid_points": lsensors,
            "initial": linitial,
            "fluid": lphy,
            "vCoupling":  vCoupling2,
            "lint_pts": linterface,
            "int_initial": initial_interface,
        }
