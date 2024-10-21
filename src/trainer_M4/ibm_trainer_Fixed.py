import os
import random
import re
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
from src.trainer_M4.base_trainer import BaseTrainer
from src.nn.nn_functions import MSE
from src.utils.max_eigenvlaue_of_hessian import power_iteration
from src.data.IBM_data_loader import IBM_data_loader

## End: Importing local packages


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_dataloader: IBM_data_loader,
        fluid_model_velocity,
        fluid_model_force,
        interface_model_velocity,
        interface_model_force,
        fluid_optimizer_velocity,
        fluid_optimizer_force,
        interface_optimizer_velocity,
        interface_optimizer_force,
        fluid_scheduler_velocity,
        fluid_scheduler_force,
        interface_scheduler_velocity,
        interface_scheduler_force,
        rank: int,
        config,
    ) -> None:
        super(Trainer, self).__init__(
            train_dataloader,
            fluid_model_velocity,
            fluid_model_force,
            interface_model_velocity,
            interface_model_force,
            fluid_optimizer_velocity,
            fluid_optimizer_force,
            interface_optimizer_velocity,
            interface_optimizer_force,
            fluid_scheduler_velocity,
            fluid_scheduler_force,
            interface_scheduler_velocity,
            interface_scheduler_force,
            rank,
            config,
        )

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            t1 = time.time()

        # print("inside _run_epoch" , epoch)

        losses_velocity, losses_force = self._compute_losses()

        total_loss_fluid_velocity = sum(
            [
                2.0 * ((losses_velocity["left"])),
                # 2.0 * ((losses_velocity["right"])),
                # 2.0 * ((losses_velocity["bottom"])),
                # 2.0 * ((losses_velocity["up"])),
                2.0 * ((losses_velocity["fluid_points_velocity"])),
                4.0 * ((losses_velocity["initial"])),
                0.1 * ((losses_velocity["fluid"])),
                0.5 * ((losses_velocity["vCoupling"])),
            ]
        )
        total_loss_fluid_force = sum(
            [
                2.0 * ((losses_velocity["fluid_points_force"])),
            ]
        )
        total_loss_interface_velocity = sum(
            [
                1.0 * ((losses_force["interface_data_velocity"])),
                1.0 * ((losses_force["interface_initial_velocity"])),
                1.0 * ((losses_velocity["vCoupling"])),
            ]
        )
        total_loss_interface_force = sum(
            [
                1.0 * ((losses_force["interface_data_force"])),
                1.0 * ((losses_force["interface_initial_force"])),
            ]
        )

        self.fluid_optimizer_velocity.zero_grad()
        self.fluid_optimizer_force.zero_grad()
        self.interface_optimizer_velocity.zero_grad()
        self.interface_optimizer_force.zero_grad()
        total_losses = {
            key: losses_velocity.get(key, 0) + losses_force.get(key, 0)
            for key in set(losses_velocity) | set(losses_force)
        }

        if self.rank == 0:
            elapsed_time1 = time.time() - t1

        self.update_epoch_loss(total_losses)

        ### printing
        if self.rank == 0 and epoch % 10000 == 0:

            loss_bc = sum(
                [
                    loss
                    for key, loss in total_losses.items()
                    if key not in ["initial", "fluid"]
                ]
            )
            loss_initial = total_losses["initial"]
            loss_res = total_losses["fluid"]

            self.max_eig_hessian_bc_log.append(
                power_iteration(self.fluid_model_velocity, loss_bc)
            )
            self.max_eig_hessian_res_log.append(
                power_iteration(self.fluid_model_velocity, loss_res)
            )
            self.max_eig_hessian_ic_log.append(
                power_iteration(self.fluid_model_velocity, loss_initial)
            )

        if self.rank == 0:
            t2 = time.time()

        total_loss_fluid_velocity.backward(retain_graph=True)
        total_loss_fluid_force.backward(retain_graph=True)
        total_loss_interface_velocity.backward(retain_graph=True)
        total_loss_interface_force.backward(retain_graph=True)

        self.fluid_optimizer_velocity.step()
        self.fluid_optimizer_force.step()
        self.fluid_scheduler_velocity.step()
        self.fluid_scheduler_force.step()

        self.interface_optimizer_velocity.step()
        self.interface_optimizer_force.step()
        self.interface_scheduler_velocity.step()
        self.interface_scheduler_force.step()

        if self.rank == 0:
            elapsed_time2 = time.time() - t2
            elapsed_time = elapsed_time1 + elapsed_time2

        if self.rank == 0 and epoch % self.config.get("print_every") == 0:
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )

    def _compute_losses(self):

        losses_velocity = self._compute_losses_velocity()
        losses_force = self._compute_losses_force()
        return losses_velocity, losses_force

    def _compute_losses_velocity(self):

        data_mean = self.train_dataloader.fluid_data.mean_x[:3]
        data_std = self.train_dataloader.fluid_data.std_x[:3]
        uvp_domain = self.train_dataloader.fluid_data.uvp_fluid

        # Access the randomly selected minibatch for each tensor
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid.shape[0]
        )
        txy_domain = self.train_dataloader.fluid_data.txy_fluid[batch_indices, :]

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
        txy_interface_data = self.train_dataloader.interface_data.txy_interface[
            batch_indices, :
        ]
        uvp_interface_data = self.train_dataloader.interface_data.uvp_interface[
            batch_indices, :
        ]
        continuity, f_u, f_v = navier_stokes_2D_IBM(
            txy_domain,
            self.fluid_model_velocity,
            self.interface_model_force,
            data_mean,
            data_std,
        )

        # pred_sensors = self.model_force(txy_domain, data_mean, data_std)
        lphy = torch.mean(
            torch.square(continuity) + torch.square(f_u) + torch.square(f_v)
        )

        # lphy = torch.mean(torch.square(f_u) + torch.square(f_v))

        pred_left = self.fluid_model_velocity(txy_left, data_mean, data_std)
        lleft = torch.mean(
            torch.square(pred_left[:, 0] - uvp_left[:, 0])
            + torch.square(pred_left[:, 1] - uvp_left[:, 1])
        )

        pred_right = self.fluid_model_velocity(txy_right, data_mean, data_std)
        lright = torch.mean(
            torch.square(pred_right[:, 0] - uvp_right[:, 0])
            + torch.square(pred_right[:, 1] - uvp_right[:, 1])
        )

        pred_bottom = self.fluid_model_velocity(txy_bottom, data_mean, data_std)
        lbottom = torch.mean(
            torch.square((pred_bottom[:, 0] - uvp_bottom[:, 0]))
            + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 1])))  ## zero  ## zero
        )

        pred_up = self.fluid_model_velocity(txy_up, data_mean, data_std)
        lup = torch.mean(
            torch.square(((pred_up[:, 0] - uvp_up[:, 0])))
            + torch.square(((pred_up[:, 1] - uvp_up[:, 1])))  ## one
        )  ## zero

        pred_initial = self.fluid_model_velocity(txy_initial, data_mean, data_std)
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_initial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 1])
            + torch.square(pred_initial[:, 2] - uvp_initial[:, 2])
        )
        ## presssure training is necessary

        pred_fluid_points_velocity = self.fluid_model_velocity(
            txy_fluid_points, data_mean, data_std
        )
        lfluid_points_velocity = torch.mean(
            torch.square(pred_fluid_points_velocity[:, 0] - uvp_fluid_points[:, 0])
            + torch.square(pred_fluid_points_velocity[:, 1] - uvp_fluid_points[:, 1])
            + torch.square(pred_fluid_points_velocity[:, 2] - uvp_fluid_points[:, 2])
        )

        pred_fluid_points_force = self.fluid_model_force(
            txy_fluid_points, data_mean, data_std
        )
        fluid_points_force = torch.mean(
            torch.square(pred_fluid_points_force[:, 0] - uvp_fluid_points[:, 3])
            + torch.square(pred_fluid_points_force[:, 1] - uvp_fluid_points[:, 4])
        )

        pred_interface_velocity1 = self.fluid_model_velocity(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        ## Add velocity coupling
        pred_interface_velocity2 = self.interface_model_velocity(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        vCoupling = torch.mean(
            torch.square(
                pred_interface_velocity1[:, 0] - pred_interface_velocity2[:, 0]
            )
            + torch.square(
                pred_interface_velocity1[:, 1] - pred_interface_velocity2[:, 1]
            )
        )

        vCoupling1 = torch.mean(
            torch.square(uvp_interface_data[:, 0] - pred_interface_velocity2[:, 0])
            + torch.square(uvp_interface_data[:, 1] - pred_interface_velocity2[:, 1])
        )
        vCoupling2 = torch.mean(
            torch.square(uvp_interface_data[:, 0] - pred_interface_velocity1[:, 0])
            + torch.square(uvp_interface_data[:, 1] - pred_interface_velocity1[:, 1])
        )
        return {
            "left": lleft + lright + lup,
            "right": lright,
            "bottom": lbottom,
            "up": lup,
            "fluid_points_velocity": lfluid_points_velocity,
            "fluid_points_force": fluid_points_force,
            "initial": linitial,
            "fluid": lphy,
            "vCoupling": vCoupling + vCoupling1 + vCoupling2,
        }

    def _compute_losses_force(self):

        # Access the randomly selected minibatch for each tensor

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_interface.shape[0]
        )
        txy_interface_data = self.train_dataloader.interface_data.txy_interface[
            batch_indices, :
        ]
        uvp_interface_data = self.train_dataloader.interface_data.uvp_interface[
            batch_indices, :
        ]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_initial_interface.shape[0]
        )
        txy_intitial = self.train_dataloader.interface_data.txy_initial_interface[
            batch_indices, :
        ]
        uvp_intitial = self.train_dataloader.interface_data.uvp_initial_interface[
            batch_indices, :
        ]

        pred_interface_initial_velocity = self.interface_model_velocity(
            txy_intitial,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )
        linterface_initial_velocity = torch.mean(
            torch.square(pred_interface_initial_velocity[:, 0] - uvp_intitial[:, 0])
            + torch.square(pred_interface_initial_velocity[:, 1] - uvp_intitial[:, 1])
            + torch.square(pred_interface_initial_velocity[:, 2] - uvp_intitial[:, 2])
        )

        pred_interface_initial_force = self.interface_model_force(
            txy_intitial,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )
        linterface_initial_force = torch.mean(
            torch.square(pred_interface_initial_force[:, 0] - uvp_intitial[:, 3])
            + torch.square(pred_interface_initial_force[:, 1] - uvp_intitial[:, 4])
        )

        pred_interface_data_velocity = self.interface_model_velocity(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        linterface_data_velocity = torch.mean(
            torch.square(pred_interface_data_velocity[:, 0] - uvp_interface_data[:, 0])
            + torch.square(
                pred_interface_data_velocity[:, 1] - uvp_interface_data[:, 1]
            )
            + torch.square(
                pred_interface_data_velocity[:, 2] - uvp_interface_data[:, 2]
            )
        )

        pred_interface_data_force = self.interface_model_force(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        linterface_data_force = torch.mean(
            torch.square(pred_interface_data_force[:, 0] - uvp_interface_data[:, 3])
            + torch.square(pred_interface_data_force[:, 1] - uvp_interface_data[:, 4])
        )

        return {
            "interface_data_velocity": linterface_data_velocity,
            "interface_data_force": linterface_data_force,
            "interface_initial_velocity": linterface_initial_velocity,
            "interface_initial_force": linterface_initial_force,
        }
