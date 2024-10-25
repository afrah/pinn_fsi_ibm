import os
import pickle
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pandas as pd

from src.utils.euler_lagrange_force_diff import get_lagracian_eulerian_pt

from src.nn.pde import navier_stokes_2D_IBM
from src.trainer_M2.base_trainer import BaseTrainer
from src.nn.nn_functions import MSE
from src.utils.max_eigenvlaue_of_hessian import power_iteration
from src.data.IBM_data_loader import IBM_data_loader

## End: Importing local packages


def get_lagracian_points():

    with open("./data/lagrace_points.pkl", "rb") as f:
        lagrace_points = pickle.load(f)

    with open("./data/eulerian_points.pkl", "rb") as f:
        eulerian_points = pickle.load(f)

    with open("./data/spatial_weights.pkl", "rb") as f:
        spatial_weights = pickle.load(f)
    return lagrace_points, eulerian_points, spatial_weights


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_dataloader: IBM_data_loader,
        model_fluid: nn.Module,
        model_force: nn.Module,
        optimizer_fluid: optim.Optimizer,
        optimizer_force: optim.Optimizer,
        scheduler_fluid,
        scheduler_force,
        rank: int,
        config,
    ) -> None:
        super(Trainer, self).__init__(
            train_dataloader,
            model_fluid,
            model_force,
            optimizer_fluid,
            optimizer_force,
            scheduler_fluid,
            scheduler_force,
            rank,
            config,
        )

        self.lagrace_points, self.eulerian_points, self.spatial_weights = (
            get_lagracian_points()
        )
        self.lagrace_points = self.lagrace_points.to(self.rank)
        self.eulerian_points = self.eulerian_points.to(self.rank)
        self.spatial_weights = self.spatial_weights.to(self.rank)

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            t1 = time.time()

        # print("inside _run_epoch" , epoch)

        losses_velocity, losses_force = self._compute_losses()

        total_loss_velocity = sum(
            [
                1.0 * ((losses_velocity["left"])),
                # 2.0 * ((losses_velocity["right"])),
                # 2.0 * ((losses_velocity["bottom"])),
                # 1.0 * ((losses_velocity["up"])),
                2.0 * ((losses_velocity["fluid_points"])),
                1.0 * ((losses_velocity["initial"])),
                0.01 * ((losses_velocity["fluid"])),
                1.0 * ((losses_velocity["vCoupling"])),
            ]
        )
        total_loss_force = sum(
            [
                1.0 * ((losses_force["lint_pts"])),
                1.0 * ((losses_velocity["vCoupling"])),
                1.0 * ((losses_force["int_initial"])),
            ]
        )

        self.optimizer_fluid.zero_grad()
        self.optimizer_force.zero_grad()

        total_losses = {
            key: losses_velocity.get(key, 0) + losses_force.get(key, 0)
            for key in set(losses_velocity) | set(losses_force)
        }

        if self.rank == 0:
            elapsed_time1 = time.time() - t1

        self.update_epoch_loss(total_losses)

        ### printing
        # if self.rank == 0 and epoch % 10000 == 0:

        #     loss_bc = sum(
        #         [
        #             loss
        #             for key, loss in total_losses.items()
        #             if key not in ["initial", "fluid"]
        #         ]
        #     )
        #     loss_initial = total_losses["initial"]
        #     loss_res = total_losses["fluid"]

        #     self.max_eig_hessian_bc_log.append(
        #         power_iteration(self.model_fluid, loss_bc)
        #     )
        #     self.max_eig_hessian_res_log.append(
        #         power_iteration(self.model_fluid, loss_res)
        #     )
        #     self.max_eig_hessian_ic_log.append(
        #         power_iteration(self.model_fluid, loss_initial)
        #     )

        if self.rank == 0:
            t2 = time.time()

        total_loss_velocity.backward(retain_graph=True)
        total_loss_force.backward(retain_graph=True)

        self.optimizer_fluid.step()
        self.scheduler_fluid.step()

        self.optimizer_force.step()
        self.scheduler_force.step()

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
        continuity, f_u, f_v = navier_stokes_2D_IBM(
            self.train_dataloader.fluid_data.txy_fluid[batch_indices, :],
            self.model_fluid,
            self.model_force,
            data_mean,
            data_std,
        )
        lphy = torch.mean(
            torch.square(continuity) + torch.square(f_u) + torch.square(f_v)
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid_points.shape[0]
        )
        txy_fluid_points = self.train_dataloader.fluid_data.txy_fluid_points[
            batch_indices, :
        ]
        uvp_fluid_points = self.train_dataloader.fluid_data.uvp_fluid_points[
            batch_indices, :
        ]

        pred_fluid_points = self.model_fluid(txy_fluid_points, data_mean, data_std)
        fluid_points = torch.mean(
            torch.square(pred_fluid_points[:, 0] - uvp_fluid_points[:, 0])
            + torch.square(pred_fluid_points[:, 1] - uvp_fluid_points[:, 1])
            + torch.square(pred_fluid_points[:, 2] - uvp_fluid_points[:, 2])
            # + torch.square(pred_sensors[:, 3] - uvp_fluid_points[:, 3])
            # + torch.square(pred_sensors[:, 4] - uvp_fluid_points[:, 4])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.solid_data.txy_solid_points.shape[0]
        )
        uvp_solid_points = self.train_dataloader.solid_data.uvp_solid_points[
            batch_indices, :
        ]

        pred_solid = self.model_fluid(
            self.train_dataloader.solid_data.txy_solid_points[batch_indices, :],
            self.train_dataloader.solid_data.mean_x[:3],
            self.train_dataloader.solid_data.std_x[:3],
        )
        solid_points = torch.mean(
            torch.square(pred_solid[:, 0] - uvp_solid_points[:, 0])
            + torch.square(pred_solid[:, 1] - uvp_solid_points[:, 1])
            + torch.square(pred_solid[:, 2] - uvp_solid_points[:, 2])
            + torch.square(pred_solid[:, 3] - uvp_solid_points[:, 3])
            + torch.square(pred_solid[:, 4] - uvp_solid_points[:, 4])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_left.shape[0]
        )
        uvp_left = self.train_dataloader.fluid_data.uvp_left[batch_indices, :]

        pred_left = self.model_fluid(
            self.train_dataloader.fluid_data.txy_left[batch_indices, :],
            data_mean,
            data_std,
        )
        lleft = torch.mean(
            torch.square(pred_left[:, 0] - uvp_left[:, 0])
            + torch.square(pred_left[:, 1] - uvp_left[:, 1])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_right.shape[0]
        )
        uvp_right = self.train_dataloader.fluid_data.uvp_right[batch_indices, :]

        pred_right = self.model_fluid(
            self.train_dataloader.fluid_data.txy_right[batch_indices, :],
            data_mean,
            data_std,
        )
        lright = torch.mean(
            torch.square(pred_right[:, 0] - uvp_right[:, 0])
            + torch.square(pred_right[:, 1] - uvp_right[:, 1])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_bottom.shape[0]
        )

        uvp_bottom = self.train_dataloader.fluid_data.uvp_bottom[batch_indices, :]
        pred_bottom = self.model_fluid(
            self.train_dataloader.fluid_data.txy_bottom[batch_indices, :],
            data_mean,
            data_std,
        )
        lbottom = torch.mean(
            torch.square((pred_bottom[:, 0] - uvp_bottom[:, 0]))
            + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 1])))
        )
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_up.shape[0]
        )

        uvp_up = self.train_dataloader.fluid_data.uvp_up[batch_indices, :]
        pred_up = self.model_fluid(
            self.train_dataloader.fluid_data.txy_up[batch_indices, :],
            data_mean,
            data_std,
        )
        lup = torch.mean(
            torch.square(((pred_up[:, 0] - uvp_up[:, 0])))
            + torch.square(((pred_up[:, 1] - uvp_up[:, 1])))
        )  ## zero

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_initial.shape[0]
        )
        uvp_initial = self.train_dataloader.fluid_data.uvp_initial[batch_indices, :]

        pred_initial = self.model_fluid(
            self.train_dataloader.fluid_data.txy_initial[batch_indices, :],
            data_mean,
            data_std,
        )
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_initial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 1])
            + torch.square(pred_initial[:, 2] - uvp_initial[:, 2])
        )
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_interface.shape[0]
        )
        txy_interface_data = self.train_dataloader.interface_data.txy_interface[
            batch_indices, :
        ]

        pred_fluid1 = self.model_fluid(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        ## Add velocity coupling
        pred_fluid2 = self.model_force(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        vCoupling = torch.mean(
            torch.square(pred_fluid1[:, 0] - pred_fluid2[:, 0])
            + torch.square(pred_fluid1[:, 1] - pred_fluid2[:, 1])
        )

        return {
            "left": lleft + lright + lbottom + lup,
            "right": lright,
            "bottom": lbottom,
            "up": lup,
            "fluid_points": solid_points + fluid_points,
            "initial": linitial,
            "fluid": lphy,
            "vCoupling": vCoupling,
        }

    def _compute_losses_force(self):

        # Access the randomly selected minibatch for each tensor

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_interface.shape[0]
        )
        pred_interface = self.model_force(
            self.train_dataloader.interface_data.txy_interface[batch_indices, :],
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )
        uvp_interface_data = self.train_dataloader.interface_data.uvp_interface[
            batch_indices, :
        ]
        interface_data = torch.mean(
            torch.square(pred_interface[:, 0] - uvp_interface_data[:, 0])
            + torch.square(pred_interface[:, 1] - uvp_interface_data[:, 1])
            + torch.square(pred_interface[:, 2] - uvp_interface_data[:, 2])
            # + torch.square(pred_interface[:, 3] - uvp_interface_data[:, 3])
            # + torch.square(pred_interface[:, 4] - uvp_interface_data[:, 4])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_initial_interface.shape[0]
        )

        pred_lagracian_pt = self.model_force(
            self.lagrace_points[:, :3],
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        lagracian_loss = torch.mean(
            torch.square(pred_lagracian_pt[:, 0] - self.lagrace_points[:, 3])
            + torch.square(pred_lagracian_pt[:, 1] - self.lagrace_points[:, 4])
            + torch.square(pred_lagracian_pt[:, 2] - self.lagrace_points[:, 5])
            + torch.square(pred_lagracian_pt[:, 3] - self.lagrace_points[:, 3])
            + torch.square(pred_lagracian_pt[:, 4] - self.lagrace_points[:, 4])
        )
        pred_eulerian_points = self.model_fluid(
            self.eulerian_points[:, :3],
            self.train_dataloader.fluid_data.mean_x[:3],
            self.train_dataloader.fluid_data.std_x[:3],
        )
        fluid_loss = torch.mean(
            torch.square(pred_eulerian_points[:, 0] - self.eulerian_points[:, 3])
            + torch.square(pred_eulerian_points[:, 1] - self.eulerian_points[:, 4])
            + torch.square(pred_eulerian_points[:, 2] - self.eulerian_points[:, 5])
            + torch.square(pred_eulerian_points[:, 3] - self.eulerian_points[:, 6])
            + torch.square(pred_eulerian_points[:, 4] - self.eulerian_points[:, 7])
        )

        sum_euelrain_fx = sum(self.spatial_weights[:, -2] * pred_eulerian_points[:, -2])
        sum_euelrain_fy = sum(self.spatial_weights[:, -1] * pred_eulerian_points[:, -1])

        sum_lagracian_fx = sum(pred_lagracian_pt[:, -2])
        sum_lagracian_fy = sum(pred_lagracian_pt[:, -1])

        coupling = torch.mean(
            torch.square(sum_euelrain_fx - sum_lagracian_fx)
            + torch.square(sum_euelrain_fy - sum_lagracian_fy)
        )

        return {
            "lint_pts": interface_data,
            "int_initial": coupling + lagracian_loss + fluid_loss,
        }
