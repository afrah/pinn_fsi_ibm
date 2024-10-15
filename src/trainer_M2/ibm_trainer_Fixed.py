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
from src.trainer_M2.base_trainer import BaseTrainer
from src.nn.nn_functions import MSE
from src.utils.max_eigenvlaue_of_hessian import power_iteration
from src.data.IBM_data_loader import IBM_data_loader

## End: Importing local packages


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

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            start_time = time.time()

        # print("inside _run_epoch" , epoch)

        losses_velocity, losses_force = self._compute_losses()

        total_loss_velocity = sum(
            [
                2.0 * ((losses_velocity["left"])),
                2.0 * ((losses_velocity["right"])),
                2.0 * ((losses_velocity["bottom"])),
                2.0 * ((losses_velocity["up"])),
                4.0 * ((losses_velocity["fluid_points"])),
                4.0 * ((losses_velocity["initial"])),
                0.01 * ((losses_velocity["fluid"])),
            ]
        )
        total_losses_force = sum(
            [
                # 2.0 * ((losses_force["left"])),
                # 2.0 * ((losses_force["right"])),
                # 2.0 * ((losses_force["bottom"])),
                # 2.0 * ((losses_force["up"])),
                1.0 * ((losses_force["fluid_points"])),
                1.0 * ((losses_force["initial"])),
            ]
        )
        if self.rank == 0:
            elapsed_time = time.time() - start_time

        self.optimizer_fluid.zero_grad()
        self.optimizer_force.zero_grad()
        total_losses = {
            key: losses_velocity.get(key, 0) + losses_force.get(key, 0)
            for key in set(losses_velocity) | set(losses_force)
        }
        self.update_epoch_loss(total_losses)

        ### printing
        if self.rank == 0 and epoch % self.config.get("print_every") == 0:

            loss_bc = sum(
                [
                    ((total_losses["left"])),
                    ((total_losses["right"])),
                    ((total_losses["bottom"])),
                    ((total_losses["up"])),
                    ((total_losses["fluid_points"])),
                ]
            )
            loss_initial = total_losses["initial"]
            loss_res = total_losses["fluid"]

            self.max_eig_hessian_bc_log.append(
                power_iteration(self.model_fluid, loss_bc)
            )
            self.max_eig_hessian_res_log.append(
                power_iteration(self.model_fluid, loss_res)
            )
            self.max_eig_hessian_ic_log.append(
                power_iteration(self.model_fluid, loss_initial)
            )
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )
        total_loss_velocity.backward()
        total_losses_force.backward()

        self.optimizer_fluid.step()
        self.scheduler_fluid.step()

        self.optimizer_force.step()
        self.scheduler_force.step()

    def _compute_losses(self):

        losses_velocity = self._compute_losses_velocity()
        losses_force = self._compute_losses_force()
        return losses_velocity, losses_force

    def _compute_losses_velocity(self):
        (
            data_mean,
            data_std,
            txy_domain,
            uvp_domain,
            txy_sensors,
            txy_left,
            txy_right,
            txy_bottom,
            txy_up,
            txy_initial,
            uvp_sensors,
            uvp_left,
            uvp_right,
            uvp_bottom,
            uvp_up,
            uvp_initial,
        ) = self.get_mini_batch_data(self.train_dataloader.fluid_data)
        # print("txy_domain shape : " , txy_domain.shape)
        continuity, f_u, f_v = navier_stokes_2D_IBM(
            txy_domain, self.model_fluid, data_mean, data_std
        )

        # pred_sensors = self.model_force(txy_domain, data_mean, data_std)
        lphy = torch.mean(
            torch.square(continuity) + torch.square(f_u) + torch.square(f_v)
        )

        # lphy = torch.mean(torch.square(f_u) + torch.square(f_v))

        pred_left = self.model_fluid(txy_left, data_mean, data_std)
        lleft = torch.mean(
            torch.square(pred_left[:, 0] - uvp_left[:, 0])
            + torch.square(pred_left[:, 1] - uvp_left[:, 1])
        )

        pred_right = self.model_fluid(txy_right, data_mean, data_std)
        lright = torch.mean(
            torch.square(pred_right[:, 0] - uvp_right[:, 0])
            + torch.square(pred_right[:, 1] - uvp_right[:, 1])
        )

        pred_bottom = self.model_fluid(txy_bottom, data_mean, data_std)
        lbottom = torch.mean(
            torch.square((pred_bottom[:, 0] - uvp_bottom[:, 0]))
            + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 1])))  ## zero  ## zero
        )

        pred_up = self.model_fluid(txy_up, data_mean, data_std)
        lup = torch.mean(
            torch.square(((pred_up[:, 0] - uvp_up[:, 0])))
            + torch.square(((pred_up[:, 1] - uvp_up[:, 1])))  ## one
        )  ## zero

        pred_initial = self.model_fluid(txy_initial, data_mean, data_std)
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_initial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 1])
            + torch.square(pred_initial[:, 2] - uvp_initial[:, 2])
        )
        ## presssure training is necessary

        pred_sensors = self.model_fluid(txy_sensors, data_mean, data_std)
        lsensors = torch.mean(
            torch.square(pred_sensors[:, 0] - uvp_sensors[:, 0])
            + torch.square(pred_sensors[:, 1] - uvp_sensors[:, 1])
            + torch.square(pred_sensors[:, 2] - uvp_sensors[:, 2])
        )
        # p ## nonzero
        # +  (pred_sensors[:, 3], uvp_sensors[:, 3])## nonzero
        # +  (pred_sensors[:, 4], uvp_sensors[:, 4])## nonzero
        ## presssure training is necessary

        return {
            "left": lleft,
            "right": lright,
            "bottom": lbottom,
            "up": lup,
            "fluid_points": lsensors,
            "initial": linitial,
            "fluid": lphy,
        }

    def _compute_losses_force(self):

        # Access the randomly selected minibatch for each tensor
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.solid_data.txy_solid_points.shape[0]
        )
        txy_domain = self.train_dataloader.solid_data.txy_solid_points[batch_indices, :]
        uvp_domain = self.train_dataloader.solid_data.uvp_solid_points[batch_indices, :]

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_left.shape[0]
        )
        txy_left = self.train_dataloader.fluid_data.txy_left[batch_indices, :]
        uvp_left = self.train_dataloader.fluid_data.txy_left[batch_indices, :]

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
        uvp_initial = self.train_dataloader.fluid_data.txy_initial[batch_indices, :]

        pred_left = self.model_force(txy_left, data_mean, data_std)
        lleft = torch.mean(
            torch.square(pred_left[:, 0] - uvp_left[:, 3])
            + torch.square(pred_left[:, 1] - uvp_left[:, 4])
        )

        pred_right = self.model_force(txy_right, data_mean, data_std)
        lright = torch.mean(
            torch.square(pred_right[:, 0] - uvp_right[:, 3])
            + torch.square(pred_right[:, 1] - uvp_right[:, 4])
        )

        pred_bottom = self.model_force(txy_bottom, data_mean, data_std)
        lbottom = torch.mean(
            torch.square((pred_bottom[:, 0] - uvp_bottom[:, 3]))
            + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 4])))  ## zero  ## zero
        )

        pred_up = self.model_force(txy_up, data_mean, data_std)
        lup = torch.mean(
            torch.square(((pred_up[:, 0] - uvp_up[:, 3])))
            + torch.square(((pred_up[:, 1] - uvp_up[:, 4])))  ## one
        )  ## zero

        pred_initial = self.model_force(txy_initial, data_mean, data_std)
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_initial[:, 3])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 4])
        )
        ## presssure training is necessary

        pred_sensors = self.model_force(txy_sensors, data_mean, data_std)
        lsensors = torch.mean(
            torch.square(pred_sensors[:, 0] - uvp_sensors[:, 3])
            + torch.square(pred_sensors[:, 1] - uvp_sensors[:, 4])
        )

        return {
            # "left": lleft,
            # "right": lright,
            # "bottom": lbottom,
            # "up": lup,
            "fluid_points": lsensors,
            "initial": linitial,
        }

    def get_mini_batch_data(self, data):
        data_mean = data.mean_x
        data_std = data.std_x

        txy_domain = data.txy_fluid
        uvp_domain = data.uvp_fluid
        txy_sensors = data.txy_fluid_points
        txy_left = data.txy_left
        txy_right = data.txy_right
        txy_bottom = data.txy_bottom
        txy_up = data.txy_up
        txy_initial = data.txy_initial

        # txy_fluid = training_dataset[1]["txy_fluid"]
        uvp_sensors = data.uvp_fluid_points
        uvp_left = data.uvp_left
        uvp_right = data.uvp_right
        uvp_bottom = data.uvp_bottom
        uvp_up = data.uvp_up
        uvp_initial = data.uvp_initial

        # Access the randomly selected minibatch for each tensor
        batch_indices = self.get_random_minibatch(txy_domain.shape[0])
        txy_domain = txy_domain[batch_indices, :]

        batch_indices = self.get_random_minibatch(txy_sensors.shape[0])
        txy_sensors = txy_sensors[batch_indices, :]
        uvp_sensors = uvp_sensors[batch_indices, :]

        batch_indices = self.get_random_minibatch(txy_left.shape[0])
        txy_left = txy_left[batch_indices, :]
        uvp_left = uvp_left[batch_indices, :]

        batch_indices = self.get_random_minibatch(txy_right.shape[0])
        txy_right = txy_right[batch_indices, :]
        uvp_right = uvp_right[batch_indices, :]

        batch_indices = self.get_random_minibatch(txy_bottom.shape[0])
        txy_bottom = txy_bottom[batch_indices, :]
        uvp_bottom = uvp_bottom[batch_indices, :]

        batch_indices = self.get_random_minibatch(txy_up.shape[0])
        txy_up = txy_up[batch_indices, :]
        uvp_up = uvp_up[batch_indices, :]

        batch_indices = self.get_random_minibatch(txy_initial.shape[0])
        txy_initial = txy_initial[batch_indices, :]
        uvp_initial = uvp_initial[batch_indices, :]
        return (
            data_mean,
            data_std,
            txy_domain,
            uvp_domain,
            txy_sensors,
            txy_left,
            txy_right,
            txy_bottom,
            txy_up,
            txy_initial,
            uvp_sensors,
            uvp_left,
            uvp_right,
            uvp_bottom,
            uvp_up,
            uvp_initial,
        )
