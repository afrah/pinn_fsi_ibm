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
from src.trainer.base_trainer import BaseTrainer
from src.nn.nn_functions import MSE
from src.utils.max_eigenvlaue_of_hessian import power_iteration

## End: Importing local packages


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_dataloader,
        fluid_model: nn.Module,
        fluid_optimizer: optim.Optimizer,
        fluid_scheduler,
        rank: int,
        config,
    ) -> None:
        super(Trainer, self).__init__(
            train_dataloader,
            fluid_model,
            fluid_optimizer,
            fluid_scheduler,
            rank,
            config,
        )

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            start_time = time.time()

        # print("inside _run_epoch" , epoch)

        bclosses = self._compute_losses()

        total_loss = sum(
            [
                2.0 * ((bclosses["left"])),
                2.0 * ((bclosses["right"])),
                2.0 * ((bclosses["bottom"])),
                2.0 * ((bclosses["up"])),
                4.0 * ((bclosses["fluid_points"])),
                4.0 * ((bclosses["initial"])),
                0.01 * ((bclosses["fluid"])),
            ]
        )

        if self.rank == 0:
            elapsed_time = time.time() - start_time

        self.optimizer_fluid.zero_grad()

        self.update_epoch_loss(bclosses)

        ### printing
        if self.rank == 0 and epoch % self.config.get("print_every") == 0:

            loss_bc = sum(
                [
                    ((bclosses["left"])),
                    ((bclosses["right"])),
                    ((bclosses["bottom"])),
                    ((bclosses["up"])),
                    ((bclosses["fluid_points"])),
                ]
            )
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
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )
        total_loss.backward()
        self.optimizer_fluid.step()
        self.fluid_scheduler.step()

    def _compute_losses(self):
        data_mean = self.train_dataloader.fluid_data.mean_x
        data_std = self.train_dataloader.fluid_data.std_x

        txy_domain = self.train_dataloader.fluid_data.txy_fluid
        uvp_domain = self.train_dataloader.fluid_data.uvp_fluid
        txy_sensors = self.train_dataloader.fluid_data.txy_fluid_points
        txy_left = self.train_dataloader.fluid_data.txy_left
        txy_right = self.train_dataloader.fluid_data.txy_right
        txy_bottom = self.train_dataloader.fluid_data.txy_bottom
        txy_up = self.train_dataloader.fluid_data.txy_up
        txy_initial = self.train_dataloader.fluid_data.txy_initial

        # txy_fluid = training_dataset[1]["txy_fluid"]
        uvp_sensors = self.train_dataloader.fluid_data.uvp_fluid_points
        uvp_left = self.train_dataloader.fluid_data.uvp_left
        uvp_right = self.train_dataloader.fluid_data.uvp_right
        uvp_bottom = self.train_dataloader.fluid_data.uvp_bottom
        uvp_up = self.train_dataloader.fluid_data.uvp_up
        uvp_initial = self.train_dataloader.fluid_data.uvp_initial

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

        # print("txy_domain shape : " , txy_domain.shape)
        continuity, f_u, f_v = navier_stokes_2D_IBM(
            txy_domain, self.fluid_model, data_mean, data_std
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

        pred_sensors = self.fluid_model(txy_sensors, data_mean, data_std)
        lsensors = torch.mean(
            torch.square(pred_sensors[:, 0] - uvp_sensors[:, 0])
            + torch.square(pred_sensors[:, 1] - uvp_sensors[:, 1])
            + torch.square(pred_sensors[:, 2] - uvp_sensors[:, 2])
            # + torch.square(pred_sensors[:, 3] - uvp_sensors[:, 3])
            # + torch.square(pred_sensors[:, 4] - uvp_sensors[:, 4])
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
