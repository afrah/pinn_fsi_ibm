import os
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

    def _run_epoch(self, epoch, training_dataset):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            start_time = time.time()

        # print("inside _run_epoch" , epoch)

        bclosses = self._compute_losses(training_dataset)
        self.update_epoch_loss(bclosses)

        total_loss = sum(
            [
                torch.mean(self.weights.get("lleft") * self.epoch_loss["lleft"]),
                torch.mean(self.weights.get("lright") * self.epoch_loss["lright"]),
                torch.mean(self.weights.get("lbottom") * self.epoch_loss["lbottom"]),
                torch.mean(self.weights.get("lup") * self.epoch_loss["lup"]),
                torch.mean(self.weights.get("lsensors") * self.epoch_loss["lsensors"]),
                torch.mean(self.weights.get("linitial") * self.epoch_loss["linitial"]),
                torch.mean(self.weights.get("lphy") * self.epoch_loss["lphy"]),
            ]
        )

        if self.rank == 0:
            elapsed_time = time.time() - start_time

        self.fluid_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        ##### ____ Update Weights ____#####

        #### update schedular and optimizer

        self.fluid_optimizer.step()
        if epoch % 1000 == 0:
            self.fluid_scheduler.step()

        ### printing
        if self.rank == 0 and epoch % self.config.get("print_every") == 0:
            self.track_training(
                int(epoch / self.config.get("print_every")),
                elapsed_time,
            )

    def _compute_losses(self):
        txy_domain = self.train_dataloader.fluid_data.txy_fluid
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

        # print("txy_domain shape : " , txy_domain.shape)
        continuity, f_u, f_v = navier_stokes_2D_IBM(txy_domain, self.fluid_model)
        lphy = torch.sqrt(continuity**2 + f_u**2 + f_v**2)

        pred_left = self.fluid_model(txy_left)
        lleft = torch.sqrt(
            (pred_left[:, 0] - uvp_left[:, 0]) ** 2
        ) + torch.sqrt(  ## zero
            (pred_left[:, 1] - uvp_left[:, 1]) ** 2
        )  ## zero

        pred_right = self.fluid_model(txy_right)
        lright = torch.sqrt(
            (pred_right[:, 0] - uvp_right[:, 0]) ** 2
        ) + torch.sqrt(  ## zero
            (pred_right[:, 1] - uvp_right[:, 1]) ** 2
        )  ## zero

        pred_bottom = self.fluid_model(txy_bottom)
        lbottom = torch.sqrt((pred_bottom[:, 0] - uvp_bottom[:, 0]) ** 2) + (  ## zero
            torch.sqrt((pred_bottom[:, 1] - uvp_bottom[:, 1]) ** 2)
        )  ## zero

        pred_up = self.fluid_model(txy_up)
        lup = (torch.sqrt((pred_up[:, 0] - uvp_up[:, 0]) ** 2)) + (  ## one
            torch.sqrt((pred_up[:, 1] - uvp_up[:, 1]) ** 2)
        )  ## zero

        pred_initial = self.fluid_model(txy_initial)
        linitial = (
            torch.sqrt((pred_initial[:, 0] - uvp_initial[:, 0]) ** 2)  ## zero
            + torch.sqrt((pred_initial[:, 1] - uvp_initial[:, 1]) ** 2)  ## zero
            + torch.sqrt((pred_initial[:, 2] - uvp_initial[:, 2]) ** 2)  ## zero
            ## presssure training is necessary
        )

        pred_sensors = self.fluid_model(txy_sensors)
        lsensors = (
            # self.weights.get("lsensors")   *
            torch.sqrt((pred_sensors[:, 0] - uvp_sensors[:, 0]) ** 2)  ## nonzero
            + torch.sqrt((pred_sensors[:, 1] - uvp_sensors[:, 1]) ** 2)  ## nonzero
            + torch.sqrt((pred_sensors[:, 2] - uvp_sensors[:, 2]) ** 2)  # p ## nonzero
            # +  torch.sqrt(pred_sensors[:, 3], uvp_sensors[:, 3])## nonzero
            # +  torch.sqrt(pred_sensors[:, 4], uvp_sensors[:, 4])## nonzero
            ## presssure training is necessary
        )

        return {
            "lleft": lleft,
            "lright": lright,
            "lbottom": lbottom,
            "lup": lup,
            "lsensors": lsensors,
            "linitial": linitial,
            "lphy": lphy,
        }
