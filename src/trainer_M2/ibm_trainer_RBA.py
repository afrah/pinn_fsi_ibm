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
        if self.config.get("weighting") == "RBA":
            self.ETA = 0.001
            self.weights = {
                key: torch.zeros(1).to(self.rank) for key in self.epoch_loss.keys()
            }
            self.GAMMA = 0.999
            self.weights = {
                key: torch.ones(
                    getattr(self.train_dataloader.fluid_data, f"txy_{key}").shape[0],
                    1,
                    requires_grad=True,
                )
                .float()
                .to(self.rank)
                for key in [
                    "fluid",
                    "left",
                    "right",
                    "bottom",
                    "up",
                    "initial",
                    "fluid_points",
                ]
            }

    # def adapt_weight_RBA(self):
    #     for key, value in self.epoch_loss.items():
    #         if self.weights.get(key) is not None:
    #             r_norm = self.ETA * torch.abs(value) / torch.max(value)
    #             new_weights = (self.weights.get(key) * self.GAMMA + r_norm).detach()
    #             self.weights[key] = new_weights

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            start_time = time.time()

        # print("inside _run_epoch" , epoch)

        bclosses = self._compute_losses()
        if self.config.get("weighting") == "RBA":
            for key in self.epoch_loss.keys():
                if self.weights.get(key) is not None:
                    r_norm = (
                        self.ETA
                        * torch.abs(torch.sqrt(bclosses[key]))
                        / torch.max(torch.abs(torch.sqrt(bclosses[key])))
                    )
                    self.weights[key] = (
                        self.weights[key] * self.GAMMA + r_norm
                    ).detach()
                    if epoch % 1000 == 0:
                        print(f"{key} : {torch.mean((self.weights[key])).item(): .2f}")

            total_loss = sum(
                torch.mean(self.weights[key] * bclosses[key])
                for key in self.weights.keys()
            )

        else:
            print("Weighting is not implemented")
        if self.rank == 0:
            elapsed_time = time.time() - start_time

        self.optimizer_fluid.zero_grad()

        ##### ____ Update Weights ____#####

        #### update schedular and optimizer

        ### printing
        bclosses["left"] = torch.mean(bclosses["left"])
        bclosses["right"] = torch.mean(bclosses["right"])
        bclosses["bottom"] = torch.mean(bclosses["bottom"])
        bclosses["up"] = torch.mean(bclosses["up"])
        bclosses["fluid_points"] = torch.mean(bclosses["fluid_points"])
        bclosses["initial"] = torch.mean(bclosses["initial"])
        bclosses["fluid"] = torch.mean(bclosses["fluid"])

        self.update_epoch_loss(bclosses)

        if self.rank == 0 and epoch % self.config.get("print_every") == 0:

            loss_bc = sum(
                [
                    torch.mean((bclosses["left"])),
                    torch.mean((bclosses["right"])),
                    torch.mean((bclosses["bottom"])),
                    torch.mean((bclosses["up"])),
                    torch.mean((bclosses["fluid_points"])),
                ]
            )
            loss_initial = torch.mean((bclosses["initial"]))
            loss_res = torch.mean((bclosses["fluid"]))

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
        total_loss.backward()
        self.optimizer_fluid.step()
        self.scheduler_fluid.step()

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
            txy_domain, self.model_fluid, data_mean, data_std
        )

        lphy = torch.square(continuity) + torch.square(f_u) + torch.square(f_v)

        pred_left = self.model_fluid(txy_left, data_mean, data_std)
        lleft = torch.square(pred_left[:, 0] - uvp_left[:, 0]) + torch.square(
            pred_left[:, 1] - uvp_left[:, 1]
        )

        pred_right = self.model_fluid(txy_right, data_mean, data_std)
        lright = torch.square(pred_right[:, 0] - uvp_right[:, 0]) + torch.square(
            pred_right[:, 1] - uvp_right[:, 1]
        )

        pred_bottom = self.model_fluid(txy_bottom, data_mean, data_std)
        lbottom = torch.square(
            (pred_bottom[:, 0] - uvp_bottom[:, 0])
        ) + torch.square(  ## zero
            ((pred_bottom[:, 1] - uvp_bottom[:, 1]))
        )  ## zero

        pred_up = self.model_fluid(txy_up, data_mean, data_std)
        lup = torch.square(((pred_up[:, 0] - uvp_up[:, 0]))) + torch.square(  ## one
            ((pred_up[:, 1] - uvp_up[:, 1]))
        )  ## zero

        pred_initial = self.model_fluid(txy_initial, data_mean, data_std)
        linitial = (
            torch.square(pred_initial[:, 0] - uvp_initial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 1])
            + torch.square(pred_initial[:, 2] - uvp_initial[:, 2])
            # + torch.square(pred_initial[:, 3] - uvp_initial[:, 3])
            # + torch.square(pred_initial[:, 4] - uvp_initial[:, 4])
        )
        ## presssure training is necessary

        pred_sensors = self.model_fluid(txy_sensors, data_mean, data_std)
        lsensors = (
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
