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
        if self.config.get("weighting") == "RBA":
            self.ETA = 0.5
            self.weights = {
                key: torch.rand(
                    self.batch_size, 1, dtype=torch.float32, device=self.rank
                )
                for key in [
                    "fluid",
                    "left",
                    "right",
                    "bottom",
                    "up",
                    "initial",
                    "fluid_points",
                    "force_points",
                ]
            }
            self.GAMMA = 0.5

    def _run_epoch(self, epoch):
        # self.train_dataloader.sampler.set_epoch(epoch)

        if self.rank == 0:
            t1 = time.time()

        # print("inside _run_epoch" , epoch)

        losses_velocity, losses_force = self._compute_losses()

        if self.config.get("weighting") == "RBA":
            if epoch % 1000 == 0:
                for key, value in self.weights.items():
                    print(f"Mean of {key} weights : {torch.mean(value).item():.2f}")

            for key in self.epoch_loss.keys():
                if self.weights.get(key) is not None and key != "force_points":
                    r_norm1 = (
                        self.ETA
                        * torch.abs(torch.sqrt(losses_velocity[key]))
                        / torch.max(torch.abs(torch.sqrt(losses_velocity[key])))
                    )
                    self.weights[key] = (
                        self.weights[key] * self.GAMMA + r_norm1 * (1.0 - self.GAMMA)
                    ).detach()
                    r_norm2 = (
                        self.ETA
                        * torch.abs(torch.sqrt(losses_force["force_points"]))
                        / torch.max(torch.abs(torch.sqrt(losses_force["force_points"])))
                    )
                    self.weights["force_points"] = (
                        self.weights["force_points"] * self.GAMMA
                        + r_norm2 * (1.0 - self.GAMMA)
                    ).detach()

            total_loss_velocity = sum(
                torch.mean(
                    torch.square(self.weights[key] * torch.sqrt(losses_velocity[key]))
                )
                for key in self.weights.keys()
                if key != "force_points"
            )

            total_losses_force = torch.mean(
                torch.square(
                    self.weights["force_points"]
                    * torch.sqrt(losses_force["force_points"])
                )
            )
        else:
            print("Weighting is not implemented")

        self.optimizer_fluid.zero_grad()
        self.optimizer_force.zero_grad()

        if self.rank == 0:
            elapsed_time1 = time.time() - t1

        losses_velocity["left"] = torch.mean(losses_velocity["left"])
        losses_velocity["right"] = torch.mean(losses_velocity["right"])
        losses_velocity["bottom"] = torch.mean(losses_velocity["bottom"])
        losses_velocity["up"] = torch.mean(losses_velocity["up"])
        losses_velocity["fluid_points"] = torch.mean(losses_velocity["fluid_points"])
        losses_velocity["initial"] = torch.mean(losses_velocity["initial"])
        losses_velocity["fluid"] = torch.mean(losses_velocity["fluid"])

        losses_force["force_points"] = torch.mean(losses_force["force_points"])

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

        if self.rank == 0:
            t2 = time.time()

        total_loss_velocity.backward()
        total_losses_force.backward()

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
        lleft = torch.square(pred_left[:, 0] - uvp_left[:, 0]) + torch.square(
            pred_left[:, 1] - uvp_left[:, 1]
        )

        pred_right = self.model_fluid(txy_right, data_mean, data_std)
        lright = torch.square(pred_right[:, 0] - uvp_right[:, 0]) + torch.square(
            pred_right[:, 1] - uvp_right[:, 1]
        )

        pred_bottom = self.model_fluid(txy_bottom, data_mean, data_std)
        lbottom = torch.square((pred_bottom[:, 0] - uvp_bottom[:, 0])) + torch.square(
            ((pred_bottom[:, 1] - uvp_bottom[:, 1]))
        )  ## zero  ## zero

        pred_up = self.model_fluid(txy_up, data_mean, data_std)
        lup = torch.square(((pred_up[:, 0] - uvp_up[:, 0]))) + torch.square(
            ((pred_up[:, 1] - uvp_up[:, 1]))
        )  ## one  ## zero

        pred_initial = self.model_fluid(txy_initial, data_mean, data_std)
        linitial = (
            torch.square(pred_initial[:, 0] - uvp_initial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 1])
            + torch.square(pred_initial[:, 2] - uvp_initial[:, 2])
        )
        ## presssure training is necessary

        pred_sensors = self.model_fluid(txy_sensors, data_mean, data_std)
        lsensors = (
            torch.square(pred_sensors[:, 0] - uvp_sensors[:, 0])
            + torch.square(pred_sensors[:, 1] - uvp_sensors[:, 1])
            + torch.square(pred_sensors[:, 2] - uvp_sensors[:, 2])
        )

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
            self.train_dataloader.fluid_data.txy_fluid_points.shape[0]
        )
        txy_fluid = self.train_dataloader.fluid_data.txy_fluid_points[batch_indices, :]
        uvp_fluid = self.train_dataloader.fluid_data.uvp_fluid_points[batch_indices, :]

        txy_solid = self.train_dataloader.solid_data.txy_solid_points[batch_indices, :]
        uvp_solid = self.train_dataloader.solid_data.uvp_solid_points[batch_indices, :]

        # txy_solid = torch.cat((txy_solid, uvp_solid[:, :3]), dim=1)

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_left.shape[0]
        )

        pred_fluid = self.model_force(
            txy_fluid,
            self.train_dataloader.fluid_data.mean_x[:3],
            self.train_dataloader.fluid_data.std_x[:3],
        )
        lfluid = torch.square(pred_fluid[:, 0] - uvp_fluid[:, 3]) + torch.square(
            pred_fluid[:, 1] - uvp_fluid[:, 4]
        )
        pred_solid = self.model_force(
            txy_solid,
            self.train_dataloader.fluid_data.mean_x[:3],
            self.train_dataloader.fluid_data.std_x[:3],
        )
        lsolid = torch.square(pred_solid[:, 0] - uvp_solid[:, 3]) + torch.square(
            pred_solid[:, 1] - uvp_solid[:, 4]
        )

        return {
            "force_points": lfluid + lsolid,
        }

    def get_mini_batch_data(self, data):
        data_mean = data.mean_x[:3]
        data_std = data.std_x[:3]

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

        # pred_left = self.model_force(
        #     txy_left,
        #     self.train_dataloader.fluid_data.mean_x[:3],
        #     self.train_dataloader.fluid_data.std_x[:3],
        # )
        # lleft = torch.mean(
        #     torch.square(pred_left[:, 0] - uvp_left[:, 3])
        #     + torch.square(pred_left[:, 1] - uvp_left[:, 4])
        # )

        # pred_right = self.model_force(
        #     txy_right,
        #     self.train_dataloader.fluid_data.mean_x[:3],
        #     self.train_dataloader.fluid_data.std_x[:3],
        # )
        # lright = torch.mean(
        #     torch.square(pred_right[:, 0] - uvp_right[:, 3])
        #     + torch.square(pred_right[:, 1] - uvp_right[:, 4])
        # )

        # pred_bottom = self.model_force(
        #     txy_bottom,
        #     self.train_dataloader.fluid_data.mean_x[:3],
        #     self.train_dataloader.fluid_data.std_x[:3],
        # )
        # lbottom = torch.mean(
        #     torch.square((pred_bottom[:, 0] - uvp_bottom[:, 3]))
        #     + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 4])))  ## zero  ## zero
        # )

        # pred_up = self.model_force(
        #     txy_up,
        #     self.train_dataloader.fluid_data.mean_x[:3],
        #     self.train_dataloader.fluid_data.std_x[:3],
        # )
        # lup = torch.mean(
        #     torch.square(((pred_up[:, 0] - uvp_up[:, 3])))
        #     + torch.square(((pred_up[:, 1] - uvp_up[:, 4])))  ## one
        # )  ## zero

        # pred_initial = self.model_force(
        #     txy_initial,
        #     self.train_dataloader.fluid_data.mean_x[:3],
        #     self.train_dataloader.fluid_data.std_x[:3],
        # )
        # linitial = torch.mean(
        #     torch.square(pred_initial[:, 0] - uvp_initial[:, 3])
        #     + torch.square(pred_initial[:, 1] - uvp_initial[:, 4])
        # )

        #         txy_left = self.train_dataloader.fluid_data.txy_left[batch_indices, :]
        # uvp_left = self.train_dataloader.fluid_data.uvp_left[batch_indices, :]
        # # txy_left = torch.cat((txy_left, uvp_left[:, :3]), dim=1)

        # batch_indices = self.get_random_minibatch(
        #     self.train_dataloader.fluid_data.txy_right.shape[0]
        # )
        # txy_right = self.train_dataloader.fluid_data.txy_right[batch_indices, :]
        # uvp_right = self.train_dataloader.fluid_data.uvp_right[batch_indices, :]
        # # txy_right = torch.cat((txy_right, uvp_right[:, :3]), dim=1)

        # batch_indices = self.get_random_minibatch(
        #     self.train_dataloader.fluid_data.txy_bottom.shape[0]
        # )
        # txy_bottom = self.train_dataloader.fluid_data.txy_bottom[batch_indices, :]
        # uvp_bottom = self.train_dataloader.fluid_data.uvp_bottom[batch_indices, :]
        # # txy_bottom = torch.cat((txy_bottom, uvp_bottom[:, :3]), dim=1)

        # batch_indices = self.get_random_minibatch(
        #     self.train_dataloader.fluid_data.txy_up.shape[0]
        # )
        # txy_up = self.train_dataloader.fluid_data.txy_up[batch_indices, :]
        # uvp_up = self.train_dataloader.fluid_data.uvp_up[batch_indices, :]
        # # txy_up = torch.cat((txy_up, uvp_up[:, :3]), dim=1)

        # batch_indices = self.get_random_minibatch(
        #     self.train_dataloader.fluid_data.txy_initial.shape[0]
        # )
        # txy_initial = self.train_dataloader.fluid_data.txy_initial[batch_indices, :]
        # uvp_initial = self.train_dataloader.fluid_data.uvp_initial[batch_indices, :]
        # # txy_initial = torch.cat((txy_initial, uvp_initial[:, :3]), dim=1)
