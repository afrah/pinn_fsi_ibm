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
from src.trainer_M5.base_trainer import BaseTrainer
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
            t1 = time.time()

        # print("inside _run_epoch" , epoch)

        losses_velocity, losses_force = self._compute_losses()

        total_loss_force = sum(
            [
                # 2.0 * ((losses_velocity["left"])),
                # 2.0 * ((losses_velocity["right"])),
                # 2.0 * ((losses_velocity["bottom"])),
                # 2.0 * ((losses_velocity["up"])),
                2.0
                * ((losses_force["fluid_points"])),
            ]
        )
        total_loss_velocity = sum(
            [
                2.0 * ((losses_velocity["lint_pts"])),
                4.0 * ((losses_velocity["int_initial"])),
                2.0 * ((losses_velocity["vCoupling"])),
                0.01 * ((losses_velocity["fluid"])),
                2 * ((losses_velocity["left"])),
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

    def _compute_losses_force(self):

        data_mean = self.train_dataloader.fluid_data.mean_x[:3]
        data_std = self.train_dataloader.fluid_data.std_x[:3]
        uvp_domain = self.train_dataloader.fluid_data.uvp_fluid

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

        # lphy = torch.mean(torch.square(f_u) + torch.square(f_v))

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_initial.shape[0]
        )
        txy_initial = self.train_dataloader.fluid_data.txy_initial[batch_indices, :]
        uvp_initial = self.train_dataloader.fluid_data.uvp_initial[batch_indices, :]

        pred_initial = self.model_force(txy_initial, data_mean, data_std)
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_initial[:, 3])
            + torch.square(pred_initial[:, 1] - uvp_initial[:, 4])
        )
        ## presssure training is necessary

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid_points.shape[0]
        )
        txy_fluid_data = self.train_dataloader.fluid_data.txy_fluid_points[
            batch_indices, :
        ]
        uvp_fluid_data = self.train_dataloader.fluid_data.uvp_fluid_points[
            batch_indices, :
        ]

        pred_fluid_data = self.model_force(txy_fluid_data, data_mean, data_std)
        fluid_data = torch.mean(
            torch.square(pred_fluid_data[:, 0] - uvp_fluid_data[:, 3])
            + torch.square(pred_fluid_data[:, 1] - uvp_fluid_data[:, 4])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.solid_data.txy_solid_points.shape[0]
        )
        txy_solid_data = self.train_dataloader.solid_data.txy_solid_points[
            batch_indices, :
        ]
        uvp_solid_data = self.train_dataloader.solid_data.uvp_solid_points[
            batch_indices, :
        ]

        pred_solid_data = self.model_force(txy_solid_data, data_mean, data_std)
        solid_data = torch.mean(
            torch.square(pred_solid_data[:, 0] - uvp_solid_data[:, 3])
            + torch.square(pred_solid_data[:, 1] - uvp_solid_data[:, 4])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_initial_interface.shape[0]
        )
        txy_intitial = self.train_dataloader.interface_data.txy_initial_interface[
            batch_indices, :
        ]
        uvp_intitial = self.train_dataloader.interface_data.uvp_initial_interface[
            batch_indices, :
        ]

        pred_initial = self.model_force(
            txy_intitial,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )
        linitial_interface = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_intitial[:, 3])
            + torch.square(pred_initial[:, 1] - uvp_intitial[:, 4])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_interface.shape[0]
        )
        txy_interface_data = self.train_dataloader.interface_data.txy_interface[
            batch_indices, :
        ]
        uvp_interface_data = self.train_dataloader.interface_data.uvp_interface[
            batch_indices, :
        ]
        pred_interface = self.model_force(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        interface_data = torch.mean(
            torch.square(pred_interface[:, 0] - uvp_interface_data[:, 3])
            + torch.square(pred_interface[:, 1] - uvp_interface_data[:, 4])
        )
        return {
            "fluid_points": fluid_data + solid_data,
            "initial": linitial,
        }

    def _compute_losses_velocity(self):

        # Access the randomly selected minibatch for each tensor

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_initial_interface.shape[0]
        )
        txy_intitial = self.train_dataloader.interface_data.txy_initial_interface[
            batch_indices, :
        ]
        uvp_intitial = self.train_dataloader.interface_data.uvp_initial_interface[
            batch_indices, :
        ]

        pred_initial = self.model_fluid(
            txy_intitial,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )
        linitial = torch.mean(
            torch.square(pred_initial[:, 0] - uvp_intitial[:, 0])
            + torch.square(pred_initial[:, 1] - uvp_intitial[:, 1])
        )

        batch_indices = self.get_random_minibatch(
            self.train_dataloader.interface_data.txy_interface.shape[0]
        )
        txy_interface_data = self.train_dataloader.interface_data.txy_interface[
            batch_indices, :
        ]
        uvp_interface_data = self.train_dataloader.interface_data.uvp_interface[
            batch_indices, :
        ]

        pred_interface = self.model_fluid(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        linterface = torch.mean(
            torch.square(pred_interface[:, 0] - uvp_interface_data[:, 0])
            + torch.square(pred_interface[:, 1] - uvp_interface_data[:, 1])
        )
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid_points.shape[0]
        )
        txy_sensors = self.train_dataloader.fluid_data.txy_fluid_points[
            batch_indices, :
        ]
        uvp_sensors = self.train_dataloader.fluid_data.uvp_fluid_points[
            batch_indices, :
        ]

        data_mean = self.train_dataloader.fluid_data.mean_x[:3]
        data_std = self.train_dataloader.fluid_data.std_x[:3]

        pred_sensors = self.model_fluid(txy_sensors, data_mean, data_std)

        fluid_data = torch.mean(
            torch.square(pred_sensors[:, 0] - uvp_sensors[:, 0])
            + torch.square(pred_sensors[:, 1] - uvp_sensors[:, 1])
            + torch.square(pred_sensors[:, 2] - uvp_sensors[:, 2])
            # + torch.square(pred_sensors[:, 4] - uvp_sensors[:, 4])
        )
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_fluid.shape[0]
        )
        txy_domain = self.train_dataloader.fluid_data.txy_fluid[batch_indices, :]

        continuity, f_u, f_v = navier_stokes_2D_IBM(
            txy_domain, self.model_fluid, self.model_force, data_mean, data_std
        )

        # pred_sensors = self.model_force(txy_domain, data_mean, data_std)
        lphy = torch.mean(
            torch.square(continuity) + torch.square(f_u) + torch.square(f_v)
        )

        pred_fluid1 = self.model_fluid(
            txy_interface_data,
            self.train_dataloader.interface_data.mean_x[:3],
            self.train_dataloader.interface_data.std_x[:3],
        )

        vCoupling = torch.mean(
            torch.square(pred_fluid1[:, 0] - uvp_interface_data[:, 0])
            + torch.square(pred_fluid1[:, 1] - uvp_interface_data[:, 1])
        )
        batch_indices = self.get_random_minibatch(
            self.train_dataloader.fluid_data.txy_up.shape[0]
        )
        txy_up = self.train_dataloader.fluid_data.txy_up[batch_indices, :]
        uvp_up = self.train_dataloader.fluid_data.uvp_up[batch_indices, :]

        pred_up = self.model_force(txy_up, data_mean, data_std)
        lup = torch.mean(
            torch.square(((pred_up[:, 0] - uvp_up[:, 0])))
            + torch.square(((pred_up[:, 1] - uvp_up[:, 1])))  ## one
        )  ## zero

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

        pred_left = self.model_force(txy_left, data_mean, data_std)
        lleft = torch.mean(
            torch.square(pred_left[:, 0] - uvp_left[:, 0])
            + torch.square(pred_left[:, 1] - uvp_left[:, 1])
        )

        pred_right = self.model_force(txy_right, data_mean, data_std)
        lright = torch.mean(
            torch.square(pred_right[:, 0] - uvp_right[:, 0])
            + torch.square(pred_right[:, 1] - uvp_right[:, 1])
        )

        pred_bottom = self.model_force(txy_bottom, data_mean, data_std)
        lbottom = torch.mean(
            torch.square((pred_bottom[:, 0] - uvp_bottom[:, 0]))
            + torch.square(((pred_bottom[:, 1] - uvp_bottom[:, 1])))  ## zero  ## zero
        )

        return {
            "left": lleft + lright + lup,
            "right": lright,
            "up": lup,
            "bottom": lbottom,
            "lint_pts": linterface + fluid_data,
            "int_initial": linitial,
            "fluid": lphy,
            "vCoupling": vCoupling,  # + vCoupling1 + vCoupling2,
        }

    # def _compute_losses_force(self):

    #     # Access the randomly selected minibatch for each tensor

    #     batch_indices = self.get_random_minibatch(
    #         self.train_dataloader.interface_data.txy_interface.shape[0]
    #     )
    #     txy_interface_data = self.train_dataloader.interface_data.txy_interface[
    #         batch_indices, :
    #     ]
    #     uvp_interface_data = self.train_dataloader.interface_data.uvp_interface[
    #         batch_indices, :
    #     ]

    #     batch_indices = self.get_random_minibatch(
    #         self.train_dataloader.interface_data.txy_initial_interface.shape[0]
    #     )
    #     txy_intitial = self.train_dataloader.interface_data.txy_initial_interface[
    #         batch_indices, :
    #     ]
    #     uvp_intitial = self.train_dataloader.interface_data.uvp_initial_interface[
    #         batch_indices, :
    #     ]

    #     pred_initial = self.model_force(
    #         txy_intitial,
    #         self.train_dataloader.interface_data.mean_x[:3],
    #         self.train_dataloader.interface_data.std_x[:3],
    #     )
    #     linitial = torch.mean(
    #         torch.square(pred_initial[:, 0] - uvp_intitial[:, 0])
    #         + torch.square(pred_initial[:, 1] - uvp_intitial[:, 1])
    #     )

    #     pred_interface = self.model_force(
    #         txy_interface_data,
    #         self.train_dataloader.interface_data.mean_x[:3],
    #         self.train_dataloader.interface_data.std_x[:3],
    #     )

    #     linterface = torch.mean(
    #         torch.square(pred_interface[:, 0] - uvp_interface_data[:, 0])
    #         + torch.square(pred_interface[:, 1] - uvp_interface_data[:, 1])
    #     )
    #     batch_indices = self.get_random_minibatch(
    #         self.train_dataloader.fluid_data.txy_fluid_points.shape[0]
    #     )
    #     txy_sensors = self.train_dataloader.fluid_data.txy_fluid_points[
    #         batch_indices, :
    #     ]
    #     uvp_sensors = self.train_dataloader.fluid_data.uvp_fluid_points[
    #         batch_indices, :
    #     ]

    #     data_mean = self.train_dataloader.fluid_data.mean_x[:3]
    #     data_std = self.train_dataloader.fluid_data.std_x[:3]

    #     pred_sensors = self.model_force(txy_sensors, data_mean, data_std)
    #     fluid_data = torch.mean(
    #         torch.square(pred_sensors[:, 0] - uvp_sensors[:, 0])
    #         + torch.square(pred_sensors[:, 1] - uvp_sensors[:, 1])
    #         # + torch.square(pred_sensors[:, 3] - uvp_sensors[:, 3])
    #         # + torch.square(pred_sensors[:, 4] - uvp_sensors[:, 4])
    #     )

    #     return {
    #         "lint_pts": linterface + fluid_data,
    #         "int_initial": linitial,
    #     }
