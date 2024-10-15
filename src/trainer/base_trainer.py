import os
import torch
import torch.nn as nn
import torch.optim as optim
import random

# from torch.utils.tensorboard import SummaryWriter


from src.utils.logger import Logging
from src.utils import printing
from src.data.IBM_data_loader import IBM_data_loader


class BaseTrainer:
    def __init__(
        self,
        train_dataloader: IBM_data_loader,
        fluid_model: nn.Module,
        fluid_optimizer: optim.Optimizer,
        fluid_scheduler,
        rank: int,
        config,
    ) -> None:
        self.rank = rank
        self.fluid_model = fluid_model.to(self.rank)
        self.train_dataloader = train_dataloader
        self.fluid_optimizer = fluid_optimizer
        self.fluid_scheduler = fluid_scheduler
        self.config = config
        self.running_time = 0.0
        self.initial_epoch_loss = {}
        self.max_eig_hessian_bc_log = []
        self.max_eig_hessian_ic_log = []
        self.max_eig_hessian_res_log = []
        self.batch_size = config.get("batch_size")

        self.epoch_loss = {
            loss: torch.tensor(0.0, requires_grad=True).to(self.rank)
            for loss in self.config["loss_list"]
        }

        if self.rank == 0:
            self._initialize_logging()

        # SA weights initialization

        if self.config.get("weighting") == "SA":
            self.weights = {
                key: nn.Parameter(
                    torch.rand(
                        (
                            getattr(
                                self.train_dataloader.fluid_data, f"txy_{key}"
                            ).shape[0],
                            1,
                        ),
                        dtype=torch.float32,
                        device=self.rank,
                    ),
                    requires_grad=True,
                )
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

            self.sa_optimizer = torch.optim.Adam(
                [param for param in self.weights.values() if param.requires_grad],
                lr=0.005,
                maximize=True,
            )
        # Uncomment the following line to use Tensorboard
        # self._initialize_tensorboard()

    def get_random_minibatch(self, dataset_length):
        batch_indices = random.sample(range(dataset_length), self.batch_size)
        return batch_indices

    def _initialize_logging(self):
        self.logger = Logging(self.config.get("log_path"))
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"checkpoint path: {self.log_path=}")
        self.loss_history = {loss: [] for loss in self.config["loss_list"]}

    def update_epoch_loss(self, losses):
        with torch.no_grad():
            for loss_type in self.config["loss_list"]:
                self.epoch_loss[loss_type] = losses.get(loss_type)

            if self.rank == 0:
                self.update_loss_history()

    def update_loss_history(self):
        for key in self.config["loss_list"]:
            self.loss_history[key].append(self.epoch_loss[key].item())

    def train_mini_batch(self):

        for epoch in range(self.config.get("total_epochs") + 1):
            self._run_epoch(epoch)
            if self.rank == 0 and epoch % self.config["save_every"] == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(self.config.get("total_epochs") + 1)

    def _tb_log_histograms(self, epoch):
        for loss_name, loss in self.epoch_loss.items():
            if loss_name not in ["lphy"]:
                self.fluid_model.zero_grad()
                loss.backward(retain_graph=True)
                for name, param in self.fluid_model.named_parameters():
                    if "weight" in name and param.grad is not None:
                        if param.grad is not None:
                            self.writer.add_histogram(
                                f"Grad/{loss_name}/{name}", param.grad.cpu(), epoch
                            )

        # Handle physics loss separately
        self.fluid_model.zero_grad()
        self.epoch_loss["lphy"].backward(retain_graph=True)
        for name, param in self.fluid_model.named_parameters():
            if "weight" in name and param.grad is not None:
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Grad/ploss/{name}", param.grad.cpu(), epoch
                    )

        for name, param in self.fluid_model.named_parameters():
            if "weight" in name and param.grad is not None:
                self.writer.add_histogram(name, param.cpu().detach().numpy(), epoch)

    def _tb_log_scalars(self, epoch):

        dicLoss = {k: v for k, v in self.epoch_loss.items() if k != "lphy"}
        self.writer.add_scalars("lhistory", dicLoss, epoch)
        self.writer.add_scalars(
            "lphy",
            {"lphy": self.epoch_loss.get("lphy")},
            epoch,
        )

    def _run_epoch(self, epoch):
        pass  # Implement in derived classes

    def _compute_losses(self):
        pass  # Implement in derived classes

    #
    # Tensorboard specific logging functions
    #

    def _initialize_tensorboard(self):
        pass
        # self.writer = SummaryWriter(self.log_path)

    def track_training(self, epoch, elapsed_time):
        # Logger tracking
        printing.print_losses(self, epoch, elapsed_time)

        # # Tensorboard tracking
        # if self.writer is not None:
        #     self._tb_log_scalars(epoch)
        #     self._tb_log_histograms(epoch)

    def _save_checkpoint(self, epoch):
        model_path = os.path.join(self.log_path, "model.pth")

        state = {
            "model_state_dict": self.fluid_model.state_dict(),
            "optimizer_state_dict": self.fluid_optimizer.state_dict(),
            "loss_history": self.loss_history,
            "max_eig_hessian_bc_log": self.max_eig_hessian_bc_log,
            "max_eig_hessian_ic_log": self.max_eig_hessian_ic_log,
            "max_eig_hessian_res_log": self.max_eig_hessian_res_log,
            "epoch": epoch,
            "data_mean": self.train_dataloader.fluid_data.mean_x,
            "data_std": self.train_dataloader.fluid_data.std_x,
            "config": self.config,
            "model_path": model_path,
        }

        torch.save(state, model_path)
        self.logger.print("Final losses:")
        self.logger.print(
            " ".join(
                [
                    "Final %s: %0.3e | " % (key, self.loss_history[key][-1])
                    for key in self.config["loss_list"]
                ]
            )
        )

        if epoch == self.config.get("total_epochs"):
            self.logger.print("_summary of the model _")
            printing.print_config(self)

        self.logger.print(
            f"_save_checkpoint: [GPU:{self.rank}] Epoch {epoch} | Training checkpoint saved at {model_path}"
        )
