from functools import partial
import sched
import sys

import torch

DATASET_PATH = "/home/vlq26735/afrah/code/pinn_fsi_ibm/data/Fluid_trainingData.mat"


def ddp_setup():
    local_rank = 0  # int(os.environ["LOCAL_RANK"])
    world_size = 4  # int(os.environ["WORLD_SIZE"])
    cuda = torch.cuda.is_available()
    print(f"ddp_setup: {0=},{world_size=}, {cuda=}")
    torch.cuda.set_device(1)

    return local_rank, world_size


def init_model_and_data(config, local_rank):
    """Initialize model and data based on the provided configuration."""

    def load_model_class(solver_name):
        solver_to_module = {
            "xsig": "src.nn.xsigmoid",
            "tanh": "src.nn.tanh",
            "bspline": "src.nn.bspline",
        }
        module = __import__(solver_to_module[solver_name], fromlist=["PINNKAN"])
        return getattr(module, "PINNKAN")

    if config.get("problem") == "fsi":
        from src.data.IBM_data_loader import IBM_data_loader

        train_dataloader = IBM_data_loader(DATASET_PATH, local_rank)
    else:
        print(
            "Error: Problem type is not set. Please provide a valid problem type (e.g., fsi)."
        )
        sys.exit(1)
    solver_name = config.get("solver")

    model_class = load_model_class(solver_name)
    model_fluid = model_class(config.get("network_fluid"), config.get("activation"))
    model_force = model_class(config.get("network_force"), config.get("activation"))

    optimizer_fluid = torch.optim.Adam(
        list(model_fluid.parameters()),
        lr=0.005,
        weight_decay=1e-8,
    )

    optimizer_force = torch.optim.Adam(
        list(model_force.parameters()),
        lr=0.005,
        weight_decay=1e-8,
    )
    scheduler_fluid = torch.optim.lr_scheduler.StepLR(
        optimizer_fluid, step_size=5000, gamma=0.85
    )
    scheduler_force = torch.optim.lr_scheduler.StepLR(
        optimizer_force, step_size=5000, gamma=0.85
    )
    return (
        train_dataloader,
        model_fluid,
        model_force,
        optimizer_fluid,
        optimizer_force,
        scheduler_fluid,
        scheduler_force,
    )


def main(config):
    """_summary_

    Args:
        config (_type_): _description_
    """

    local_rank, world_size = ddp_setup()
    (
        train_dataloader,
        model_fluid,
        model_force,
        optimizer_fluid,
        optimizer_force,
        scheduler_fluid,
        scheduler_force,
    ) = init_model_and_data(config, local_rank)

    if config.get("problem") == "fsi":
        if config.get("weighting") == "Fixed":
            from src.trainer_M5 import ibm_trainer_Fixed

            trainer = ibm_trainer_Fixed.Trainer(
                train_dataloader,
                model_fluid,
                model_force,
                optimizer_fluid,
                optimizer_force,
                scheduler_fluid,
                scheduler_force,
                local_rank,
                config,
            )
        elif config.get("weighting") == "RBA":
            from src.trainer_M5 import ibm_trainer_RBA

            trainer = ibm_trainer_RBA.Trainer(
                train_dataloader,
                model_fluid,
                model_force,
                optimizer_fluid,
                optimizer_force,
                scheduler_fluid,
                scheduler_force,
                local_rank,
                config,
            )
        elif config.get("weighting") == "SA":
            from src.trainer_M5 import ibm_trainer_SA

            trainer = ibm_trainer_SA.Trainer(
                train_dataloader,
                model_fluid,
                model_force,
                optimizer_fluid,
                optimizer_force,
                scheduler_fluid,
                scheduler_force,
                local_rank,
                config,
            )

        elif config.get("weighting") == "grad_stat":
            from src.trainer_M5 import ibm_trainer_grad_stat

            trainer = ibm_trainer_grad_stat.Trainer(
                train_dataloader,
                model_fluid,
                model_force,
                optimizer_fluid,
                optimizer_force,
                scheduler_fluid,
                scheduler_force,
                local_rank,
                config,
            )
        else:
            print(
                "Error: Weighting type is not set. Please provide a valid weighting type (e.g., RBA or SA)."
            )
            sys.exit(1)
    if local_rank == 0:
        print(f"DATA_FILE: {config.get('dataset_path')=}")
    trainer.train_mini_batch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("--save_every", type=int, help="How often to save a snapshot")
    parser.add_argument("--print_every", type=int, help="How often to print a snapshot")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Input batch size on each device (default: 32)",
    )

    parser.add_argument(
        "--log_path",
        type=str,
        help="Path to save log files at",
    )

    parser.add_argument(
        "--solver",
        choices=[
            "tanh",
            "xsig",
            "bspline",
        ],
        required=True,
        help="solver",
    )

    parser.add_argument(
        "--activation",
        choices=[
            "tanh",
            "xsig",
        ],
        required=True,
        help="activation",
    )

    parser.add_argument(
        "--weighting",
        choices=[
            "RBA",
            "SA",
            "Fixed",
            "grad_stat",
        ],
        required=True,
        help="weighting",
    )

    parser.add_argument(
        "--problem",
        choices=[
            "fsi",
        ],
        required=True,
        help="problem",
    )

    def parse_list(weights_str, data_type):
        if data_type not in [int, float]:
            raise ValueError("data_type must be either 'int' or 'float'")

        parsed_list = []
        for weight in weights_str.strip("[]").split(","):
            weight = weight.strip()  # Remove any leading/trailing whitespace
            parsed_list.append(data_type(weight))  # Convert to the specified data type
        return parsed_list

    parser.add_argument(
        "--network_force",
        required=True,
        type=partial(parse_list, data_type=int),  # Predefine data_type=int
        help="list of [input, nxhidden , output] weights (e.g., [2, 10, 10, 1])",
    )
    parser.add_argument(
        "--network_fluid",
        required=True,
        type=partial(parse_list, data_type=int),  # Predefine data_type=int
        help="list of [input, nxhidden , output] weights (e.g., [2, 10, 10, 1])",
    )

    def parse_weights(weights_str):
        return [float(weight) for weight in weights_str.strip("[]").split(",")]

    args = parser.parse_args()

    if args.problem == "fsi":
        loss_list = [
            "left",
            "right",
            "bottom",
            "up",
            "fluid_points",
            "initial",
            "fluid",
            "vCoupling",
            "lint_pts",
            "int_initial",
        ]

    configuration = {
        "batch_size": args.batch_size,
        "network_fluid": args.network_fluid,
        "network_force": args.network_force,
        "activation": args.activation,
        "solver": args.solver,
        "weighting": args.weighting,
        "problem": args.problem,
        "total_epochs": args.total_epochs,
        "print_every": args.print_every,
        "save_every": args.save_every,
        "loss_list": loss_list,
        "log_path": args.log_path,
    }

    main(configuration)

    ### Hard-swish is not good with this code
    ### torch.sin is not good
