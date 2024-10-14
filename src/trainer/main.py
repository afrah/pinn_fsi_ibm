from functools import partial
import sched
import sys

import torch

DATASET_PATH = "../data/Fluid_trainingData.mat"


def ddp_setup():
    local_rank = 0  # int(os.environ["LOCAL_RANK"])
    world_size = 4  # int(os.environ["WORLD_SIZE"])
    cuda = torch.cuda.is_available()
    print(f"ddp_setup: {0=},{world_size=}, {cuda=}")
    torch.cuda.set_device(0)

    return local_rank, world_size


def init_model_and_data(config, local_rank):
    """Initialize model and data based on the provided configuration."""

    def load_model_class(solver_name):
        solver_to_module = {
            "bspline": "src.nn.bspline",
            "tanh": "src.nn.tanh",
        }
        module = __import__(solver_to_module[solver_name], fromlist=["PINNKAN"])
        return getattr(module, "PINNKAN")

    if config.get("problem") == "IBM":
        from src.data.IBM_data_loader import IBM_data_loader

        train_dataloader = IBM_data_loader(DATASET_PATH, local_rank)
    else:
        print(
            "Error: Problem type is not set. Please provide a valid problem type (e.g., IBM)."
        )
        sys.exit(1)
    solver_name = config.get("solver")

    model_class = load_model_class(solver_name)
    model = model_class(config.get("network"), config.get("activation"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return train_dataloader, model, optimizer, scheduler


def main(config):
    """_summary_

    Args:
        config (_type_): _description_
    """

    local_rank, world_size = ddp_setup()
    train_dataloader, model, optimizer, scheduler = init_model_and_data(
        config, local_rank
    )

    if config.get("problem") == "IBM":
        from src.trainer import ibm_trainer_m1

        trainer = ibm_trainer_m1.Trainer(
            train_dataloader,
            model,
            optimizer,
            scheduler,
            local_rank,
            config,
        )

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
            "param_tanh",
            "tanh",
            "bspline",
        ],
        required=True,
        help="solver",
    )

    parser.add_argument(
        "--problem",
        choices=[
            "IBM",
        ],
        required=True,
        help="solver",
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
        "--network",
        required=True,
        type=partial(parse_list, data_type=int),  # Predefine data_type=int
        help="list of [input, nxhidden , output] weights (e.g., [2, 10, 10, 1])",
    )

    def parse_weights(weights_str):
        return [float(weight) for weight in weights_str.strip("[]").split(",")]

    args = parser.parse_args()

    if args.problem == "IBM":
        loss_list = [
            "lleft",
            "lright",
            "lbottom",
            "lup",
            "linitial",
            "lphy",
        ]

    # TODO
    configuration = {
        "batch_size": args.batch_size,
        "network": args.network,
        "weights": args.weights,
        "solver": args.solver,
        "problem": args.problem,
        "dataset_path": args.dataset_path,
        "total_epochs": args.total_epochs,
        "print_every": args.print_every,
        "save_every": args.save_every,
        "loss_list": loss_list,
        "log_path": args.log_path,
    }
    assert len(configuration.get("weights")) == len(
        configuration.get("loss_list")
    ), "Length of 'weights' and 'loss_list' must be equal."

    main(configuration)

    ### Hard-swish is not good with this code
    ### torch.sin is not good
