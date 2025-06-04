import torch
from src.utils.utils import flatten


def compute_gradients(model, loss):
    weights = list(model.parameters())
    # Compute gradients with respect to the model's parameters
    loss_gradients = torch.autograd.grad(
        loss,
        weights,
        create_graph=True,
        retain_graph=True,
    )

    # Apply torch.abs to each tensor in loss_gradients
    loss_gradients_abs = [torch.abs(g) for g in loss_gradients if g is not None]

    # Flatten all gradients into one tensor
    loss_gradients_flat = flatten(loss_gradients_abs)

    # Compute mean and max of the flattened gradients
    mean_grad = loss_gradients_flat.mean()
    max_grad = loss_gradients_flat.max()

    return max_grad, mean_grad


def update_weights_grad_stat(model, loss_list, alpha, weights, gamma):
    loss_grads = {}
    device = next(model.parameters()).device

    # Compute gradients for each loss
    for loss_name, loss in loss_list.items():
        loss_grads[loss_name] = compute_gradients(model, loss)

    if loss_grads:
        # Collect max gradients across all losses
        max_grads = torch.tensor(
            [grad[0] for grad in loss_grads.values()],
        ).to(device)
        mean_of_maxs = max_grads.mean()

        # Update weights for each loss
        weight_values = {}
        for loss in loss_list:
            if loss in loss_grads:
                grad_mean = loss_grads[loss][1]
                if grad_mean != 0:  # Prevent division by zero
                    weight_values[loss] = (
                        alpha * weights[loss].data
                        + (1.0 - alpha) * (mean_of_maxs / grad_mean).detach()
                    )
                else:
                    weight_values[loss] = (
                        alpha * weights[loss].data
                        + (1.0 - alpha) * (mean_of_maxs / 1.0e-7).detach()
                    )

        # Find the maximum weight value for normalization
        max_weight_value = max([w.max() for w in weight_values.values()])

        # Normalize and rescale the weights to be between 0 and gamma
        for loss in weight_values:
            weights[loss].data = (weight_values[loss] / max_weight_value) * gamma


def print_grad(model):
    loss_grads = {}
    # Compute gradients for each loss
    for loss_name in model.config["loss_list"]:
        loss_grads[loss_name] = model.compute_gradients(model.epoch_loss[loss_name])

    model.logger.print("*** Mean gradients***")
    model.logger.print(
        "".join(
            [
                f"{key}: {loss_grads[key][0].item(): 0.3e} | "
                for key in loss_grads.keys()
            ]
        )
    )


def print_weights(model):
    message = ", ".join(
        [
            f"{loss}: {weight.item():.3f}"
            for loss, weight in model.weights.values().items()
        ]
    )
    model.logger.print("Current Weights: " + message)
