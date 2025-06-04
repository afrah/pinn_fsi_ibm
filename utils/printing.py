def print_losses(
    epoch_loss,
    config,
    running_time,
    lr,
    epoch,
    logger,
    elapsed,
    max_eig_hessian_bc_log=None,
    max_eig_hessian_ic_log=None,
    max_eig_hessian_res_log=None,
):

    tloss = sum(epoch_loss[loss].item() for loss in config["loss_list"])

    running_time += elapsed
    message = ""
    message += "".join(
        f"{loss}: {epoch_loss.get(loss).item():.3e} | " for loss in config["loss_list"]
    )

    additional_message = (
        f" Epoch: {epoch} | Time: {elapsed:.2f}s | rTime: {running_time:.3e}h | "
        f"LR: {lr:.3e} |loss: {tloss:.3e} | "
    )

    if max_eig_hessian_bc_log:
        additional_message += f"max_eigH_bc: {max_eig_hessian_bc_log[-1]:.3e} | "
    if max_eig_hessian_ic_log:
        additional_message += f"max_eigH_ic: {max_eig_hessian_ic_log[-1]:.3e} | "
    if max_eig_hessian_res_log:
        additional_message += f"max_eigH_res: {max_eig_hessian_res_log[-1]:.3e} | "

    final_message = additional_message + message
    logger.print(final_message)


def print_config(model):
    model.logger.print("model configuration:")
    for key, value in model.config.items():
        model.logger.print(f"{key} : {value}")


# model.optimizer_fluid.param_groups[0]['lr']
