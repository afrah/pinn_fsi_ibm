import torch


def tonp(tensor):
    """Torch to Numpy"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(
            "Unknown type of input, expected torch.Tensor or "
            "np.ndarray, but got {}".format(type(input))
        )


def grad(u, x):
    """Get grad"""
    gradient = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    return gradient
