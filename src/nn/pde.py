import torch


def navier_stokes_2D_IBM(fluid_model, t, x, y):
    """_summary_

    Args:
        txy (_type_): _description_
        uvwp_pred (_type_): _description_
        nu (_type_): _description_

    Returns:
        _type_: _description_
    """
    t.requires_grad_(True)
    x.requires_grad_(True)
    y.requires_grad_(True)

    inputs = torch.cat([t, x, y], dim=1).squeeze(1)
    output = fluid_model(inputs)
    DENSITY = 1.0
    mu = 0.01  # kinematic viscosity
    u = output[:, 0]
    v = output[:, 1]
    pressure = output[:, 2]

    # First Derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[
        0
    ]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[
        0
    ]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[
        0
    ]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[
        0
    ]
    p_x = torch.autograd.grad(
        pressure,
        x,
        grad_outputs=torch.ones_like(pressure),
        create_graph=True,
    )[0]
    p_y = torch.autograd.grad(
        pressure,
        y,
        grad_outputs=torch.ones_like(pressure),
        create_graph=True,
    )[0]

    # Second Derivatives
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0]

    v_xx = torch.autograd.grad(
        v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True
    )[0]
    v_yy = torch.autograd.grad(
        v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True
    )[0]

    # Navier-Stokes Equations Residuals
    continuity = u_x + v_y

    f_u = u_t + (u * u_x + v * u_y) + 1.0 / DENSITY * p_x - mu * (u_xx + u_yy)  # - fx
    f_v = v_t + (u * v_x + v * v_y) + 1.0 / DENSITY * p_y - mu * (v_xx + v_yy)  # - fy

    return [continuity, f_u, f_v]


def force_stress_2D_IBM(
    txy, fluid_model, force_model, fluid_min, fluid_max, solid_min, solid_max
):
    txy.requires_grad_(True)

    # Variables

    ttime = txy[:, 0]
    x = txy[:, 1]
    y = txy[:, 2]
    p = txy[:, 3]

    uvp_pred = fluid_model.forward(
        torch.stack([ttime, x, y], dim=1), fluid_min, fluid_max
    )

    fs_pred = force_model.forward(
        torch.stack([ttime, x, y, p], dim=1), solid_min, solid_max
    )

    SOLID_DENSITY = 1.0
    u = uvp_pred[:, 0]
    v = uvp_pred[:, 1]

    fx = fs_pred[:, 0]
    fy = fs_pred[:, 1]

    # First Derivatives
    u_t = torch.autograd.grad(
        u, ttime, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    v_t = torch.autograd.grad(
        v, ttime, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    r_u = SOLID_DENSITY * u_t - p - fx
    r_v = SOLID_DENSITY * v_t - p - fy

    return [r_u, r_v]
