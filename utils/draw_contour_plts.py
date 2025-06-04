import numpy as np
from src.utils.plotting_regular_2D import plot_time_profile_regular_data_IBM
from src.utils.plotting_regular_2D import draw_contourf_regular_2D
from src.utils.plotting_irregular_2D_interface import draw_contourf_irregular_2D


def plot_time_profile_regular_data_IBM_uvp(
    x_fluid,
    y_fluid,
    time_fluid,
    u_fluid,
    u_pred_fluid,
    v_fluid,
    v_pred_fluid,
    p_fluid,
    p_pred_fluid,
    fx_fluid,
    fx_pred_fluid,
    fy_fluid,
    fy_pred_fluid,
    output_dir,
    tstep=101,
    xstep=102,
    ystep=102,
):
    img_width, img_height = 20, 12
    steps = [tstep, xstep, ystep]
    txy = [x_fluid, y_fluid, time_fluid]

    plot_time_profile_regular_data_IBM(
        output_dir,
        steps,
        txy,
        u_fluid,
        u_pred_fluid,
        "u",
        img_width,
        img_height,
    )

    plot_time_profile_regular_data_IBM(
        output_dir,
        steps,
        txy,
        v_fluid,
        v_pred_fluid,
        "v",
        img_width,
        img_height,
    )
    plot_time_profile_regular_data_IBM(
        output_dir,
        steps,
        txy,
        p_fluid,
        p_pred_fluid,
        "p",
        img_width,
        img_height,
    )
    plot_time_profile_regular_data_IBM(
        output_dir,
        steps,
        txy,
        fx_fluid,
        fx_pred_fluid,
        "F_x",
        img_width,
        img_height,
    )
    plot_time_profile_regular_data_IBM(
        output_dir,
        steps,
        txy,
        fy_fluid,
        fy_pred_fluid,
        "F_y",
        img_width,
        img_height,
    )


def draw_contour_plts(
    time_fluid,
    x_fluid,
    y_fluid,
    u_fluid,
    u_pred_fluid,
    v_fluid,
    v_pred_fluid,
    p_fluid,
    p_pred_fluid,
    fx_fluid,
    fx_pred_fluid,
    fy_fluid,
    fy_pred_fluid,
    model_dirname,
    tstep=101,
    xstep=102,
    ystep=102,
):
    #  t = tf.reshape(tstep,N_data)[:,0].T
    tf100 = time_fluid.reshape(tstep, xstep, ystep)[:, 0, 0]
    fluid_x = x_fluid.reshape(tstep, xstep, ystep)[0, :, 0]  # .reshape(100,100)[0,:]
    fluid_y = y_fluid.reshape(tstep, xstep, ystep)[0, 0, :]  # .reshape(100,100)[:,0]

    u_fluid_cfd_reshape = u_fluid.reshape(tstep, xstep, ystep).transpose((0, 2, 1))
    v_fluid_cfd_reshape = v_fluid.reshape(tstep, xstep, ystep).transpose((0, 2, 1))
    p_fluid_cfd_reshape = p_fluid.reshape(tstep, xstep, ystep).transpose((0, 2, 1))
    fx_fluid_cfd_reshape = fx_fluid.reshape(tstep, xstep, ystep).transpose((0, 2, 1))
    fy_fluid_cfd_reshape = fy_fluid.reshape(tstep, xstep, ystep).transpose((0, 2, 1))

    u_fluid_pred_reshape = u_pred_fluid.reshape(tstep, xstep, ystep).transpose(
        (0, 2, 1)
    )
    v_fluid_pred_reshape = v_pred_fluid.reshape(tstep, xstep, ystep).transpose(
        (0, 2, 1)
    )
    p_fluid_pred_reshape = p_pred_fluid.reshape(tstep, xstep, ystep).transpose(
        (0, 2, 1)
    )
    fx_fluid_pred_reshape = fx_pred_fluid.reshape(tstep, xstep, ystep).transpose(
        (0, 2, 1)
    )
    fy_fluid_pred_reshape = fy_pred_fluid.reshape(tstep, xstep, ystep).transpose(
        (0, 2, 1)
    )

    u_error_fluid = np.abs(u_fluid_cfd_reshape - u_fluid_pred_reshape)
    v_error_fluid = np.abs(v_fluid_cfd_reshape - v_fluid_pred_reshape)
    p_error_fluid = np.abs(p_fluid_cfd_reshape - p_fluid_pred_reshape)
    fx_error_fluid = np.abs(fx_fluid_cfd_reshape - fx_fluid_pred_reshape)
    fy_error_fluid = np.abs(fy_fluid_cfd_reshape - fy_fluid_pred_reshape)

    fluid_data = [
        u_fluid_pred_reshape,
        v_fluid_pred_reshape,
        p_fluid_pred_reshape,
        fx_fluid_pred_reshape,
        fy_fluid_pred_reshape,
        u_fluid_cfd_reshape,
        v_fluid_cfd_reshape,
        p_fluid_cfd_reshape,
        fx_fluid_cfd_reshape,
        fy_fluid_cfd_reshape,
        u_error_fluid,
        v_error_fluid,
        p_error_fluid,
        fx_error_fluid,
        fy_error_fluid,
    ]

    nrows_ncols = (3, 5)
    time_values = [99, 90, 70, 50, 20, 0]
    titles = [
        "u_pinn",
        "v_pinn",
        "p_pinn",
        "Fx_pinn",
        "Fy_pinn",
        "u_cfd",
        "v_cfd",
        "p_cfd",
        "Fx_cfd",
        "Fy_cfd",
        "error_u",
        "error_v",
        "error_p",
        "error_Fx",
        "error_Fy",
    ]

    draw_contourf_regular_2D(
        tf100,
        fluid_x,
        fluid_y,
        fluid_data,
        titles,
        nrows_ncols,
        time_values,
        1.0,
        1.0,
        model_dirname,
        20,
        12,
        ticks=3,
        fontsize=16,
        labelsize=16,
        axes_pad=(1.4, 0.8),  # Set padding for x and y axes separately,
    )


def draw_interface_contour_plts(
    time_interface,
    x_interface,
    y_interface,
    u_interface,
    u_pred_interface,
    v_interface,
    v_pred_interface,
    p_interface,
    p_pred_interface,
    fx_interface,
    fx_pred_interface,
    fy_interface,
    fy_pred_interface,
    model_dirname,
    nrows_ncols,
    time_steps,
):

    tstep = 101
    N_data = int(time_interface.shape[0] / tstep)

    #  t = tf.reshape(tstep,N_data)[:,0].T
    tf100 = time_interface.reshape(tstep, N_data)[:, 0].T
    interface_x = x_interface.reshape(tstep, N_data)[0, :]
    interface_y = y_interface.reshape(tstep, N_data)[0, :]

    u_interface_reshape = u_interface.reshape(tstep, N_data)
    v_interface_reshape = v_interface.reshape(tstep, N_data)
    p_interface_reshape = p_interface.reshape(tstep, N_data)
    fx_interface_reshape = fx_interface.reshape(tstep, N_data)
    fy_interface_reshape = fy_interface.reshape(tstep, N_data)

    u_pred_interface_reshape = u_pred_interface.reshape(tstep, N_data)
    v_pred_interface_reshape = v_pred_interface.reshape(tstep, N_data)
    p_pred_interface_reshape = p_pred_interface.reshape(tstep, N_data)
    fx_pred_interface_reshape = fx_pred_interface.reshape(tstep, N_data)
    fy_pred_interface_reshape = fy_pred_interface.reshape(tstep, N_data)

    u_error_interface = np.abs(u_interface_reshape - u_pred_interface_reshape)
    v_error_interface = np.abs(v_interface_reshape - v_pred_interface_reshape)
    p_error_interface = np.abs(p_interface_reshape - p_pred_interface_reshape)
    fx_error_interface = np.abs(fx_interface_reshape - fx_pred_interface_reshape)
    fy_error_interface = np.abs(fy_interface_reshape - fy_pred_interface_reshape)

    interface_data = [
        u_pred_interface_reshape,
        v_pred_interface_reshape,
        p_pred_interface_reshape,
        fx_pred_interface_reshape,
        fy_pred_interface_reshape,
        u_interface_reshape,
        v_interface_reshape,
        p_interface_reshape,
        fx_interface_reshape,
        fy_interface_reshape,
        u_error_interface,
        v_error_interface,
        p_error_interface,
        fx_error_interface,
        fy_error_interface,
    ]

    titles = [
        "u_pinn",
        "v_pinn",
        "p_pinn",
        "fx_pinn",
        "fy_pinn",
        "u_cfd",
        "v_cfd",
        "p_cfd",
        "fx_cfd",
        "fy_cfd",
        "u_error",
        "v_error",
        "p_error",
        "fx_error",
        "fy_error",
    ]

    fig_size = (25, 25)
    draw_contourf_irregular_2D(
        tf100,
        interface_x,
        interface_y,
        interface_data,
        titles,
        model_dirname,
        nrows_ncols,
        time_steps,
        ticks=5,
        fontsize=10.5,
        labelsize=7,
        axes_pad=1,
    )
