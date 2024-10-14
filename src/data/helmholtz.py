import numpy as np
import scipy
import scipy.io
import torch
from pyDOE import lhs

from src.utils.utils import tonp


def generate_dataset(device):
    file_path = "../../data/"
    file_name = "losses.txt"

    data = scipy.io.loadmat(file_path + "AC.mat")
    Exact = data["uu"]
    Exact0 = np.real(Exact)
    t0 = data["tt"].flatten()[:, None]
    x0 = data["x"].flatten()[:, None]
    lbc = torch.tensor([x0.min(), t0.min()]).to(torch.float32).to(device)
    ubc = torch.tensor([x0.max(), t0.max()]).to(torch.float32).to(device)

    xdim, tdim = 4.0, 1.0

    nx, nt = (1001, 1001)
    x = np.linspace(-1, 1, nx)
    t = np.linspace(-1, 1, nt)
    x0, t0 = np.meshgrid(x, t)
    Exact0 = np.sin(xdim * np.pi * x0) * np.sin(tdim * np.pi * t0)
    Exact0 = Exact0.T
    x0, t0 = x[:, None], t[:, None]
    lbc = torch.tensor([x0.min(), t0.min()]).to(torch.float32).to(device)
    ubc = torch.tensor([x0.max(), t0.max()]).to(torch.float32).to(device)

    # Collocation points
    dimx = nx
    dimt = nt
    N_u = 50
    N_r = 25600

    # Definition
    Exact = Exact0.T
    tm = np.linspace(t0.min(), t0.max(), dimt)[:, None]
    xm = np.linspace(x0.min(), x0.max(), dimx)[:, None]
    X, T = np.meshgrid(xm, tm)

    # Doman bounds
    lb = tonp(lbc)
    ub = tonp(ubc)
    xx1 = np.hstack((x0 * 0 - 1, t0))
    uu1 = np.zeros_like(x0)

    # Top/bot boundaries
    xx2 = np.hstack((x0, t0 * 0 + 1))
    xx3 = np.hstack((x0, t0 * 0 - 1))
    xx4 = np.hstack((x0 * 0 + 1, t0))
    uu2 = uu1
    uu3 = uu1
    uu4 = uu1

    # Random choice
    idx0 = np.random.choice(dimt, N_u, replace=False)
    idx = np.random.choice(dimt, N_u, replace=False)
    idx2 = np.random.choice(dimt, N_u, replace=False)
    idx3 = np.random.choice(dimt, N_u, replace=False)

    u_train = np.vstack([uu1[idx0, :], uu2[idx, :], uu3[idx2, :], uu4[idx3, :]])
    X_u_train = np.vstack([xx1[idx0, :], xx2[idx, :], xx3[idx2, :], xx4[idx3, :]])

    # Collocation points
    X_r_train = lb + (ub - lb) * lhs(2, N_r)

    print("X_r shape:", X_r_train.shape)
    print("X_u shape:", X_u_train.shape)
    print("u shape:", u_train.shape)

    return nx, nt, t0, x0, X_r_train, X_u_train, u_train, Exact0, lbc, ubc
    # plt.figure(1)
    # plt.title("Collocation points")
    # plt.scatter(X_r_train[:, 1], X_r_train[:, 0], s=0.5)
    # plt.scatter(X_u_train[:, 1], X_u_train[:, 0], s=0.5, c="k")
    # plt.show()
