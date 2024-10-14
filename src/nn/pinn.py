import os
import sys
import numpy as np
import scipy
import scipy.io
import torch
from src.nn.dnn import DNN
import time
from torch.optim import lr_scheduler

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.utils import tonp, grad

layers = [2] + [128] * 6 + [1]


class PINN:
    """PINN Class"""

    def __init__(self, X_u, u, X_r, nx, nt, t0, x0, Exact0, device="cpu"):

        # Initialization
        self.rba = 1  # RBA weights
        self.sa = 0  # SA weights
        self.iter = 0
        self.exec_time = 0
        self.print_step = 1000
        self.dimx, self.dimt = nx, nt
        self.dimx_, self.dimt_ = nx, nt  # solution dim
        self.first_opt = 20000
        self.freeze = self.first_opt
        self.it, self.l2, self.linf = [], [], []
        self.loss, self.losses = None, []
        self.opt = 1

        # Intermediate results
        self.Exact = Exact0
        X, T = np.meshgrid(x0, t0)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.xx = torch.tensor(X_star[:, 0:1]).float().to(device)
        self.tt = torch.tensor(X_star[:, 1:2]).float().to(device)

        # Data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True).float().to(device)
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.ub = self.u[self.dimx :]
        self.u = self.u[: self.dimx]
        self.N_r = tonp(self.x_r).size
        self.N_u = tonp(self.u).size
        self.dnn = DNN(layers).to(device)

        # RBA initialization
        if self.rba == 1:
            self.rsum = 0
            self.eta = 0.001
            self.gamma = 0.999
            self.init = 1  # initialization mode (1 or 2)

        # SA weights initialization
        if self.sa == 1:
            self.lamr = (
                torch.rand(self.N_r, 1, requires_grad=True).float().to(device) * 1
            )
            self.lamu = (
                torch.rand(self.N_u, 1, requires_grad=True).float().to(device) * 1
            )
            self.lamr = torch.nn.Parameter(self.lamr)
            self.lamu = torch.nn.Parameter(self.lamu)
            # Optimizer2 (SA weights)
            self.optimizer2 = torch.optim.Adam(
                [self.lamr] + [self.lamu], lr=0.005, maximize=True
            )

        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(
            self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999)
        )
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9, verbose=True
        )
        self.step_size = 5000

    def net_u(self, x, t):
        """Get the velocities"""

        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_r(self, x, t):
        """Get the residuals"""

        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        a1 = self.dimx
        a2 = self.dimt
        ksq = 1.0
        force = (
            -((a1 * np.pi) ** 2) * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * t)
            - (a2 * np.pi) ** 2 * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * t)
            + ksq * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * t)
        )
        f = u_xx + u_tt + ksq * u - force
        return f, u_x

    def loss_func(self):
        """Loss function"""

        self.optimizer.zero_grad()
        if self.sa == 1:
            self.optimizer2.zero_grad()

        # Predictions
        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, u_x_pred = self.net_r(self.x_r, self.t_r)

        if self.rba == True:
            if self.init == 2 and self.iter == 0:
                eta = 1
            else:
                eta = self.eta
            r_norm = eta * torch.abs(self.r_pred) / torch.max(torch.abs(self.r_pred))
            self.rsum = (self.rsum * self.gamma + r_norm).detach()
            loss_r = torch.mean((self.rsum * self.r_pred) ** 2)
            loss_u = torch.mean((self.u_pred - self.u) ** 2)

        elif self.sa == True and self.iter < self.freeze:
            loss_r = torch.mean((self.lamr * self.r_pred) ** 2)
            loss_u = torch.mean((self.lamu * (self.u_pred - self.u)) ** 2)

        else:
            loss_r = torch.mean(self.r_pred**2)
            loss_u = torch.mean((self.u_pred - self.u) ** 2)

        # Loss calculation
        self.loss = loss_r + loss_u
        self.loss.backward()
        self.iter += 1

        if self.iter % self.print_step == 0:

            with torch.no_grad():
                # Grid prediction (for relative L2)
                res = self.net_u(self.xx, self.tt)
                sol = tonp(res)
                sol = np.reshape(sol, (self.dimt_, self.dimx_)).T

                # L2 calculation
                l2_rel = np.linalg.norm(
                    self.Exact.flatten() - sol.flatten(), 2
                ) / np.linalg.norm(self.Exact.flatten(), 2)
                l_inf = np.linalg.norm(
                    self.Exact.flatten() - sol.flatten(), np.inf
                ) / np.linalg.norm(self.Exact.flatten(), np.inf)
                print(
                    "Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, t/iter: %.1e"
                    % (self.iter, self.loss.item(), l2_rel, l_inf, self.exec_time)
                )

                self.it.append(self.iter)
                self.l2.append(l2_rel)
                self.linf.append(l_inf)

        # Optimizer properties
        if self.opt == 1:
            self.optimizer.step()
            if self.sa == True:
                self.optimizer2.step()
        elif self.opt == 2:
            return self.loss

    def train(self):
        """Train model"""

        self.dnn.train()
        for epoch in range(self.first_opt):
            start_time = time.time()
            self.loss_func()
            end_time = time.time()
            self.exec_time = end_time - start_time
            if (epoch + 1) % self.step_size == 0:
                self.scheduler.step()

        self.opt = 2
        print("LBFGS switch")

        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.8,
            max_iter=100,
            # max_eval=50000,
            # history_size=50,
            tolerance_grad=1.0e-5,
            tolerance_change=1.0e-9,
            # line_search_fn="strong_wolfe"
        )
        for epoch in range(20):
            self.optimizer.step(self.loss_func)

        # Write data
        a = np.array(self.it)
        b = np.array(self.l2)
        c = np.array(self.linf)
        # Stack them into a 2D array.
        d = np.column_stack((a, b, c))
        np.savetxt("losses.txt", d, fmt="%.10f %.10f %.10f")

    def predict(self, X):
        x = torch.tensor(X[:, 0:1]).float().to(X.device)
        t = torch.tensor(X[:, 1:2]).float().to(X.device)
        self.dnn.eval()
        u = self.net_u(x, t)
        u = tonp(u)
        return u
