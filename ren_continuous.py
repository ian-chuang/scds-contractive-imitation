"""
    Code is modified from the original version by Martinelli et. al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

from ren import REN

from bijection import BijectionNet


class CREN(REN):
    def __init__(self, dim_in: int, dim_out: int, dim_x: int, dim_v: int,
                 batch_size: int, posdef_tol: float = 5.0e-2, contraction_rate_lb: float = 0.0,
                 add_bias: bool = False, linear_output: bool = True, device: str = "cpu",
                 weight_init_std: float = 0.5, horizon: int = None, bijection: bool = False,
                 num_bijection_layers: int = 0):
        """ Initialize a recurrent equilibrium network. This can also be viewed as a single layer
        of a larger network.

        Args:
            dim_in (int): Input dimension. The input is the u vector defined in the paper.
            dim_out (int): Output dimension.
            dim_x (int): Internal state dimension. This state evolves with contraction properties.
            dim_v (int): Complexity of the implicit layer.

            batch_size(int, optional): Parallel batch size for efficient computing. Defaults to 1.
            weight_init_std (float, optional): Weight initialization. Set to 0.1 by default.

            linear_output (bool, optional): If set True, the output matrices are arranged in a way so that
                the output is a linear transformation of the input. Defaults to False.
            add_bias (bool, optional): If set True, the trainable b_xvy biases are added. Defaults to False.

            posdef_tol (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 0.
            device(str, optional): Pass the name of the device. Defaults to cpu.
        """
        super().__init__(dim_in, dim_out, dim_x, dim_v, batch_size, weight_init_std, linear_output, posdef_tol,
                         contraction_rate_lb, add_bias, device, horizon, bijection, num_bijection_layers)

        # auxiliary matrices
        self.Pstar = nn.Parameter(torch.randn(dim_x, dim_x, device=device) * self.weight_init_std)
        self.Chi = nn.Parameter(torch.randn(dim_x, dim_v, device=device) * self.weight_init_std)
        self.X = nn.Parameter(torch.randn(dim_x + dim_v, dim_x + dim_v, device=device) * self.weight_init_std)
        self.Y1 = nn.Parameter(torch.randn(dim_x, dim_x, device=device) * self.weight_init_std)
        self.B2 = nn.Parameter(torch.randn(dim_x, dim_in, device=device) * self.weight_init_std)
        self.D12 = nn.Parameter(torch.randn(dim_v, dim_in, device=device) * self.weight_init_std)
        self.C2 = nn.Parameter(torch.randn(dim_out, dim_x, device=device) * self.weight_init_std)

        # linear output setup
        if linear_output:
            self.D21 = torch.zeros(dim_out, dim_v, device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(dim_out, dim_v, device=device) * self.weight_init_std)

        self.D22 = nn.Parameter(torch.randn(dim_out, dim_in, device=device) * self.weight_init_std)

        # non-trainable matrices
        self.A = torch.zeros(dim_x, dim_x, device=device)
        self.D11 = torch.zeros(dim_v, dim_v, device=device)
        self.C1 = torch.zeros(dim_v, dim_x, device=device)
        self.B1 = torch.zeros(dim_x, dim_v, device=device)
        self.P = torch.zeros(dim_x, dim_x, device=device)

        # update model parameters
        self.update_model_param()

    def update_model_param(self):
        """ Used at the end of each batch training for the update of the constrained matrices.
        """
        P = 0.5 * F.linear(self.Pstar, self.Pstar) + self.epsilon * torch.eye(self.dim_x, device=self.device)
        self.P = P

        H = F.linear(self.X, self.X) + self.epsilon * torch.eye(self.dim_x + self.dim_v, device=self.device)

        # partition of H into [H1 H2; H3 H4]
        h1, h2 = torch.split(H, (self.dim_x, self.dim_v), dim=0)
        H1, H2 = torch.split(h1, (self.dim_x, self.dim_v), dim=1)
        H3, H4 = torch.split(h2, (self.dim_x, self.dim_v), dim=1)

        Y = -0.5 * (H1 + self.contraction_rate_lb * P + self.Y1 - self.Y1.T)
        Lambda = 0.5 * torch.diag_embed(torch.diagonal(H4))

        self.A = F.linear(torch.inverse(P), Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda), torch.tril(H4, -1).T)
        self.C1 = F.linear(torch.inverse(Lambda), self.Chi)
        self.eye = torch.eye(self.dim_v, device=self.device)

        Z = -H2 - self.Chi
        self.B1 = F.linear(torch.inverse(P), Z.T)

    def forward(self, t, x):
        """ Forward pass of the network.

        Args:
            t (optional): Time variable according to the NeuralODE framework.
            x (torch.Tensor): Input data for the forward pass.

        Returns:
            torch.Tensor: Time derivative of x.
        """
        u_in = torch.zeros((1, 2), device=self.device) # TODO: Remove this line and use the input argument

        w = torch.zeros(self.x.size(0), 1, self.dim_v, device=self.device)

        for i in range(self.dim_v):
            v = F.linear(x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i, :])
            w = w + (self.eye[i, :] * self.act(v)).unsqueeze(1)

        # state evolution
        x_dot = (F.linear(x, self.A) + F.linear(w, self.B1) + F.linear(u_in, self.B2))

        return x_dot

    def output(self, x):
        """ Calculates the output yt given the state xi and the input u.

        This is reduced to a single transformation applied via the C2 matrix which is trained during training.
        The linear transformation preserves contraction in the target space, if the latent space is contractive.
        """

        y_out = F.linear(x, self.C2)

        # apply a bijection net
        if self.bijection:
            y_out = self.bijection_net(y_out) - self.bijection_net(torch.zeros(y_out.shape, device=self.device))

        return y_out


    def forward_trajectory(self, u_in: torch.Tensor, y_init: torch.Tensor, horizon: int):
        """ Get a trajectory of forward passes.

        First element can be either y_init, as used here, or y_1. Depends on the application.

        Args:
            u_in (torch.Tensor): Input at each time step. Must be fixed for autonomous systems.
            y_init (torch.Tensor): Initial condition of the output.
            horizon (int, optional): Length of the forward trajectory. Defaults to 20.
        """

        self.set_y_init(y_init)

        # discrete horizon
        self.horizon = horizon
        time_vector = torch.linspace(0.0, 1.0, horizon, device=self.device)

        x_sim = odeint(self, self.x, time_vector, method='dopri5', rtol=1e-4, atol=1e-4,
                       adjoint_rtol=1e-4, adjoint_atol=1e-4)
        out = self.output(x_sim)
        out = out.reshape(out.shape[1], out.shape[0], out.shape[3]) # TODO: Fix the reshape problem and dimensions in CREN

        return out


