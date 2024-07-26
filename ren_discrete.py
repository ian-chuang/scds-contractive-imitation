"""
    Code is modified from the original version by Galimberti et. al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ren import REN


class DREN(REN):
    def __init__(self, dim_in: int, dim_out: int, dim_x: int, dim_v: int,
                 batch_size: int = 1, weight_init_std: float = 0.5, linear_output: bool = False,
                 posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0, add_bias: bool = False,
                 device: str = "cpu", horizon: int = None, bijection: bool = False,
                 num_bijection_layers: int = 0):
        """ Initialize a recurrent equilibrium network. This can also be viewed as a single layer
        of a larger network.

        NOTE: The equations for REN upon which this class is built can be found in the following paper
        "Revay M et al. Recurrent equilibrium networks: Flexible dynamic models with guaranteed
        stability and robustness."

        The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
        The model is described as,

                        [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                        [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_v ]
                        [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_y ]

        where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
        are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

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
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
            device(str, optional): Pass the name of the device. Defaults to cpu.
        """
        super().__init__(dim_in, dim_out, dim_x, dim_v, batch_size, weight_init_std, linear_output, posdef_tol,
                         contraction_rate_lb, add_bias, device, horizon, bijection, num_bijection_layers)

        # auxiliary matrices
        self.X_shape = (2 * self.dim_x + self.dim_v, 2 * self.dim_x + self.dim_v)
        self.Y_shape = (self.dim_x, self.dim_x)

        # nn state dynamics
        self.B2_shape = (self.dim_x, self.dim_in)

        # nn output
        self.C2_shape = (self.dim_out, self.dim_x)
        self.D21_shape = (self.dim_out, self.dim_v)
        self.D22_shape = (self.dim_out, self.dim_in)

        # v signal
        self.D12_shape = (self.dim_v, self.dim_in)

        # define training nn params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        if self.linear_output:
            # set D21 to zero
            self.training_param_names.remove('D21') # not trainable anymore
            self.D21 = torch.zeros(*self.D21_shape, device=self.device) * self.weight_init_std
            # set D22 to zero
            self.D22 = torch.zeros(*self.D22_shape, device=self.device) * self.weight_init_std
            self.training_param_names.remove('D22') # not trainable anymore

        for name in self.training_param_names:
            # read the defined shapes
            shape = getattr(self, name + '_shape')
            # define each param as nn.Parameter
            setattr(self, name, nn.Parameter((torch.randn(*shape, device=self.device) * self.weight_init_std)))

        # auxiliary elements
        self.F = torch.zeros(dim_x, dim_x, device=self.device)
        self.B1 = torch.zeros(dim_x, dim_v, device=self.device)
        self.E = torch.zeros(dim_x, dim_x, device=self.device)
        self.Lambda = torch.ones(dim_v, device=self.device)
        self.C1 = torch.zeros(dim_v, dim_x, device=self.device)
        self.D11 = torch.zeros(dim_v, dim_v, device=self.device)

        # update non-trainable model params
        self.update_model_param()

    def update_model_param(self):
        """ Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * self.dim_x + self.dim_v, device=self.device)
        h1, h2, h3 = torch.split(H, [self.dim_x, self.dim_v, self.dim_x], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_x, self.dim_v, self.dim_x], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_x, self.dim_v, self.dim_x], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_x, self.dim_v, self.dim_x], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)
        self.eye = torch.eye(self.dim_v, device=self.device)

        # v signal
        # NOTE: change the following lines when you don't want a strictly acyclic REN!
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in):
        """ Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """

        w = torch.zeros(self.batch_size, 1, self.dim_v, device=self.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_v):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])
            w = w + (self.eye[i, :] * self.act(v / self.Lambda[i])).unsqueeze(1)

        # compute next state using Eq. 18
        self.x = F.linear(F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2),
                          self.E.inverse())

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)

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
        self.horizon = horizon

        outs = [y_init]
        for _ in range(horizon - 1):
            out = self.forward(u_in)
            outs.append(out)

        stacked_outs = torch.cat(outs, dim=1)
        return stacked_outs
