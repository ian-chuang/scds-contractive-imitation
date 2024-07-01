import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"

class REN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_x: int,
                 l: int, initialization_std: float = 0.5, linear_output: bool = False,
                 posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0):
        """ Initialize a recurrent equilibrium network. This can also be viewed as a single layer
        of a larger network.

        # IMPORTANT TODO: Fix all the parameters and fixed tensors and their device!!!!!!

        NOTE: The equations for REN upon which this class is built can be found in the following paper
        "Revay M et al. Recurrent equilibrium networks: Flexible dynamic models with guaranteed
        stability and robustness."

        The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
        The model is described as,

                        [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                        [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                        [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

        where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
        are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

        Args:
            dim_in (int): Input dimension. The input is the u vector defined in the paper.
            dim_out (int): Output dimension.
            dim_xi (int): Internal state dimension. This state evolves with contraction properties.
            l (int): Complexity of the implicit layer.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            xi_init (torch.Tensor, optional): Initial condition for the internal state. Defaults to None.

            linear_output (bool, optional): If set True, the output matrices are arranged in a way so that
                the output is a linear transformation of the input. Defaults to False.

            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.

        """
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_x = dim_x
        self.dim_out = dim_out
        self.l = l

        # set functionalities
        self.linear_output = linear_output
        self.contraction_rate_lb = contraction_rate_lb

        # initialize internal state
        self.x = torch.zeros(1, 1, self.dim_x, device=device)

        # auxiliary matrices
        self.X_shape = (2 * self.dim_x + self.l, 2 * self.dim_x + self.l)
        self.Y_shape = (self.dim_x, self.dim_x)

        # nn state dynamics
        self.B2_shape = (self.dim_x, self.dim_in)

        # nn output
        self.C2_shape = (self.dim_out, self.dim_x)
        self.D21_shape = (self.dim_out, self.l)
        self.D22_shape = (self.dim_out, self.dim_in)

        # v signal
        self.D12_shape = (self.l, self.dim_in)

        # define training nn params TODO: Replace this with straightforward definition
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        if self.linear_output:
            # set D21 to zero
            self.training_param_names.remove('D21') # not trainable anymore
            self.D21 = torch.zeros(*self.D21_shape, device=device) * initialization_std
            # set D22 to zero
            self.D22 = torch.zeros(*self.D22_shape, device=device) * initialization_std
            self.training_param_names.remove('D22') # not trainable anymore

        for name in self.training_param_names:
            # read the defined shapes
            shape = getattr(self, name + '_shape')
            # define each param as nn.Parameter
            setattr(self, name, nn.Parameter((torch.randn(*shape, device=device) * initialization_std)))

        # auxiliary elements
        self.epsilon = posdef_tol
        self.F = torch.zeros(dim_x, dim_x, device=device)
        self.B1 = torch.zeros(dim_x, l, device=device)
        self.E = torch.zeros(dim_x, dim_x, device=device)
        self.Lambda = torch.ones(l, device=device)
        self.C1 = torch.zeros(l, dim_x, device=device)
        self.D11 = torch.zeros(l, l, device=device)

        # update non-trainable model params
        self.update_model_param()

    def set_x_init(self, x_init):
        """ Set the initial condition of x.

        Args:
            x_init (torch.Tensor): Initial value for x.
        """

        # fix the initial condition of the internal state
        self.x = x_init

    def set_y_init(self, y_init):
        """ Set x_init that results in a given y_init, when
        output is a linear transformation of the state (x),
        y_t = C_2 x_t.

        If dim_x > dim_out, infinitely many solutions might exist.
        in this case, the min norm solution is returned.

        Args:
            x_init (torch.Tensor): Initial value for x.
        """

        x_init = torch.linalg.lstsq(self.C2,  y_init.squeeze(1).T)[0].view(1, 1, self.dim_x)
        self.set_x_init(x_init)

    def update_model_param(self):
        """ Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * self.dim_x + self.l, device=device)
        h1, h2, h3 = torch.split(H, [self.dim_x, self.l, self.dim_x], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_x, self.l, self.dim_x], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_x, self.l, self.dim_x], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_x, self.l, self.dim_x], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)
        self.eye = torch.eye(self.l, device=device)

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
        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.l, device=device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.l):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])
            w = w + (self.eye[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.l)

        # compute next state using Eq. 18
        self.x = F.linear(
            F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2),
            self.E.inverse())

        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        # TODO: this is kind of a diffeomorphism? replace with a bijection layer of normalizing flow?
        return y_out

    def forward_trajectory(self, u_in: torch.Tensor, y_init: torch.Tensor, horizon: int = 20):
        """ Get a trajectory of forward passes.

        First element can be either y_init, as used here, or y_1. Depends on the application.

        Args:
            u_in (torch.Tensor): Input at each time step. Must be fixed for autonomous systems.
            y_init (torch.Tensor): Initial condition of the output.
            horizon (int, optional): Length of the forward trajectory. Defaults to 20.
        """

        self.set_y_init(y_init)

        outs = [y_init]
        for _ in range(horizon - 1):
            out = self.forward(u_in)
            outs.append(out)

        stacked_outs = torch.cat(outs, dim=0)
        return stacked_outs
