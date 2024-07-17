"""
    Code is modified from the original version by Galimberti et. al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class REN(nn.Module, ABC):
    def __init__(self, dim_in: int, dim_out: int, dim_x: int, dim_v: int,
                 batch_size: int = 1, weight_init_std: float = 0.5, linear_output: bool = False,
                 posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0, add_bias: bool = False,
                 device: str = "cpu"):
        """ Initialize a recurrent equilibrium network. This can also be viewed as a single layer
        of a larger network.

        # TODO: Redundancy in horizon should be removed
        # TODO: Use kwargs to reduce argument redundancy

        This is an abstract class for CREN and DREN modules. Please refer to the named modules for
        specific implementations.

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
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_x = dim_x
        self.dim_out = dim_out
        self.dim_v = dim_v
        self.batch_size = batch_size
        self.horizon = None
        self.device = device

        # set functionalities
        self.linear_output = linear_output
        self.contraction_rate_lb = contraction_rate_lb
        self.add_bias = add_bias
        self.act = nn.Tanh()

        # std and tolerance
        self.weight_init_std = weight_init_std
        self.epsilon = posdef_tol

        # initialize internal state
        self.x = torch.zeros(self.batch_size, 1, self.dim_x, device=self.device)

    def get_init_params(self):
        return {
            "dim_in": self.dim_in,
            "dim_out": self.dim_out,
            "dim_x": self.dim_x,
            "dim_v": self.dim_v,
            "batch_size": self.batch_size,
            "weight_init_std": self.weight_init_std,
            "linear_output": self.linear_output,
            "contraction_rate_lb": self.contraction_rate_lb,
            "add_bias": self.add_bias,
            "horizon": self.horizon
        }

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

        x_init = torch.linalg.lstsq(self.C2,  y_init.squeeze(1).T)[0].T.unsqueeze(1)
        self.set_x_init(x_init)

    @abstractmethod
    def update_model_param(self):
        """ Update non-trainable matrices according to the REN formulation to preserve contraction.
        """

    @abstractmethod
    def forward_trajectory(self, u_in: torch.Tensor, y_init: torch.Tensor, horizon: int):
        """ Get a trajectory of forward passes.

        First element can be either y_init, as used here, or y_1. Depends on the application.

        Args:
            u_in (torch.Tensor): Input at each time step. Must be fixed for autonomous systems.
            y_init (torch.Tensor): Initial condition of the output.
            horizon (int, optional): Length of the forward trajectory. Defaults to 20.
        """
