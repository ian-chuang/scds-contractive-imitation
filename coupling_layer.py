""" An implementation of a coupling layer
from RealNVP (https://arxiv.org/abs/1605.08803).
"""

import torch

import torch.nn as nn
import matplotlib.pyplot as plt

class CouplingLayer(nn.Module):

    def __init__(self, num_inputs, num_hidden, mask, s_act='elu', t_act='elu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
        self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)

        nn.init.zeros_(self.translate_net.network[-1].weight.data)
        nn.init.zeros_(self.translate_net.network[-1].bias.data)

        nn.init.zeros_(self.scale_net.network[-1].weight.data)
        nn.init.zeros_(self.scale_net.network[-1].bias.data)

    def forward(self, inputs, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask

        log_s = self.scale_net(masked_inputs) * (1 - mask)
        t = self.translate_net(masked_inputs) * (1 - mask)

        if mode == 'direct':
            s = torch.exp(log_s)
            return inputs * s + t
        else:
            s = torch.exp(-log_s)
            return (inputs - t) * s

    def jacobian(self, inputs):
        return get_jacobian(self, inputs, inputs.size(-1))


class FCNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, act=nn.Tanh):
        super(FCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.network(x)


def get_jacobian(net, x, output_dims, reshape_flag=True):
    """

    """
    if x.ndimension() == 1:
        n = 1
    else:
        n = x.size()[0]
    x_m = x.repeat(1, output_dims).view(-1, output_dims)
    x_m.requires_grad_(True)
    y_m = net(x_m)
    mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
    # y.backward(mask)
    J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
    if reshape_flag:
        J = J.reshape(n, output_dims, output_dims)
    return J
