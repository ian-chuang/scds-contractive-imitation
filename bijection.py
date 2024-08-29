""" An implementation of a coupling layer
from RealNVP (https://arxiv.org/abs/1605.08803).
"""

import torch
import torch.nn as nn


class BijectionNet(nn.Sequential):
    """ A sequential container of flows based on coupling layers.

    # NOTE: You use this one, which stacks the coupling layers.
    """
    def __init__(self, num_dims, num_blocks, num_hidden, device):
        self.num_dims = num_dims
        modules = []
        mask = torch.arange(0, num_dims) % 2  # alternating inputs
        mask = mask.float()
        mask = mask.to(device)

        for _ in range(num_blocks):
            modules += [
                CouplingLayer(
                    num_inputs=num_dims, num_hidden=num_hidden, mask=mask, device=device),
            ]
            mask = 1 - mask  # flipping mask
        super(BijectionNet, self).__init__(*modules)

    def forward(self, inputs):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        for module in self._modules.values():
            inputs = module(inputs)

        return inputs


class CouplingLayer(nn.Module):

    def __init__(self, num_inputs, num_hidden, mask, device):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden)
        self.scale_net.to(device)
        self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden)
        self.translate_net.to(device)

        nn.init.zeros_(self.translate_net.network[-1].weight.data)
        nn.init.zeros_(self.translate_net.network[-1].bias.data)

        nn.init.zeros_(self.scale_net.network[-1].weight.data)
        nn.init.zeros_(self.scale_net.network[-1].bias.data)

    def forward(self, inputs):
        mask = self.mask
        masked_inputs = inputs * mask

        log_s = self.scale_net(masked_inputs) * (1 - mask)
        t = self.translate_net(masked_inputs) * (1 - mask)

        s = torch.exp(log_s)
        return inputs * s + t


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

