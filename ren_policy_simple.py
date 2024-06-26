#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from ren import REN

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/ren_trainer_{}'.format(timestamp))


def linear_ref(start_point, horizon, device):
    # create a 2D reference trajectory
    ref = torch.from_numpy(np.array([[i, i] for i in np.linspace(start_point, 0, horizon)], dtype=np.float32))
    ref = ref.unsqueeze(1)
    ref.to(device)
    return ref

def polynomial_ref(start_point, horizon, device, coefficients=[16, -16, 0.4, 0]):
    # Generate x values from start_point to 0
    x_values = np.linspace(start_point, 0, horizon, dtype=np.float32)

    # Create the polynomial from the coefficients
    poly = np.poly1d(coefficients)

    y_values = poly(x_values)

    ref = torch.from_numpy(np.stack((y_values, y_values), axis=1))
    ref = ref.unsqueeze(1)  # Add the extra dimension

    ref = ref.to(device)
    return ref

# experiment configs
device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
start_point = 1
horizon = 100
dim_x = 2

ref = linear_ref(start_point, horizon, device)

# input is set to zero
u_in = torch.zeros((1, 1, 2), device=device)
x_init = start_point * torch.ones((1, 1, dim_x), device=device)

# define REN
ren_module = REN(dim_in=2, dim_out=2, dim_x=dim_x, l=8, initialization_std=0.1, linear_output=True,
                 contraction_rate_lb=1.0)
ren_module.to(device=device)


# optimizer
optimizer = torch.optim.Adam(ren_module.parameters(), lr=0.001)

# loss
criterion = nn.MSELoss()

# temps
trajectories = []
total_epochs = 1000
log_epoch = 100

# training epochs
for epoch in range(total_epochs):

    optimizer.zero_grad()
    out = ren_module.forward_trajectory(u_in, x_init, horizon)

    loss = criterion(out, ref)
    loss.backward()

    optimizer.step()
    ren_module.update_model_param()

    if epoch % log_epoch == 0:
        print(f'Epoch: {epoch}/{total_epochs} | Loss: {loss}')
        trajectories.append(out.detach().cpu().numpy())

    writer.add_scalars('Training Loss',
                    { 'Training' : loss.item()},
                    epoch + 1)
    writer.flush()

# plot the training trajectories
fig = plt.figure(figsize=(10, 10), dpi=120)
for idx, tr in enumerate(trajectories):
    plt.plot(tr[:, 0, 1], linewidth=idx * 0.05, c='blue')
plt.plot(ref[:, 0, 0], linewidth=1, linestyle='dashed', c='green')
plt.savefig(f'runs/ren_trainer_{timestamp}.png')

# generate rollouts
rollouts = []
rollouts_horizon = 5 * horizon
num_rollouts = 10

for _ in range(num_rollouts):
    rollouts.append(ren_module.forward_trajectory(u_in, x_init, rollouts_horizon).detach().cpu().numpy())

fig = plt.figure(figsize=(10, 10), dpi=120)
for idx, tr in enumerate(rollouts):
    plt.plot(tr[:, 0, 1], linewidth=0.1, c='orange')
plt.plot(ref[:, 0, 0], linewidth=1, linestyle='dashed', c='green')
plt.savefig(f'runs/ren_rollouts_{timestamp}.png')
