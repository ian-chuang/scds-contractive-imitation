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
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

# get the device
device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
start_point = 10
num_data_points = 10

# create a 2D reference trajectory
ref = torch.from_numpy(np.array([[i, i] for i in np.linspace(start_point, 0, num_data_points)], dtype=np.float32))
ref = ref.unsqueeze(1)
ref.to(device)

# input is set to zero
u_in = torch.zeros((1, 1, 2), device=device)
x_init = start_point * torch.ones((1, 1, 2), device=device)

# define REN
ren_module = REN(dim_in=2, dim_out=2, dim_x=2, l=8, initialization_std=0.1, linear_output=True,
                 contraction_rate_lb=1.0)
ren_module.to(device)


# optimizer
optimizer = torch.optim.Adam(ren_module.parameters(), lr=0.001)

# loss
criterion = nn.MSELoss()

# temps
trajectories = []
total_epochs = 2000
log_epoch = 50

# training epochs
for epoch in range(total_epochs):

    optimizer.zero_grad()
    out = ren_module.forward_trajectory(u_in, x_init, num_data_points)

    loss = criterion(out, ref)
    loss.backward()

    optimizer.step()
    ren_module.update_model_param()

    if epoch % log_epoch == 0:
        print(f'Epoch: {epoch}/{total_epochs} | Loss: {loss}')
        trajectories.append(out.detach().numpy())

    writer.add_scalars('Training Loss',
                    { 'Training' : loss.item()},
                    epoch + 1)
    writer.flush()

fig = plt.figure(figsize=(10, 10), dpi=120)
for idx, tr in enumerate(trajectories):
    plt.plot(tr[:, 0, 1], linewidth=idx * 0.05, c='blue')
plt.plot(ref[:, 0, 0], linewidth=1.0, c='green')
plt.savefig('ren_out.png')

