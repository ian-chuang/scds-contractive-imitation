# -*- coding: utf-8 -*-
"""DS_Policy_NeuralODE_IDEA

Author: Amin Abyaneh
Email: aminabyaneh@gmail.com
"""

import matplotlib.pyplot as plt

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *

import torch

import pyLasaDataset as lasa
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sine_data = lasa.DataSet.Sine
data = sine_data.demos

pos = np.array(data[0].pos.T)
vel = np.array(data[0].vel.T)

y_train = torch.from_numpy(vel.astype(np.float32))
x_train = torch.from_numpy(pos.astype(np.float32))

x_train.requires_grad = True
y_train.requires_grad = True

# generate a dataset for normal nn
dataset_nn = TensorDataset(x_train, y_train)
dataset_loader_nn = DataLoader(dataset_nn, batch_size=64, shuffle=True)

# dataset of initial conditions
base_value = x_train[0]

# generate random noise
noise = torch.randn(x_train.shape) / 10
noise.requires_grad = True

# create the vector with the first element of x_train plus noise
noisy_x_ic = noise + base_value

# generate a dataloader for ode
dataset_ode = TensorDataset(noisy_x_ic, y_train)
dataset_loader_ode = DataLoader(dataset_ode, batch_size=1, shuffle=True)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.scatter(noisy_x_ic[:, 0].detach(), noisy_x_ic[:, 1].detach(), s=2, c='blue')
ax.scatter(x_train[:, 0].detach(), x_train[:, 1].detach(), s=2, c='green')
plt.savefig('data.png')


import torch.nn as nn
import pytorch_lightning as pl

losses_ode = []

class LearnerODE(pl.LightningModule):
    def __init__(self, t_span: torch.Tensor, model: nn.Module,
                 expert_traj: torch.Tensor):
        super().__init__()
        self.model, self.t_span = model, t_span
        self.criterion = nn.MSELoss()
        self.expert_traj = expert_traj.reshape((expert_traj.shape[0], 1, expert_traj.shape[1]))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)

        # NOTE: here comes the connection to DS policy, we now try with just MSE, but
        # KL divergence and distribution learning is the way to go!!
        loss = self.criterion(y_hat, self.expert_traj)
        losses_ode.append(loss.item())

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return dataset_loader_ode

losses = []

class LearnerNN(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)
        losses.append(loss.item())

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return dataset_loader_nn

exp_name = f'ode_1000_Worm'

num_samples = x_train.size(0)
num_subsamples = int(num_samples * 0.01)
indices = torch.randperm(num_samples)[:num_subsamples]
train_positions = pos.astype(np.float32)

x_train_ds = torch.from_numpy(train_positions[indices])


""" Train the neural ODE """
t_span = torch.linspace(0, 1, 1000)

f = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )


node = NeuralODE(f, sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4)
learn = LearnerODE(t_span, node, x_train)

trainer = pl.Trainer(min_epochs=10, max_epochs=20, default_root_dir=exp_name, accelerator='cpu')
trainer.fit(learn)

# NOTE: To save and load
# learn = LearnerODE.load_from_checkpoint("ode_1000_Sine/lightning_logs/version_0/checkpoints/epoch=12-step=13000.ckpt",
#                                         t_span=t_span, model=node, expert_traj=x_train)

t_eval, y_hat = node(base_value, t_span)
fig = plt.figure(figsize=(8, 8))
plt.scatter(y_hat[:, 0].detach(), y_hat[:, 1].detach(), s=3, c='red', label='NODE')
plt.scatter(x_train[:, 0].detach(), x_train[:, 1].detach(), s=3, c='blue', label='Expert')
plt.legend(loc='upper right')
plt.savefig(f'{exp_name}.png')
