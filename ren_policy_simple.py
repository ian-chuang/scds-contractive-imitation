#!/usr/bin/env python
import copy
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from ren import REN

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Training ren for learning contractive motion through imitation.")

    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')
    parser.add_argument('--horizon', type=int, default=10, help='Horizon value for the computation. Default is 10.')
    parser.add_argument('--dim-x', type=int, default=8, help='Dimension x. Default is 8.')
    parser.add_argument('--dim-in', type=int, default=2, help='Dimension u, or exogenous input. Default is 2.')
    parser.add_argument('--dim-out', type=int, default=2, help='Dimension y, or output. Default is 2.')
    parser.add_argument('--l-hidden', type=int, default=8, help='Hidden layer size. Default is 8.')
    parser.add_argument('--total-epochs', type=int, default=1000, help='Total number of epochs for training. Default is 200.')
    parser.add_argument('--log-epoch', type=int, default=None, help='Frequency of logging in epochs. Default is 50.')
    parser.add_argument('--expert', type=str, default="poly", help='expert type. Default is "poly".')
    parser.add_argument('--motion-shape', type=str, default="Worm", help='Motion shape in lasa dataset. Choose from [Angle, CShape, GShape, Sine, Snake, Worm, etc].')

    args = parser.parse_args()
    return args

# synthetic data
def linear_ref(horizon, device, start_point=1.0):
    # create a 2D expert trajectory
    ref = torch.from_numpy(np.array([[i, i] for i in np.linspace(start_point, 0, horizon)], dtype=np.float32))
    ref = ref.unsqueeze(1)
    ref.to(device)
    return ref

def polynomial_ref(horizon, device, start_point=1.0, coefficients=[16, -16, 0.4, 0]):
    # Generate x values from start_point to 0
    x_values = np.linspace(start_point, 0, horizon, dtype=np.float32)

    # Create the polynomial from the coefficients
    poly = np.poly1d(coefficients)

    y_values = poly(x_values)

    ref = torch.from_numpy(np.stack((y_values, x_values), axis=1))
    ref = ref.unsqueeze(1)  # add the extra dimension

    ref = ref.to(device)
    return ref


def lasa_ref(motion_shape: str, horizon: int, device: str):
    import pyLasaDataset as lasa

    # load motion data
    motion_data = getattr(lasa.DataSet, motion_shape).demos
    pos = np.array(motion_data[0].pos.T)
    vel = np.array(motion_data[0].vel.T)


    def normalize(arr: np.ndarray):
        """ Normalization of data in the form of array. Each row is first
        summed and elements are then divided by the sum.

        Args:
            arr (np.ndarray): The input array to be normalized in the shape of (n_dim, n_samples).

        Returns:
            np.ndarray: The normalized array.
        """

        assert arr.shape[0] < arr.shape[1]
        max_magnitude = np.max(np.linalg.norm(arr, axis=0))
        return arr / max_magnitude

    # normalize
    pos = normalize(pos.T).T

    # load motion data into tensors
    y_train = torch.from_numpy(vel.astype(np.float32))
    x_train = torch.from_numpy(pos.astype(np.float32))
    x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1])

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # down sample the original dataset
    downscale_rate = horizon / x_train.shape[0]

    # samples and sub-samples
    num_samples = x_train.size(0)
    num_subsample = int(num_samples * downscale_rate)

    # Create a range of indices to select elements at regular intervals
    step = num_samples // num_subsample
    indices = torch.arange(0, num_samples, step)[:num_subsample]

    x_train_ds = torch.from_numpy(pos.astype(np.float32)[indices])
    x_train_ds = x_train_ds.view(x_train_ds.shape[0], 1, x_train_ds.shape[1])

    return x_train_ds


# main entry
if __name__ == '__main__':
    # TODO: torch lightening
    # TODO: fix the batch index
    # TODO: neural ode layer instead of consecutive rollouts
    # TODO: fix u_in size

    # parse and set experiment arguments
    args = argument_parser()

    # experiment and REN configs
    device = args.device
    ren_horizon = args.horizon
    ren_dim_x = args.dim_x
    ren_dim_in = args.dim_in
    ren_dim_out = args.dim_out
    ren_l = args.l_hidden
    total_epochs = args.total_epochs
    log_epoch = (args.total_epochs // 10) if args.log_epoch is None else args.log_epoch
    expert = args.expert
    lasa_motion_shape = args.motion_shape

    # set expert traj (equal to ren horizon for now)
    if expert == "poly":
        expert_trajectory = polynomial_ref(ren_horizon, device)
        x_init = 1.0 * torch.ones((1, 1, ren_dim_x), device=device)

    elif expert == "lin":
        expert_trajectory = linear_ref(ren_horizon, device)
        x_init = 1.0 * torch.ones((1, 1, ren_dim_x), device=device)

    elif expert == "lasa":
        expert_trajectory = lasa_ref(lasa_motion_shape, ren_horizon, device)
        x_init = torch.ones((1, 1, ren_dim_x), device=device)

    # input is set to zero
    u_in = torch.zeros((1, 1, 2), device=device)

    # define REN
    ren_module = REN(dim_in=ren_dim_in, dim_out=ren_dim_out, dim_x=ren_dim_x, l=ren_l, initialization_std=0.1, linear_output=True,
                    contraction_rate_lb=1.0)
    ren_module.to(device=device)

    # optimizer
    optimizer = torch.optim.Adam(ren_module.parameters(), lr=0.01)

    # loss
    criterion = nn.MSELoss()

    # temps
    trajectories = []
    best_model_stat_dict = None
    best_loss = torch.tensor(float('inf'))
    best_train_epoch = 0

    # experiment log setup
    timestamp = datetime.now().strftime('%d_%H%M')
    experiment_name = f'{expert}-{ren_horizon}-{ren_dim_x}-{ren_l}-{total_epochs}-{timestamp}'
    writer_dir = f'boards/ren-training-{experiment_name}'
    writer = SummaryWriter(writer_dir)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                            end_factor=0.01, total_iters=total_epochs)

    # training epochs
    for epoch in range(total_epochs):
        # zero grad
        optimizer.zero_grad()
        out = ren_module.forward_trajectory(u_in, x_init, ren_horizon)

        # loss
        loss = criterion(out, expert_trajectory)

        # best model
        if loss < best_loss:
            best_model_stat_dict = copy.deepcopy(ren_module.state_dict())
            best_loss = loss
            best_train_epoch = epoch

        # check no progress
        if epoch - best_train_epoch > 5000:
            print(f'No significant progress in a while, aborting training')
            break

        # backward and steps
        loss.backward()
        optimizer.step()
        scheduler.step()
        ren_module.update_model_param()

        # logs
        if epoch % log_epoch == 0:
            print(f'Epoch: {epoch}/{total_epochs} | Best Loss: {best_loss:.8f} | Best Epoch: {best_train_epoch} | LR: {scheduler.get_last_lr()[0]:.6f}')
            trajectories.append(out.detach().cpu().numpy())

        # tensorboard
        writer.add_scalars('Training Loss', {'Training' : loss.item()}, epoch + 1)
        writer.flush()

    # save the best model
    best_state = {
        'model_state_dict': best_model_stat_dict,
    }
    file = f'{writer_dir}/best_model.pth'
    torch.save(best_state, file)

    # load the best model for plotting
    ren_module.load_state_dict(best_model_stat_dict)
    ren_module.update_model_param()

    # TODO: move plots to plot tools
    # plot the training trajectories
    fig = plt.figure(figsize=(10, 10), dpi=120)
    for idx, tr in enumerate(trajectories):
        plt.plot(tr[:, 0, 0], tr[:, 0, 1], linewidth=idx * 0.05, c='blue')
    plt.plot(expert_trajectory[:, 0, 0], expert_trajectory[:, 0, 1], linewidth=1, linestyle='dashed', c='green')
    plt.xlabel('dim0')
    plt.ylabel('dim1')
    plt.savefig(f'{writer_dir}/ren-training-motion-{experiment_name}.png')

    # plot the training trajectories
    fig = plt.figure(figsize=(10, 10), dpi=120)
    for idx, tr in enumerate(trajectories):
        plt.plot(tr[:, 0, 0], linewidth=idx * 0.05, c='blue')
    plt.plot(expert_trajectory[:, 0, 0], linewidth=1, linestyle='dashed', c='green')
    plt.xlabel('time')
    plt.ylabel('dim0')
    plt.savefig(f'{writer_dir}/ren-training-time-{experiment_name}.png')

    # generate rollouts std
    rollouts = []
    rollouts_horizon = 10 * ren_horizon
    num_rollouts = 10
    x_init_std = 0.1

    for _ in range(num_rollouts):
        x_init_rollout = x_init + x_init_std * (2 * torch.rand(*x_init.shape, device=device) - 1)
        rollouts.append(ren_module.forward_trajectory(u_in, x_init_rollout, rollouts_horizon).detach().cpu().numpy())

    fig = plt.figure(figsize=(10, 10), dpi=120)
    for idx, tr in enumerate(rollouts):
        plt.plot(tr[:, 0, 0], linewidth = 0.5, c='orange')
    plt.plot(expert_trajectory[:, 0, 0], linewidth=1, linestyle='dashed', c='green')
    plt.xlabel('time')
    plt.ylabel('dim0')
    plt.savefig(f'{writer_dir}/ren-rollouts-std-time-{experiment_name}.png')

    fig = plt.figure(figsize=(10, 10), dpi=120)
    for idx, tr in enumerate(rollouts):
        plt.plot(tr[:, 0, 0], tr[:, 0, 1], linewidth=0.5, c='blue')
    plt.plot(expert_trajectory[:, 0, 0], expert_trajectory[:, 0, 1], linewidth=1, linestyle='dashed', c='green')
    plt.xlabel('dim0')
    plt.ylabel('dim1')
    plt.savefig(f'{writer_dir}/ren-rollouts-std-motion-{experiment_name}.png')

    # generate rollouts
    rollouts = []
    rollouts_horizon = 10 * ren_horizon
    num_rollouts = 10
    x_init_rollout = x_init

    for _ in range(num_rollouts):
        rollouts.append(ren_module.forward_trajectory(u_in, x_init_rollout, rollouts_horizon).detach().cpu().numpy())

    fig = plt.figure(figsize=(10, 10), dpi=120)
    for idx, tr in enumerate(rollouts):
        plt.plot(tr[:, 0, 0], linewidth = 0.5, c='orange')
    plt.plot(expert_trajectory[:, 0, 0], linewidth=1, linestyle='dashed', c='green')
    plt.xlabel('time')
    plt.ylabel('dim0')
    plt.savefig(f'{writer_dir}/ren-rollouts-time-{experiment_name}.png')

    fig = plt.figure(figsize=(10, 10), dpi=120)
    for idx, tr in enumerate(rollouts):
        plt.plot(tr[:, 0, 0], tr[:, 0, 1], linewidth=0.5, c='blue')
    plt.plot(expert_trajectory[:, 0, 0], expert_trajectory[:, 0, 1], linewidth=1, linestyle='dashed', c='green')
    plt.xlabel('dim0')
    plt.ylabel('dim1')
    plt.savefig(f'{writer_dir}/ren-rollouts-motion-{experiment_name}.png')