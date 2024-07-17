#!/usr/bin/env python
import os
import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ren_discrete import DREN
from ren_continuous import CREN

from ren_trainer import train_ren_model
from cli import argument_parser
from dataset import lasa_expert
from plot import plot_trajectories, plot_policy


# main entry
if __name__ == '__main__':
    # TODO: this function should work for both discrete and continuous ren modules
    # TODO: create a superclass for REN-IL that encompasses both CREN and DREN

    # parse and set experiment arguments
    args = argument_parser()

    # set expert traj (equal to ren horizon for now, TODO: Relax later with KL or other measures)
    if args.expert == "lasa":
        expert_trajectory, dataloader = lasa_expert(args.motion_shape, args.horizon, args.device, n_dems=1)

        y_init = torch.Tensor(expert_trajectory[:, 0, :]).unsqueeze(1)
        y_init = y_init.to(args.device)

        # input is set to zero
        u_in = torch.zeros((args.batch_size, 1, args.dim_in), device=args.device)

    else:
        raise(NotImplementedError(f'Expert data is not available!'))

    # define REN model
    if args.model_type == 'continuous':
        model = CREN(dim_in=args.dim_in, dim_out=args.dim_out, dim_x=args.dim_x, dim_v=args.dim_v,
                     batch_size=args.batch_size, device=args.device, horizon=args.horizon)

    elif args.model_type == 'discrete':
        model = DREN(dim_in=args.dim_in, dim_out=args.dim_out, dim_x=args.dim_x, dim_v=args.dim_v,
                     batch_size=args.batch_size, device=args.device, horizon=args.horizon)

    else:
        raise(NotImplementedError('Please determine a correct model type: ["continuous", "discrete"]!'))

    # send the model to device
    model.to(device=args.device)

    # experiment log setup
    timestamp = datetime.now().strftime('%d-%H%M')
    experiment_name = f'{type(model).__name__.lower()}-{args.expert}-{args.motion_shape}-h{args.horizon}' \
                      f'-x{args.dim_x}-l{args.dim_v}-e{args.total_epochs}-b{args.batch_size}-n{args.ic_noise_rate}-t{timestamp}'

    writer_dir = f'{args.experiment_dir}/{experiment_name}'
    writer = SummaryWriter(writer_dir)

    # training loop # TODO: make this more efficient using kwargs
    ren_trained = train_ren_model(model, args.lr, u_in, y_init, args.horizon, expert_trajectory,
                                  args.total_epochs, args.lr_start_factor, args.lr_end_factor,
                                  args.patience_epoch, args.log_epoch, args.ic_noise_rate, writer,
                                  args.device, args.batch_size)

    # plot the training trajectories
    expert_trajectory = expert_trajectory.cpu().numpy()

    # generate rollouts # TODO: add some parameters for num_rollouts and y_init_std
    policy_rollouts = []
    rollouts_horizon = args.horizon
    num_rollouts = args.num_test_rollouts
    y_init_std = args.ic_test_std

    for _ in range(num_rollouts):
        y_init_rollout = y_init + y_init_std * (2 * (torch.rand(*y_init.shape, device=args.device) - 0.5))
        policy_rollouts.append(model.forward_trajectory(u_in, y_init_rollout, rollouts_horizon).detach().cpu().numpy())
        policy_rollouts.append(model.forward_trajectory(u_in, y_init, rollouts_horizon).detach().cpu().numpy())

    plot_trajectories(rollouts=policy_rollouts, reference=expert_trajectory, save_dir=writer_dir, plot_name="rollouts")
