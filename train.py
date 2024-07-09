#!/usr/bin/env python
import os
import torch

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from ren_discrete import REN
from ren_train import train_ren_model
from cli import argument_parser
from dataset import lasa_expert
from plot import plot_trajectories, plot_policy


# main entry
if __name__ == '__main__':
    # TODO: neural ode layer instead of consecutive rollouts
    # TODO: move the plots
    # TODO: this function should work for both discrete and continuous ren modules

    # parse and set experiment arguments
    args = argument_parser()

    # experiment and REN configs
    patience_epoch = (args.total_epochs // 5) if args.patience_epoch is None else args.patience_epoch
    log_epoch = (args.total_epochs // 10) if args.log_epoch is None else args.log_epoch

    # set expert traj (equal to ren horizon for now, TODO: Relax later with KL or other measures)
    if args.expert == "lasa":
        expert_trajectory, dataloader = lasa_expert(args.motion_shape, args.horizon, args.device,
                                                    n_dems=1)

        y_init = torch.Tensor(expert_trajectory[:, 0, :]).unsqueeze(1)
        y_init = y_init.to(args.device)

        # input is set to zero
        u_in = torch.zeros((args.batch_size, 1, 2), device=args.device)

    # define REN
    ren_module = REN(dim_in=args.dim_in, dim_out=args.dim_out, dim_x=args.dim_x, dim_v=args.dim_v,
                     initialization_std=0.1, linear_output=True, add_bias=True, contraction_rate_lb=1.0,
                     batch_size=args.batch_size, device=args.device)
    ren_module.to(device=args.device)

    if args.load_model is None:
        # experiment log setup
        timestamp = datetime.now().strftime('%d-%H%M')
        experiment_name = f'{args.expert}-{args.motion_shape}-h{args.horizon}-x{args.dim_x}-' \
                          f'l{args.dim_v}-e{args.total_epochs}-b{args.batch_size}-t{timestamp}'

        writer_dir = f'{args.experiment_dir}/ren-training-{experiment_name}'
        writer = SummaryWriter(writer_dir)

        # training loop
        ren_trained = train_ren_model(ren_module, args.lr, u_in, y_init, args.horizon, expert_trajectory,
                                      args.total_epochs, args.lr_start_factor, args.lr_end_factor,
                                      patience_epoch, log_epoch, args.ic_noise_rate, writer, args.device,
                                      args.batch_size)
    else:
        # make sure the model exists
        assert os.path.isfile(args.load_model), f"File not found: {args.load_model}"

        # load the state dictionary
        experiment_data = torch.load(args.load_model)
        ren_module.load_state_dict(experiment_data['model_state_dict'])
        ren_module.update_model_param()

        # writer dir
        writer_dir = os.path.dirname(args.load_model)

    # plot the training trajectories
    expert_trajectory = expert_trajectory.cpu().numpy()

    # generate rollouts # TODO: add some parameters for num_rollouts and y_init_std
    policy_rollouts = []
    rollouts_horizon = args.horizon
    num_rollouts = 100
    y_init_std = 0.3

    for _ in range(num_rollouts):
        y_init_rollout = y_init + y_init_std * (2 * (torch.rand(*y_init.shape, device=args.device) - 0.5))
        policy_rollouts.append(ren_module.forward_trajectory(u_in, y_init_rollout, rollouts_horizon).detach().cpu().numpy())
        policy_rollouts.append(ren_module.forward_trajectory(u_in, y_init, rollouts_horizon).detach().cpu().numpy())

    plot_trajectories(rollouts=policy_rollouts, reference=expert_trajectory, save_dir=writer_dir, plot_name="rollouts")
    # plot_policy(ren_module, policy_rollouts, expert_trajectory, ".", "TEST")
