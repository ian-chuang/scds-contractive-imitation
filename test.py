#!/usr/bin/env python
import os
import torch

from ren_discrete import DREN
from ren_continuous import CREN

from cli import argument_parser
from dataset import lasa_expert
from plot import plot_trajectories, plot_policy


# main entry
if __name__ == '__main__':

    # parse and set experiment arguments
    args = argument_parser()

    # make sure the model exists
    assert os.path.isfile(args.load_model), f"File not found: {args.load_model}"

    # load the state dictionary
    experiment_data = torch.load(args.load_model)
    model_type = DREN if experiment_data['model_name'] == "DREN" else CREN

    # build the ren module
    ren_module = model_type(**experiment_data['model_params'])
    ren_module.load_state_dict(experiment_data['model_state_dict'])
    ren_module.update_model_param()
    print(f'Model loaded with \n params: {experiment_data["model_params"]} \n time: {experiment_data["training_time"]}')

    # writer dir
    writer_dir = os.path.dirname(args.load_model)

    # plot the training trajectories
    if args.expert == "lasa":
        expert_trajectory, dataloader = lasa_expert(args.motion_shape, experiment_data['model_params']['horizon'], args.device, n_dems=1) # TODO: number of dems here needs to be fixed wrt batch size

        # y_init is the first state in trajectory
        y_init = torch.Tensor(expert_trajectory[:, 0, :]).unsqueeze(1)

        # numpy for expert trajectory
        expert_trajectory = expert_trajectory.cpu().numpy()

        # input is set to zero
        u_in = torch.zeros((experiment_data['model_params']['batch_size'], 1, experiment_data['model_params']['dim_in']), device=args.device)

    # generate rollouts # TODO: add some parameters for num_rollouts and y_init_std
    policy_rollouts = []
    rollouts_horizon = experiment_data['model_params']['horizon']
    rollouts_horizon = args.horizon if rollouts_horizon is None else rollouts_horizon
    num_rollouts = 100
    y_init_std = 0.05

    with torch.no_grad():
        for _ in range(num_rollouts):
            y_init_rollout = y_init + y_init_std * (2 * (torch.rand(*y_init.shape, device=args.device) - 0.5))
            policy_rollouts.append(ren_module.forward_trajectory(u_in, y_init_rollout, rollouts_horizon).cpu().numpy())
            policy_rollouts.append(ren_module.forward_trajectory(u_in, y_init, rollouts_horizon).cpu().numpy())

    plot_trajectories(rollouts=policy_rollouts, reference=expert_trajectory, save_dir=writer_dir, plot_name="rollouts_stdtest_0.5")
