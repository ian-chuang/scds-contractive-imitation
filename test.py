#!/usr/bin/env python
import os
import torch

from typing import List, Dict

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
    assert os.path.isfile(args.load_model) or os.path.isdir(args.load_model), f"File not found: {args.load_model}"

    experiment_data_list: List[Dict] = []
    if os.path.isdir(args.load_model):
        for sub_exp_path in os.listdir(args.load_model):
            print(f'Loading the model:  {sub_exp_path}')
            model_dir = os.path.join(args.load_model, sub_exp_path)
            experiment_data_list.append({'model': torch.load(os.path.join(model_dir, 'best_model.pth')),
                                         'dir': model_dir})

        print(f'Plotting results for {len(experiment_data_list)} experiments')

    else:
        # load the state dictionary
        experiment_data_list = [torch.load(args.load_model)]
        print(f'Plotting results for a single experiment')

    # plot operation for all experiments
    for experiment_data in experiment_data_list:

        # continuous or discrete model
        model_type = DREN if experiment_data['model']['model_name'] == "DREN" else CREN

        # build the ren module
        ren_module = model_type(**experiment_data['model']['model_params'])
        ren_module.load_state_dict(experiment_data['model']['model_state_dict'])
        ren_module.update_model_param()
        print(f'Model loaded with \n params: {experiment_data["model"]["model_params"]} \n time: {experiment_data["model"]["training_time"]}')

        # writer dir
        writer_dir = experiment_data['dir']

        # plot the training trajectories
        if args.expert == "lasa":
            expert_trajectory, dataloader = lasa_expert(args.motion_shape, experiment_data['model']['model_params']['horizon'], args.device, n_dems=1) # TODO: number of dems here needs to be fixed wrt batch size

            # y_init is the first state in trajectory
            y_init = torch.Tensor(expert_trajectory[:, 0, :]).unsqueeze(1)

            # numpy for expert trajectory
            expert_trajectory = expert_trajectory.cpu().numpy()

            # input is set to zero
            u_in = torch.zeros((experiment_data['model']['model_params']['batch_size'], 1, experiment_data['model']['model_params']['dim_in']), device=args.device)

        # generate rollouts # TODO: add some parameters for num_rollouts and y_init_std
        policy_rollouts = []
        rollouts_horizon = experiment_data['model']['model_params']['horizon']
        num_rollouts = args.num_test_rollouts
        y_init_std = args.ic_test_std

        with torch.no_grad():
            for _ in range(num_rollouts):
                y_init_rollout = y_init + y_init_std * (2 * (torch.rand((experiment_data['model']['model_params']['batch_size'], 1, experiment_data['model']['model_params']['dim_in']), device=args.device) - 1.0))
                y_init = y_init + torch.zeros((experiment_data['model']['model_params']['batch_size'], 1, experiment_data['model']['model_params']['dim_in']), device=args.device)

                policy_rollouts.append(ren_module.forward_trajectory(u_in, y_init_rollout, rollouts_horizon).cpu().numpy())
                policy_rollouts.append(ren_module.forward_trajectory(u_in, y_init, rollouts_horizon).cpu().numpy())

        plot_trajectories(rollouts=policy_rollouts, reference=expert_trajectory, save_dir=writer_dir, plot_name=f'ic-rollouts-std{y_init_std}')

        # TODO: patch grid plots
        # plot_policy(ren_module, policy_rollouts, expert_trajectory, save_dir=writer_dir, plot_name=f'global-rollouts-std{y_init_std}', grid_coordinates=np.array([]))