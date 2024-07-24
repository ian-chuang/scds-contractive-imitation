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
                                         'dir': model_dir,
                                         'name': sub_exp_path})

        print(f'Plotting results for {len(experiment_data_list)} experiments')

    else:
        # load the state dictionary
        experiment_data_list = [{'model': torch.load(args.load_model),
                                 'dir': os.path.dirname(args.load_model),
                                 'name': os.path.basename(os.path.dirname(args.load_model))}]
        print(f'Plotting results for a single experiment')

    # plot operation for all experiments
    for experiment_data in experiment_data_list:

        # continuous or discrete model
        model_type = DREN if experiment_data['model']['model_name'] == "DREN" else CREN

        # build the ren module
        ren_module = model_type(**experiment_data['model']['model_params'], device=args.device)
        ren_module.load_state_dict(experiment_data['model']['model_state_dict'])
        ren_module.update_model_param()
        print(f'Model loaded with \n params: {experiment_data["model"]["model_params"]} \n time: {experiment_data["model"]["training_time"]} \n loss: {experiment_data["model"]["best_loss"]}')

        # writer dir
        writer_dir = experiment_data['dir']
        batch_size = experiment_data['model']['model_params']['batch_size']
        rollouts_horizon = experiment_data['model']['model_params']['horizon']

        # plot the training trajectories
        try: #TODO: temporary structure to for backward compatibility
            expert = experiment_data['model']['expert']
        except KeyError:
            expert = args.expert

        if expert == "lasa":
            try: # TODO: temporary structure to for backward compatibility
                motion_type = experiment_data['motion_shape']
            except KeyError:
                motion_type = experiment_data['name'].split('-')[2]

            expert_trajectory, dataloader = lasa_expert(motion_type, experiment_data['model']['model_params']['horizon'], args.device, n_dems=args.num_expert_trajectories)

            # y_init is the first state in trajectory
            y_init = torch.Tensor(expert_trajectory[:, 0, :]).unsqueeze(1)

            # numpy for expert trajectory
            expert_trajectory = expert_trajectory.cpu().numpy()

            # input is set to zero
            u_in = torch.zeros((batch_size, 1, experiment_data['model']['model_params']['dim_in']), device=args.device)

        # test parameters
        policy_rollouts = []
        num_rollouts = args.num_test_rollouts
        y_init_std = args.ic_test_std

        # stack y_init according to the batch size
        y_init_stacked = y_init.repeat(batch_size // y_init.shape[0], 1, 1)

        with torch.no_grad():
            for _ in range(num_rollouts):
                # set noisy initial condition for test
                y_init_noisy = y_init_stacked + y_init_std * (2 * (torch.rand((batch_size, 1, experiment_data['model']['model_params']['dim_out']), device=args.device) - 0.5))

                # generate rollouts
                rollouts_noisy = ren_module.forward_trajectory(u_in, y_init_noisy, rollouts_horizon).cpu()
                rollouts_fixed = ren_module.forward_trajectory(u_in, y_init_stacked, rollouts_horizon).cpu()

                # add to plots
                policy_rollouts.append(rollouts_fixed)
                policy_rollouts.append(rollouts_noisy)

        plot_trajectories(rollouts=policy_rollouts, reference=expert_trajectory, save_dir=writer_dir, plot_name=f'ic-rollouts-std{y_init_std}')

        # TODO: patch grid plots
        # plot_policy(ren_module, policy_rollouts, expert_trajectory, save_dir=writer_dir, plot_name=f'global-rollouts-std{y_init_std}', grid_coordinates=np.array([]))