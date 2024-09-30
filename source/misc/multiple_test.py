#!/usr/bin/env python
import os
import torch

from typing import List, Dict

from source.model.ren_discrete import DREN
from source.model.ren_continuous import CREN

from source.misc.cli import argument_parser
from source.data.lasa import DatasetKeys, lasa_expert
from source.misc.plot import plot_isolated_trajectories
from source.misc.plot import smooth_trajectory

# main entry
if __name__ == '__main__':

    # parse and set experiment arguments
    args = argument_parser()

    # make sure the model exists
    main_directory = args.load_model
    assert os.path.isdir(main_directory), f"Valid directory not found: {main_directory}"

    experiment_data_list: List[Dict] = []
    for folder in os.listdir(main_directory):
        print(f'Loading the model:  {folder}')
        sub_directory = os.path.join(main_directory, folder)
        experiment_data_list.append({'model': torch.load(os.path.join(sub_directory, 'best_model.pth')),
                                        'dir': sub_directory,
                                        'name': folder})

        print(f'Plotting results for {len(experiment_data_list)} experiments')


    # plot operation for all experiments
    results: List = []

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
        rollouts_horizon = experiment_data['model']['model_params']['horizon']
        expert = experiment_data['model']['expert']
        num_expert_trajectories = experiment_data['model']['num_expert_trajectories']
        motion_type = experiment_data['model']['motion_shape']

        dataloader = lasa_expert(motion_type, experiment_data['model']['model_params']['horizon'],
                                 args.device, num_exp_trajectories=num_expert_trajectories,
                                 num_aug_trajectories=0, batch_size=num_expert_trajectories)

        # test parameters
        policy_rollouts_o = []
        policy_rollouts_n = []
        num_rollouts = args.num_test_rollouts
        y_init_std = args.ic_test_std

        with torch.no_grad():

            # load the data as a single batch
            y_init, expert_trajectories = next(iter(dataloader))
            batch_size = y_init.size(0)

            for _ in range(num_rollouts):

                # set noisy initial condition for test
                y_init_noisy = y_init - y_init_std * (2 * (torch.rand(y_init.shape, device=args.device) - 0.5))

                # input is set to zero
                u_in = torch.zeros((batch_size, 1, experiment_data['model']['model_params']['dim_in']), device=args.device)

                # generate rollouts
                rollouts_noisy = ren_module.forward_trajectory(u_in, y_init_noisy, rollouts_horizon).cpu()
                rollouts_fixed = ren_module.forward_trajectory(u_in, y_init, rollouts_horizon).cpu()

                # add to plots
                policy_rollouts_o.append(rollouts_fixed)
                policy_rollouts_n.append(smooth_trajectory(rollouts_noisy))

        results.append(policy_rollouts_n)

    plot_isolated_trajectories(rollouts=results,
                               reference=expert_trajectories.numpy(),
                               save_dir=writer_dir,
                               plot_name=f'rollout-comp-{y_init_std}',
                               show_legends=args.legends)