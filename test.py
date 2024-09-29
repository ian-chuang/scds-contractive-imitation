#!/usr/bin/env python
import os
import torch

from typing import List, Dict, Union

from source.model.ren_discrete import DREN
from source.model.ren_continuous import CREN

from source.misc.cli import argument_parser
from source.data.lasa import lasa_expert
from source.data.robomimic import robomimic_expert, DatasetKeys

from source.misc.plot import plot_trajectories, plot_3d_trajectories, plot_start_template
from source.misc.plot import smooth_trajectory


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
        rollouts_horizon = experiment_data['model']['model_params']['horizon']

        # plot the training trajectories
        try: # TODO: temporary structure to for backward compatibility
            expert = experiment_data['model']['expert']
        except KeyError:
            expert = args.expert

        try: # TODO: temporary for backward compatibility
            num_expert_trajectories = experiment_data['model']['num_expert_trajectories']
        except KeyError:
            num_expert_trajectories = args.num_expert_trajectories

        if expert == "lasa":
            try: # TODO: temporary structure to for backward compatibility
                motion_type = experiment_data['model']['motion_shape']
            except KeyError:
                motion_type = experiment_data['name'].split('-')[2]

            dataloader = lasa_expert(motion_type, rollouts_horizon,
                                     args.device, num_exp_trajectories=num_expert_trajectories,
                                     num_aug_trajectories=0, batch_size=num_expert_trajectories)

        elif expert == "robomimic":
            motion_type = experiment_data['model']['motion_shape']
            dataloader = robomimic_expert(task=motion_type, device=args.device,
                                          batch_size=args.batch_size,
                                          dataset_keys=[DatasetKeys.EEF_POS.value],
                                          n_demos=num_expert_trajectories)

        # test parameters
        policy_rollouts_o = []
        policy_rollouts_n = []
        num_rollouts = args.num_test_rollouts
        y_init_std = args.ic_test_std

        with torch.no_grad():

            # load the initial conditions (if any)
            y_inits_saved: Union[torch.tensor, None] = None
            if os.path.exists(os.path.join(writer_dir, 'y_init.pt')):
                y_inits_saved = torch.load(os.path.join(writer_dir, 'y_init.pt'))

            # load the data as a single batch
            y_init, expert_trajectories = next(iter(dataloader))
            batch_size = y_init.size(0)

            # input is set to zero
            u_in = torch.zeros((batch_size, 1, experiment_data['model']['model_params']['dim_in']), device=args.device)

            if y_inits_saved is None or args.new_ic_test:
                # initialize a list to store y_init_noisy
                y_init_noisy_list = []

                # generate original rollouts
                rollouts_fixed = ren_module.forward_trajectory(u_in, y_init, rollouts_horizon).cpu()
                policy_rollouts_o.append(smooth_trajectory(rollouts_fixed))
                y_init_noisy_list.append(y_init)

                for _ in range(num_rollouts):

                    # set noisy initial condition for test
                    y_init_noisy = y_init + y_init_std * (2 * (torch.rand(y_init.shape, device=args.device) - 0.5))

                    # generate rollouts
                    rollouts_noisy = ren_module.forward_trajectory(u_in, y_init_noisy, rollouts_horizon).cpu()

                    # add to plots
                    policy_rollouts_n.append(smooth_trajectory(rollouts_noisy))
                    y_init_noisy_list.append(y_init_noisy)

                # save the array
                y_init_noisy_array = torch.cat(y_init_noisy_list, dim=0)
                torch.save(y_init_noisy_array, os.path.join(writer_dir, 'y_init.pt'))

            else:
                y_init_noisy_list_diff = []
                print(f'Initial {y_inits_saved.shape} conditions loaded from file')

                # separate true and noisy
                y_init = y_inits_saved[:batch_size]
                y_init_noisy = y_inits_saved[batch_size:]

                # generate rollouts
                u_in = torch.zeros((y_init_noisy.size(0), 1, experiment_data['model']['model_params']['dim_in']), device=args.device)
                rollouts_noisy = ren_module.forward_trajectory(u_in, y_init_noisy, rollouts_horizon).cpu()

                u_in = torch.zeros((batch_size, 1, experiment_data['model']['model_params']['dim_in']), device=args.device)
                rollouts_fixed = ren_module.forward_trajectory(u_in, y_init, rollouts_horizon).cpu()

                # add to plots
                policy_rollouts_o.append(smooth_trajectory(rollouts_fixed))
                policy_rollouts_n.append(smooth_trajectory(rollouts_noisy))

                y_init_noisy_list_diff.append(policy_rollouts_o[-1][:, 0].unsqueeze(1))
                y_init_noisy_list_diff.append(policy_rollouts_n[-1][:, 0].unsqueeze(1))

                # save the array
                y_init_noisy_array = torch.cat(y_init_noisy_list_diff, dim=0)
                torch.save(y_init_noisy_array, os.path.join(writer_dir, 'y_init_diff.pt'))

        if expert == "lasa":
            # plot_start_template(reference=expert_trajectories.numpy(),
            #                     save_dir=writer_dir,
            #                     plot_name=f'start')

            plot_trajectories(rollouts=[policy_rollouts_o, policy_rollouts_n],
                              reference=expert_trajectories.numpy(),
                              save_dir=writer_dir,
                              plot_name=f'ic-rollouts-std{y_init_std}',
                              show_legends=args.legends)

        elif expert == "robomimic":
            plot_3d_trajectories(rollouts=[policy_rollouts_o, policy_rollouts_n],
                                 reference=expert_trajectories.numpy(),
                                 save_dir=writer_dir,
                                 plot_name=f'ic-rollouts-std{y_init_std}')