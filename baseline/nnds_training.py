#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split

from learn_nn_ds import NL_DS
from plot_trajectories import plot_trajectories
from data import load_pylasa_data

def train_neural_policy(network: str, mode: str, motion_shape: str, n_dems: int,
                        n_epochs: int, plot: bool, model_name: str, test_size: float,
                        save: bool, save_dir: str, gpu: bool, alpha: float, eps: float,
                        relaxed: bool):
    """ Training sequence for a stable/unstable neural policy to estimate a
    nonlinear dynamical system.

    Args:
        network(str): Type of the nonlinear estimator, could be nn, snds, sdsef.
        motion_shape (str): Shape of the trajectories.
        n_dems (int): Number of augmented demonstrations.
        plot (bool): Whether to plot trajectories and final ds or not.
        n_epochs (int): Total number of epochs.
        model_name (str): Name of the model for save and load.
        test_size (float): Size of the test dataset.
        save_dir (str): In case save is activated, files will be saved in this directory.

        # SNDS specific params
        alpha (float): Constant for exponential stability, useless if relaxed is set to True.
        eps (float): Constant for quadratic Lyapunov function added to the ICNN.
        relaxed (bool): Relax the exponential condition for SNDS to global asymptotic stability.
    """

    ''' Load the dataset '''
    model_name = model_name.lower()
    name = f'{model_name}-{network}-{motion_shape.lower()}-{datetime.now().strftime("%d-%m-%H-%M")}'

    states, state_der = load_pylasa_data(motion_shape=motion_shape, normalized=True)
    split = train_test_split(states, state_der, test_size=test_size, random_state=np.random.randint(10))
    states_train, states_test, state_der_train, state_der_test = split
    print(f'Shape of the train data is {states_train.shape} and test is {states_test.shape}.')

    ''' Train and save a model'''
    nl_ds = NL_DS(network=network, data_dim=states.shape[1], gpu=gpu, alpha=alpha, eps=eps, relaxed=relaxed)

    if mode == 'train':
        nl_ds.fit(states_train, state_der_train, n_epochs=n_epochs, trajectory_test=states_test, velocity_test=state_der_test)

    if mode == 'test':
        nl_ds.load(model_name, dir=save_dir)

    ''' Plot the DS '''
    if plot:
        plot_trajectories(nl_ds, states, save_dir=save_dir, file_name=name)

    ''' Save the DS '''
    if save:
        nl_ds.save(model_name=name, dir=save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')

    # general params
    parser.add_argument('-nt', '--neural-tool', type=str, default="nn",
                        help='The neural policy or tool among snds, nn, sdsef.')
    parser.add_argument('-m', '--mode', type=str, default="train",
                        help='Mode between train and test. Test mode only loads the model with the provided name.')
    parser.add_argument('-ms', '--motion-shape', type=str, default="Worm",
                        help='Shape of the trajectories as in LASA dataset.')
    parser.add_argument('-nd', '--num-demonstrations', type=int, default=7,
                        help='Number of additional demonstrations to the original dataset.')
    parser.add_argument('-ne', '--num-epochs', type=int, default=10000,
                        help='Number of training epochs.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=True,
                        help='Show extra plots of final result and trajectories.')

    parser.add_argument('-ts', '--test-size', type=float, default=0.01, help='Size of the validation set, not very important in this context.')

    parser.add_argument('-sm', '--save-model', action='store_true', default=False,
                        help='Save the model in the res folder.')
    parser.add_argument('-sd', '--save-dir', type=str, default=os.path.join(os.pardir, 'res', 'nlds_policy'),
                        help='Optional destination for save/load.')
    parser.add_argument('-mn', '--model-name', type=str, default='test', help='Optional model name for saving.')

    parser.add_argument('-gp', '--gpu', type=bool, default=True, help='Enable or disable GPU support.')

    # SNDS params
    parser.add_argument('-rl', '--relaxed', type=bool, default=True, help='Relax asymptotic stability for SNDS.')
    parser.add_argument('-al', '--alpha', type=float, default=0.01, help='Exponential stability constant for SNDS as explained in the paper.')
    parser.add_argument('-ep', '--eps', type=float, default=0.01, help='Quadratic Lyapunov addition constant for SNDS as explained in the paper.')

    args = parser.parse_args()

    train_neural_policy(args.neural_tool, args.mode, args.motion_shape,
                        args.num_demonstrations, args.num_epochs, args.show_plots,
                        test_size=args.test_size, model_name=args.model_name,
                        save=args.save_model, save_dir=args.save_dir, gpu=args.gpu,
                        relaxed=args.relaxed, eps=args.eps, alpha=args.alpha)
