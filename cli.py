import argparse
import torch


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description='Training ren for learning contractive motion through imitation.')

    # model args
    parser.add_argument('--model-type', type=str, default=None, help='Choose a model between "continuous" and "discrete" for the underlying REN.')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else "cuda:0", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')
    parser.add_argument('--horizon', type=int, default=50, help='Horizon value for the computation. Default is 50.')
    parser.add_argument('--dim-x', type=int, default=32, help='Dimension x. Default is 32.')
    parser.add_argument('--dim-in', type=int, default=2, help='Dimension u, or exogenous input. Default is 2.')
    parser.add_argument('--dim-out', type=int, default=2, help='Dimension y, or output. Default is 2.')
    parser.add_argument('--dim-v', type=int, default=2, help='Implicit equation size. Default is 8.')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of forward trajectories of the network and expert trajectories at each step. Default is 1.')
    parser.add_argument('--experiment-dir', type=str, default='boards', help='Name tag for the experiments. By default it will be the "boards" folder.')

    # bijection args
    parser.add_argument('--bijection', action='store_true', default=False, help='Use bijection net before projecting the output.')
    parser.add_argument('--num-bijection-layers', type=int, default=4, help='Number of hidden layers in the coupling layer design and blocks. Default is 2.')

    # training args
    parser.add_argument('--total-epochs', type=int, default=10000, help='Total number of epochs for training. Default is 200.')
    parser.add_argument('--log-epoch', type=int, default=None, help='Frequency of logging in epochs. Default is None which sets it to 0.1 * total_epochs.')
    parser.add_argument('--patience-epoch', type=int, default=None, help='Patience epochs for no progress. Default is None which sets it to 0.2 * total_epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate during training. Default is 0.01.')
    parser.add_argument('--lr-start-factor', type=float, default=1.0, help='Start factor of the linear learning rate scheduler. Default is 1.0.')
    parser.add_argument('--lr-end-factor', type=float, default=0.01, help='End factor of the linear learning rate scheduler. Default is 0.01.')
    parser.add_argument('--ic-noise-rate', type=float, default=0.0, help='Applied noise to the initial condition for further robustness.')
    parser.add_argument('--crate-lb', type=float, default=0.0, help='Lower bound for the contraction rate. Defaults to 0.0 for continuous and 1 for discrete.')
    parser.add_argument('--loss', type=str, default='mse', help='Training loss to be selected between "dtw" and "mse". Default is "mse".')

    # dataset args
    parser.add_argument('--expert', type=str, default='lasa', help='Expert type among ["lasa", "robomimic"]. Default is "lasa".')
    parser.add_argument('--motion-shape', type=str, default="CShape", help='Motion shape in LASA or Robomimic dataset. Choose from ["Angle", "CShape", "GShape", "Sine", "Snake", "Worm", etc] for LASA and ["lift", "can", "transport", "square"] for robomimic.')
    parser.add_argument('--dataset-key', type=str, default="eef_pos", help='Robomimic dataset keys in ["eef_pos", "eef_pos_ori", "joint_pos", "joint_pos_vel"]')
    parser.add_argument('--num-expert-samples', type=int, default=50, help='Number of samples per expert trajectories. Default is 50 for LASA dataset.')
    parser.add_argument('--num-expert-trajectories', type=int, default=None, help='Number of expert trajectories for training for either LASA or Robomimic data. Default is None.')
    parser.add_argument('--num-augment-trajectories', type=int, default=0, help='Number of augmented trajectories for training. Default is 0 for LASA dataset.')

    # test args
    parser.add_argument('--num-test-rollouts', type=int, default=20, help='Number of test rollouts for plots.')
    parser.add_argument('--ic-test-std', type=float, default=0.1, help='Initial condition std during test and plotting phase.')
    parser.add_argument('--load-model', type=str, default=None, help='If it is not set to None, a pretrained model will be loaded instead of training.')
    parser.add_argument('--legends', action='store_true', default=False, help='Add legend to the plots.')
    parser.add_argument('--new-ic-test', action='store_true', default=False, help='Load the saved initial conditions for consistency with other baselines.')

    args = parser.parse_args()

    # assertions and warning
    if args.total_epochs < 10000 and args.load_model is None:
        print(f'Minimum of 10000 epochs are required for proper training')

    if args.dim_v > 2:
        print(f'Complexities higher than 2 for dimension of the implicit layer (dim_v) are typically not necessary and cause serious computational overhead')

    if args.horizon > 100 and args.load_model is None:
        print(f'Long horizons may be unnecessary and pose significant computation')

    if args.crate_lb == 0.0 and args.model_type == 'discrete': # TODO: Fix this safety check
        print(f'Fixing invalid lower bound {args.crate_lb}  for contraction rate in {args.model_type} setting ')
        args.crate_lb = 1.0

    if args.loss == "mse" and args.load_model is None:
        assert args.horizon == args.num_expert_samples, 'Horizon and number of samples in each trajectory should be the same if loss is "mse"'
    return args
