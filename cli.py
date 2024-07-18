import argparse
import torch


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Training ren for learning contractive motion through imitation.")

    # model args
    parser.add_argument('--model-type', type=str, default=None, help='Choose a model between "continuous" and "discrete" for the underlying REN.')
    parser.add_argument('--device', type=str, default="cpu" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')
    parser.add_argument('--horizon', type=int, default=50, help='Horizon value for the computation. Default is 50.')
    parser.add_argument('--dim-x', type=int, default=8, help='Dimension x. Default is 8.')
    parser.add_argument('--dim-in', type=int, default=2, help='Dimension u, or exogenous input. Default is 2.')
    parser.add_argument('--dim-out', type=int, default=2, help='Dimension y, or output. Default is 2.')
    parser.add_argument('--dim-v', type=int, default=2, help='Implicit equation size. Default is 8.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of forward trajectories of the network and expert trajectories at each step. Default is 1.')

    # training args
    parser.add_argument('--total-epochs', type=int, default=1000, help='Total number of epochs for training. Default is 200.')
    parser.add_argument('--log-epoch', type=int, default=None, help='Frequency of logging in epochs. Default is None which sets it to 0.1 * total_epochs.')
    parser.add_argument('--patience-epoch', type=int, default=None, help='Patience epochs for no progress. Default is None which sets it to 0.2 * total_epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate during training. Default is 0.01.')
    parser.add_argument('--lr-start-factor', type=float, default=1.0, help='Start factor of the linear learning rate scheduler. Default is 1.0.')
    parser.add_argument('--lr-end-factor', type=float, default=0.01, help='End factor of the linear learning rate scheduler. Default is 0.01.')
    parser.add_argument('--ic-noise-rate', type=float, default=0.0, help='Applied noise to the initial condition for further robustness.')

    # test args
    parser.add_argument('--num-test-rollouts', type=int, default=50, help='Number of test rollouts for plots.')
    parser.add_argument('--ic-test-std', type=float, default=0.3, help='Initial condition std during test and plotting phase.')

    # dataset args
    parser.add_argument('--expert', type=str, default="lasa", help='Expert type. Default is "lasa".')
    parser.add_argument('--motion-shape', type=str, default=None, help='Motion shape in lasa dataset. Choose from [Angle, CShape, GShape, Sine, Snake, Worm, etc].')

    # save/load args
    parser.add_argument('--experiment-dir', type=str, default='boards', help='Name tag for the experiments. By default it will be the "boards" folder.')
    parser.add_argument('--load-model', type=str, default=None, help='If it is not set to None, a pretrained model will be loaded instead of training.')

    args = parser.parse_args()

    # assertions and warning
    if args.expert == "lasa":
        assert args.motion_shape is not None, "Motion shape must be passed if expert is set to lasa."

    if args.total_epochs < 10000 and args.load_model is None:
        print(f'Minimum of 10000 epochs are required for proper training')

    if args.dim_v > 2:
        print(f'Complexities higher than 2 for dimension of the implicit layer (dim_v) are typically not necessary and cause serious computational overhead')

    if args.horizon > 100 and args.load_model is None:
        print(f'Long horizons may be unnecessary and pose significant computation')

    return args
