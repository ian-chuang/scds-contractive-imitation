import argparse
import torch


# argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Training ren for learning contractive motion through imitation.")

    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to run the computations on, "cpu" or "cuda:0". Default is "cuda:0" if available, otherwise "cpu".')
    parser.add_argument('--horizon', type=int, default=10, help='Horizon value for the computation. Default is 10.')
    parser.add_argument('--dim-x', type=int, default=8, help='Dimension x. Default is 8.')
    parser.add_argument('--dim-in', type=int, default=2, help='Dimension u, or exogenous input. Default is 2.')
    parser.add_argument('--dim-out', type=int, default=2, help='Dimension y, or output. Default is 2.')
    parser.add_argument('--l-hidden', type=int, default=8, help='Hidden layer size. Default is 8.')
    parser.add_argument('--total-epochs', type=int, default=1000, help='Total number of epochs for training. Default is 200.')
    parser.add_argument('--log-epoch', type=int, default=None, help='Frequency of logging in epochs. Default is 50.')
    parser.add_argument('--expert', type=str, default="lasa", help='Expert type. Default is "lasa".')
    parser.add_argument('--motion-shape', type=str, default="Worm", help='Motion shape in lasa dataset. Choose from [Angle, CShape, GShape, Sine, Snake, Worm, etc].')

    args = parser.parse_args()
    return args