import torch

import numpy as np
import matplotlib.pyplot as plt

from os import PathLike
from typing import List, Union


class PlotConfigs:
    """Hardcoded plot configurations.
    """

    COLORS = ["blue", "orange", "green", "purple", "brown"]
    FMTS = ['d--', 'o-', 's:', 'x-.', '*-', 'd--', 'o-']

    FIGURE_SIZE = (10, 10)
    FIGURE_DPI = 120
    POLICY_COLOR = 'grey'
    TRAJECTORY_COLOR = 'blue'
    ROLLOUT_COLOR = 'red'
    ANNOTATE_COLOR = 'black'
    TICKS_SIZE = 16
    LABEL_SIZE = 18
    LEGEND_SIZE = 18
    TITLE_SIZE = 18
    FILE_TYPE = "png"
    REFERENCE_SIZE = 3
    ROLLOUT_LINEWIDTH = 0.1


def plot_trajectories(rollouts: List[np.ndarray], reference: np.ndarray, save_dir: PathLike, plot_name: str,
                      space_stretch = 0.5):
    """ Plot the rollout and reference trajectories.

    Args:
        trajectories (List[np.ndarray]): Rollouts.
        references (List[np.ndarray]): Reference.
        save_dir (PathLike): Save directory.
        plot_name (str): Name of the plot file.
    """

    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(reference[0, :, :])

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])
    plt.grid()

    plt.scatter(reference[:, :, 0], reference[:, :, 1], s=PlotConfigs.REFERENCE_SIZE, marker='o', c=PlotConfigs.TRAJECTORY_COLOR,
                label='Expert Demonstrations')

    for tr in rollouts:
        plt.plot(tr[0, :, 0], tr[0, :, 1], linewidth=PlotConfigs.ROLLOUT_LINEWIDTH, c=PlotConfigs.ROLLOUT_COLOR)

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)
    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_policy(ren, rollouts: List[np.ndarray], reference: np.ndarray,
                save_dir: str, plot_name: str, space_stretch: float = 0.3,
                stream_density: float = 1.0, policy_density: int = 50,
                traj_density: int = 0.4, horizon: int = 1000):

    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(reference[0, :, :])

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])

    plt.scatter(reference[:, :, 0], reference[:, :, 1], color=PlotConfigs.TRAJECTORY_COLOR, marker='o',
                s=PlotConfigs.REFERENCE_SIZE, label='Expert Demonstrations')

    for tr in rollouts:
        plt.plot(tr[0, :, 0], tr[0, :, 1], linewidth=1, c=PlotConfigs.ROLLOUT_COLOR)

    # generate the grid data
    x = np.linspace(x_min - space_stretch, x_max + space_stretch, policy_density)
    y = np.linspace(y_min - space_stretch, y_max + space_stretch, policy_density)
    X, Y = np.meshgrid(x, y)

    step = 0.1 # np.linalg.norm(np.array([x_max - x_min + 2 * space_stretch, y_max - y_min + 2 * space_stretch]))
    # print(f'Step: {step}')
    # print(x_max, x_min, y_max, y_min, space_stretch)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    def fun(x):
        out = (ren.forward(torch.Tensor(x + step)) - ren.forward(torch.Tensor(x))).detach().cpu().numpy()
        out = out.reshape(1, 2)
        norm = np.linalg.norm(out, ord=2)
        return out / norm

    def fun_2(x):
        out = (ren.forward(torch.Tensor(x + step)) - ren.forward(torch.Tensor(x))).detach().cpu().numpy()
        print(out)
        out = out.reshape(1, 2)
        print(out)
        norm = np.linalg.norm(out, ord=2)
        return out / norm

    Z = np.apply_along_axis(fun, 1, data)
    print(Z.shape)
    U, V = Z[:, :, 1].reshape(policy_density, policy_density), \
           Z[:, :, 0].reshape(policy_density, policy_density)

    # plot the vector field
    plt.quiver(X[20, 20], Y[20, 20], U[20, 20], V[20, 20])
    print(X[20, 20], Y[20, 20])
    print(U[20, 20], V[20, 20])
    print(fun_2(np.array([X[20, 20], Y[20, 20]])))
    plt.quiver(X, Y, V, U) #density=stream_density, color=PlotConfigs.POLICY_COLOR, linewidth=1)

    # plot trajectory start and end
    # start_handle = plt.scatter(start_points[:, 0], start_points[:, 1], marker='x',
    #     color=PlotConfigs.ANNOTATE_COLOR, linewidth=3, s=120, label='Start')
    # target_handle = plt.scatter(goal_point[0], goal_point[1], marker='*',
    #     color=PlotConfigs.ANNOTATE_COLOR, linewidth=2, s=250, label='Target')

    green_arrows = plt.Line2D([0], [0], color=PlotConfigs.POLICY_COLOR,
                              linestyle='-', marker='>', label='Policy')
    red_arrows = plt.Line2D([0], [0], color=PlotConfigs.ROLLOUT_COLOR,
                            linestyle='-', marker='>', label='Reproduced')
    blue_dots = plt.Line2D([0], [0], color=PlotConfigs.TRAJECTORY_COLOR,
                           marker='o', label='Expert Demonstrations')

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    # add legend with the custom handle
    # plt.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='upper right',
    #     handles=[green_arrows, red_arrows, blue_dots, start_handle, target_handle])

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def find_limits(trajectory):
    """ Find the trajectory limits.

    Args:
        trajectory (np.ndarray): The given trajectory for finding limitations. Can be 2 or
            3 dimensions.

    Raises:
        NotSupportedError: Dimensions more than 3 are invalid.

    Returns:
        Tuple: A tuple of limits based on the dimensions (4 or 6 elements)
    """

    x_min = np.min(trajectory[:, 0])
    y_min = np.min(trajectory[:, 1])
    x_max = np.max(trajectory[:, 0])
    y_max = np.max(trajectory[:, 1])

    return x_min, x_max, y_min, y_max


# def plot_static_vector_field(model, trajectory, N=50, device='cpu', ax=None):
#     X, Y = np.mgrid[trajectory[..., 0].min():trajectory[..., 0].max():N*1j,
#                     trajectory[..., 1].min():trajectory[..., 1].max():N*1j]
#     X = X.T
#     Y = Y.T
#     P = np.vstack([X.ravel(), Y.ravel()]).T
#     P = torch.Tensor(P).to(device)

#     with torch.no_grad():
#         vector_field = model.odefunc(0.0, P).cpu()
#     vector_norm = vector_field.norm(dim=1).view(N, N).numpy()

#     vector_field = vector_field.view(N, N, 2).numpy()

#     if ax is None:
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111)
#     ax.contourf(X, Y, vector_norm, cmap='RdYlBu')
#     ax.streamplot(X, Y, vector_field[:, :, 0], vector_field[:, :, 1], color='k')

#     ax.set_xlim([X.min(), X.max()])
#     ax.set_ylim([Y.min(), Y.max()])
#     ax.set_xlabel(r"$x$")
#     ax.set_ylabel(r"$y$")
#     ax.set_title("Learned Vector Field")