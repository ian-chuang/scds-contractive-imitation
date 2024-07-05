import numpy as np
import matplotlib.pyplot as plt

from os import PathLike
from typing import List, Union


class PlotConfigs:
    """Hardcoded plot configurations.
    """

    COLORS = ["blue", "orange", "green", "purple", "brown"]
    FMTS = ['d--', 'o-', 's:', 'x-.', '*-', 'd--', 'o-']

    FIGURE_SIZE = (8, 8)
    FIGURE_DPI = 120
    POLICY_COLOR = 'grey'
    TRAJECTORY_COLOR = 'blue'
    ROLLOUT_COLOR = 'red'
    ANNOTATE_COLOR = 'black'
    TICKS_SIZE = 16
    LABEL_SIZE = 18
    LEGEND_SIZE = 18
    TITLE_SIZE = 18
    FILE_TYPE = "svg"
    REFERENCE_SIZE = 3

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
    x_min, x_max, y_min, y_max = find_limits(reference)

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])
    plt.grid()

    plt.scatter(reference[:, 0], reference[:, 1], s=PlotConfigs.REFERENCE_SIZE, marker='o', c=PlotConfigs.TRAJECTORY_COLOR,
                label='Expert Demonstrations')

    for tr in rollouts:
        plt.plot(tr[:, 0], tr[:, 1], linewidth=1, c=PlotConfigs.ROLLOUT_COLOR)

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)
    plt.savefig(f'save_dir/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_policy(ren, rollouts: List[np.ndarray], reference: np.ndarray,
                save_dir: str, plot_name: str, space_stretch: float = 0.1,
                stream_density: float = 1.0, policy_density: int = 100,
                traj_density: int = 0.4, horizon: int = 1000):

    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(reference)

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])

    plt.scatter(reference[:, 0], reference[:, 1], color=PlotConfigs.TRAJECTORY_COLOR, marker='o',
                s=PlotConfigs.REFERENCE_SIZE, label='Expert Demonstrations')

    # generate the grid data
    x = np.linspace(x_min - space_stretch, x_max + space_stretch, policy_density)
    y = np.linspace(y_min - space_stretch, y_max + space_stretch, policy_density)
    X, Y = np.meshgrid(x, y)

    step = np.linalg.norm(np.array[x_max - x_min + 2 * space_stretch, y_max - y_min + 2 * space_stretch])
    print(step, x_max, x_min, y_max, y_min, space_stretch)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    Z = np.apply_along_axis(lambda x: (ren.forward(x + step) - ren.forward(x)).detach().cpu().numpy(), 1, data)
    U, V = Z[:,:,0].reshape(policy_density, policy_density), \
        Z[:,:,1].reshape(policy_density, policy_density)

    # plot the vector field
    plt.streamplot(X, Y, U, V, density=stream_density, color=PlotConfigs.POLICY_COLOR, linewidth=1)

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

    plt.savefig(f'save_dir/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


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
