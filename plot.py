import torch
import numpy as np
import matplotlib.pyplot as plt

from os import PathLike
from typing import List
from tqdm import tqdm

from ren import REN


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

    x_min = np.min(trajectory[:, :, 0])
    y_min = np.min(trajectory[:, :, 1])
    x_max = np.max(trajectory[:, :, 0])
    y_max = np.max(trajectory[:, :, 1])

    return x_min, x_max, y_min, y_max


def plot_trajectories(rollouts: List[np.ndarray], reference: np.ndarray, save_dir: PathLike, plot_name: str,
                      space_stretch = 0.5, is_3d: bool = False):
    """ Plot the rollout and reference trajectories.

    # TODO: Use REN to generate data here instead
    # TODO: Animate maybe?

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

    plt.scatter(reference[:, :, 0], reference[:, :, 1], s=PlotConfigs.REFERENCE_SIZE, marker='o', c=PlotConfigs.TRAJECTORY_COLOR,
                label='Expert Demonstrations')

    for tr in rollouts:
        for batch_idx in range(tr.shape[0]):
            plt.plot(tr[batch_idx, :, 0], tr[batch_idx, :, 1], linewidth=PlotConfigs.ROLLOUT_LINEWIDTH, c=PlotConfigs.ROLLOUT_COLOR)

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)
    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_trajectories_time(ren: REN, reference: np.ndarray, horizon: int, save_dir: PathLike, plot_name: str,
                           space_stretch = 0.5, density: int = 10, is_3d: bool = False):
    """ Plot the rollout and reference trajectories.

    Args:
        ren (REN): The ren module to generate trajectories.
        trajectories (List[np.ndarray]): Rollouts.
        references (List[np.ndarray]): Reference.
        save_dir (PathLike): Save directory.
        plot_name (str): Name of the plot file.
    """

    # TODO: idea for 3d
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(trajectories.shape[1]):
    #     ax.plot(trajectories[:, i, 0], trajectories[:, i, 1], time_span,
    #             color=class_colors[i], alpha=0.1)

    # ax.set_title('3D Trajectories')
    # ax.set_xlabel(r"$\mathbf{z}_0(t)$")
    # ax.set_ylabel(r"$\mathbf{z}_1(t)$")
    # ax.set_zlabel(r"$t$")

    x_min, x_max, y_min, y_max = find_limits(reference[0, :, :])

    # calibrate the axis
    fig = plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    # generate the grid data
    x = np.linspace(x_min - space_stretch, x_max + space_stretch, density)
    y = np.linspace(y_min - space_stretch, y_max + space_stretch, density)
    X, Y = np.meshgrid(x, y)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    data = np.expand_dims(data, axis=0)

    trajectories = []
    u_in = torch.zeros(1, 1, 2, device="cpu") # TODO: REPLACE

    for d in tqdm(range(data.shape[1]), desc="Generating plot trajectories"):
        data_point = np.expand_dims(data[:, d, :], axis=0)
        traj = ren.forward_trajectory(u_in, torch.Tensor(data_point), horizon).detach().cpu().numpy()
        trajectories.append(traj)

    trajectories = np.concatenate(trajectories, 0)

    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    time_span = np.linspace(0.0, 1.0, horizon)

    for i in range(trajectories.shape[0]):
        ax0.plot(time_span, trajectories[i, :, 0], alpha=0.5)
        ax1.plot(time_span, trajectories[i, :, 1], alpha=0.5)

    ax0.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)
    ax1.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)
    plt.grid()

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')

def plot_policy(ren, rollouts: List[np.ndarray], reference: np.ndarray,
                save_dir: str, plot_name: str, space_stretch: float = 0.8,
                grid_coordinates: np.ndarray = np.array([[0.0, -0.4], [0.0, -0.1]]), grid_density: int = 3,
                horizon: int = 100):

    # TODO: change this to cover patches of state space instead of the entire state space
    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(reference)

    # calibrate the axis
    fig = plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])
    plt.grid()

    plt.scatter(reference[:, :, 0], reference[:, :, 1], color=PlotConfigs.TRAJECTORY_COLOR, marker='o',
                s=PlotConfigs.REFERENCE_SIZE, label='Expert Demonstrations')

    for i, tr in enumerate(rollouts): # TODO: Bugs here!
        plt.plot(tr[0, :, 0], tr[0, :, 1], linewidth=0.1, c=PlotConfigs.ROLLOUT_COLOR)
        # start_handle = plt.scatter(tr[0, 0, 0], tr[0, 0, 1], marker='x',
        #                            color=PlotConfigs.ANNOTATE_COLOR, linewidth=3, s=100,
        #                            label='Start')

    # generate the grid data
    x = np.linspace(grid_coordinates[0][0], grid_coordinates[0][1], grid_density)
    y = np.linspace(grid_coordinates[1][0], grid_coordinates[1][1], grid_density)
    X, Y = np.meshgrid(x, y)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    data = np.expand_dims(data, axis=0)

    trajectories = []
    horizon = ren.horizon
    u_in = torch.zeros(1, 1, 2, device="cpu")

    for d in tqdm(range(data.shape[1]), desc="Generating plot rollouts"):
        data_point = np.expand_dims(data[:, d, :], axis=0)
        traj = ren.forward_trajectory(u_in, torch.Tensor(data_point), horizon).detach().cpu().numpy()
        trajectories.append(traj)

    trajectories = np.concatenate(trajectories, 0)

    for i in range(trajectories.shape[0]):
    #     start_handle = plt.scatter(trajectories[i, 0, 0], trajectories[i, 0, 1], marker='x',
    #                                color=PlotConfigs.ANNOTATE_COLOR, linewidth=3, s=100,
    #                                label='Start')

        plt.plot(trajectories[i, :, 0], trajectories[i, :, 1],  linewidth=0.1, c=PlotConfigs.ROLLOUT_COLOR)
    #     plt.streamplot(trajectories[i, :, 0], trajectories[i, :, 1], np.diff(trajectories[i, :, 0]), np.diff(trajectories[i, :, 1]), color="black", linewidth=1)

    red_arrows = plt.Line2D([0], [0], color=PlotConfigs.ROLLOUT_COLOR,
                            linestyle='-', marker='>', label='Reproduced')
    blue_dots = plt.Line2D([0], [0], color=PlotConfigs.TRAJECTORY_COLOR,
                           marker='o', label='Expert Demonstrations')

    axes.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    # add legend with the custom handle
    # plt.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='upper right',
    #     handles=[green_arrows, red_arrows, blue_dots, start_handle, target_handle])

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')



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



# def plot_trajectories_animation(time_span, trajectories, colors, classes, lim=10.0):
#     def animate_frame(t):
#         ax.cla()
#         ax.set_xlim(-lim, lim)
#         ax.set_ylim(-lim, lim)
#         ax.set_title('Trajectories')
#         ax.set_xlabel(r"$\mathbf{z}_0(t)$")
#         ax.set_ylabel(r"$\mathbf{z}_1(t)$")

#         zero_classes = np.array(classes) == 0
#         one_classes = np.array(classes) == 1

#         scatter_zero = ax.plot(
#             trajectories[t, zero_classes, 0], trajectories[t, zero_classes, 1],
#             'o', color=colors[0], alpha=0.2+0.8*t/len(time_span))
#         scatter_one = ax.plot(
#             trajectories[t, one_classes, 0], trajectories[t, one_classes, 1],
#             'o', color=colors[1], alpha=0.2+0.8*t/len(time_span))
#         return scatter_zero, scatter_one

#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111)
#     anim = FuncAnimation(fig, animate_frame, frames=len(time_span))
#     plt.close(fig)
#     return anim