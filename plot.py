import torch
import numpy as np
import matplotlib.pyplot as plt

from os import PathLike
from typing import Dict, List
from tqdm import tqdm
from scipy.signal import savgol_filter

from ren import REN


def smooth_trajectory(trajectory, window_length=5, polyorder=1):
    """
    Apply Savitzky-Golay filter to smooth the trajectory.

    Parameters:
    trajectory (torch.Tensor): Tensor of shape (1, 100, 2)
    window_length (int): The length of the filter window (must be an odd number).
    polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
    torch.Tensor: Smoothed trajectory tensor of the same shape.
    """
    trajectory_np = trajectory.numpy()  # Convert to numpy and remove singleton dimension

    smoothed_np = np.zeros_like(trajectory_np)
    for t in range(trajectory_np.shape[0]):
        for i in range(trajectory_np.shape[2]):  # Loop over each dimension (x and y)
            smoothed_np[t, :, i] = savgol_filter(trajectory_np[t, :, i], window_length, polyorder)

    return torch.tensor(smoothed_np)  # Convert back to torch tensor and add singleton dimension


class PlotConfigs:
    """Hardcoded plot configurations.
    """

    COLORS = ["blue", "orange", "green", "purple", "brown"]
    FMTS = ['d--', 'o-', 's:', 'x-.', '*-', 'd--', 'o-']

    FIGURE_SIZE = (10, 10)
    FIGURE_DPI = 120
    POLICY_COLOR = 'grey'
    TRAJECTORY_COLOR = '#377eb8'
    ROLLOUT_ORIGINAL_COLOR = '#ff7f00'
    ROLLOUT_NOISY_COLOR = 'grey'
    ANNOTATE_COLOR = 'black'
    STAR_WIDTH = 2
    STAR_SIZE = 480
    CROSS_WIDTH = 10
    CROSS_SIZE = 1
    TICKS_SIZE = 16
    LABEL_SIZE = 18
    LEGEND_SIZE = 25
    TITLE_SIZE = 18
    FILE_TYPE = "png"
    REFERENCE_SIZE = 18
    ROLLOUT_NOISY_LINEWIDTH = 0.2
    ROLLOUT_ORIGINAL_LINEWIDTH = 2



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


def plot_trajectories(rollouts: List[np.ndarray], reference: np.ndarray,
                      save_dir: PathLike, plot_name: str, space_stretch = 0.1,
                      show_legends: bool = False, no_ticks: bool = True):
    """ Plot the rollout and reference trajectories.

    Args:
        rollouts (List[np.ndarray]): Rollout trajectories, noisy, true etc.
        references (List[np.ndarray]): Reference trajectories.
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

    blue_dots = plt.scatter(reference[:, :, 0], reference[:, :, 1], s=PlotConfigs.REFERENCE_SIZE, marker='o',
                c=PlotConfigs.TRAJECTORY_COLOR, label='Expert data', zorder=2)

    if rollouts is not None:
        # plot original rollouts
        rollouts_o = rollouts[0]
        for tr in rollouts_o:
            for batch_idx in range(tr.shape[0]):
                o_rollouts_handle = plt.plot(tr[batch_idx, :, 0], tr[batch_idx, :, 1], linewidth=PlotConfigs.ROLLOUT_ORIGINAL_LINEWIDTH,
                                            c=PlotConfigs.ROLLOUT_ORIGINAL_COLOR, zorder=2, label='True IC')

                start_handle = plt.scatter(tr[batch_idx, 0, 0], tr[batch_idx, 0, 1], marker='x',
                                color=PlotConfigs.ANNOTATE_COLOR, linewidth=PlotConfigs.CROSS_WIDTH,
                                s=PlotConfigs.CROSS_SIZE, label='Start', zorder=3)

        # plot noisy rollouts
        rollouts_n = rollouts[1]
        for tr in rollouts_n:
            for batch_idx in range(tr.shape[0]):
                n_rollouts_handle = plt.plot(tr[batch_idx, :, 0], tr[batch_idx, :, 1], linewidth=PlotConfigs.ROLLOUT_NOISY_LINEWIDTH,
                                            c=PlotConfigs.ROLLOUT_NOISY_COLOR, zorder=1, label='Noisy IC')

                start_handle = plt.scatter(tr[batch_idx, 0, 0], tr[batch_idx, 0, 1], marker='x',
                                color=PlotConfigs.ANNOTATE_COLOR, linewidth=PlotConfigs.CROSS_WIDTH,
                                s=PlotConfigs.CROSS_SIZE, label='Start', zorder=3)

    target_handle = plt.scatter(reference[0, -1, 0], reference[0, -1, 1], marker='*',
                                color=PlotConfigs.ANNOTATE_COLOR, linewidth=PlotConfigs.STAR_WIDTH,
                                s=PlotConfigs.STAR_SIZE, label='Target',
                                zorder=3)

    if show_legends:
        plt.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='upper left',
                   handles=[blue_dots, o_rollouts_handle[0], n_rollouts_handle[0],
                            start_handle, target_handle], facecolor='white', framealpha=1)

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    if no_ticks:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')



def plot_start_template(reference: np.ndarray, save_dir: PathLike, plot_name: str,
                        space_stretch = 0.1, show_legends: bool = True,
                        no_ticks: bool = True):
    """ Plot the rollout and reference trajectories.

    Args:
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

    blue_dots = plt.scatter(reference[:, :, 0], reference[:, :, 1],
                            s=PlotConfigs.REFERENCE_SIZE * 5, marker='o',
                            c=PlotConfigs.TRAJECTORY_COLOR, label='        ', zorder=2)

    # start_handle = plt.scatter(reference[0, 0, 0], reference[0, 0, 1], marker='x',
    #                 color=PlotConfigs.ANNOTATE_COLOR, linewidth=25,
    #                 s=8, label='Start', zorder=3)

    target_handle = plt.scatter(reference[0, -1, 0], reference[0, -1, 1], marker='*',
                                color=PlotConfigs.ANNOTATE_COLOR, linewidth=PlotConfigs.STAR_WIDTH,
                                s=PlotConfigs.STAR_SIZE, label='Target',
                                zorder=3)

    stable = plt.plot(reference[0, 0, 0], reference[0, 0, 1], linewidth=5,
                c="#F08080", label='        ', zorder=2)

    high_cont = plt.plot(reference[0, 0, 0], reference[0, 0, 1], linewidth=5,
                c="#8B4513", label='        ', zorder=2)

    low_cont = plt.plot(reference[0, 0, 0], reference[0, 0, 1], linewidth=5,
                c="#F4A460", label='        ', zorder=2)

    if show_legends:
        plt.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='lower center',
                   handles=[blue_dots, stable[0], high_cont[0], low_cont[0],
                            ],
                            ncol=2)

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    if no_ticks:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_multiple_motions(rollout_sets: Dict[str, List[np.ndarray]], reference: np.ndarray,
                          save_dir: PathLike, plot_name: str, space_stretch = 0.25,
                          show_legends: bool = True, no_ticks: bool = True):
    """ Plot the rollout and reference trajectories for multiple sets.

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
    axes.grid()

    blue_dots = plt.scatter(reference[:, :, 0], reference[:, :, 1], s=PlotConfigs.REFERENCE_SIZE * 5, marker='o',
                c=PlotConfigs.TRAJECTORY_COLOR, label='Expert', zorder=2)

    if show_legends:
        plt.legend(fontsize=PlotConfigs.LEGEND_SIZE - 5, loc='upper right',
            handles=[blue_dots])

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    if no_ticks:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE, dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_3d_trajectories(rollouts: List[np.ndarray], reference: np.ndarray,
                         save_dir: PathLike, plot_name: str, space_stretch = 0.2,
                         show_legends: bool = False):
    """ Plot the rollout and reference trajectories for 3D data.

    Args:
        trajectories (List[np.ndarray]): Rollouts.
        references (List[np.ndarray]): Reference.
        save_dir (PathLike): Save directory.
        plot_name (str): Name of the plot file.
    """

    # find trajectory limits
    fig = plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)
    ax = fig.add_subplot(111, projection='3d')

    blue_dots = ax.scatter(reference[:, :, 0], reference[:, :, 1], reference[:, :, 2],
                           s=PlotConfigs.REFERENCE_SIZE, marker='o', c=PlotConfigs.TRAJECTORY_COLOR,
                           label='Expert data', zorder=2)

    # TODO: add terminal points for robomimic

    # plot original rollouts
    rollouts_o = rollouts[0]
    for tr in rollouts_o:
        for batch_idx in range(tr.shape[0]):
            rollout_dots = ax.plot(tr[batch_idx, :, 0], tr[batch_idx, :, 1], tr[batch_idx, :, 2],
                                      linewidth=PlotConfigs.ROLLOUT_NOISY_LINEWIDTH * 2, c=PlotConfigs.ROLLOUT_ORIGINAL_COLOR,
                                      zorder=1, label='Policy Rollout')

            start_handle = ax.scatter(tr[batch_idx, 0, 0], tr[batch_idx, 0, 1], tr[batch_idx, 0, 2],
                                      marker='x', color=PlotConfigs.ANNOTATE_COLOR, linewidth=2,
                                      s=PlotConfigs.ANNOTATE_SIZE, label='Start', zorder=3)

    if show_legends:
        ax.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='upper left',
                  handles=[blue_dots, rollout_dots[0], start_handle])

    ax.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)
    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE,
                dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_mse_box(data: List[np.ndarray], labels: List[str], save_dir: PathLike, plot_name: str):
    """ Plot box plots for MSE data.

    Args:
        data (List[np.ndarray]): Input data for the box plot. A list which contains multiple
            numpy vectors, each representing a box.

        labels (List[str]): Input labels for the box plot. A list which contains multiple
        str labels, each tagging a box.

        save_dir (PathLike): Save directory for the plot file.
        plot_name (str): Name of the plot to be saved by.
    """
    plt.figure(figsize=(8, 2))
    box = plt.boxplot(data, vert=False, patch_artist=True, showmeans=True)

    # Customizing box colors
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = ['#0072B2', '#D55E00', '#E69F00', '#009E73', '#56B4E9']

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.yticks([1, 2, 3], labels)
    plt.xlabel('Dynamic Time Warping (Soft-DTW)')

    plt.savefig(f'{save_dir}/{plot_name}.{PlotConfigs.FILE_TYPE}', format=PlotConfigs.FILE_TYPE,
                dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')


def plot_trajectories_time(ren: REN, reference: np.ndarray, horizon: int, save_dir: PathLike, plot_name: str,
                           space_stretch = 0.5, density: int = 10):
    """ Plot the rollout and reference trajectories.

    Args:
        ren (REN): The ren module to generate trajectories.
        trajectories (List[np.ndarray]): Rollouts.
        references (List[np.ndarray]): Reference.
        save_dir (PathLike): Save directory.
        plot_name (str): Name of the plot file.
    """

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
    u_in = torch.zeros(1, 1, 2, device="cpu")

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

# TODO:
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

if __name__ == '__main__':
    plot_mse_box()