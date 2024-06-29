import torch

import numpy as np
import matplotlib.pyplot as plt


def plot_ds_stream(ds, trajectory: np.ndarray, title: str = None,
                   space_stretch: float = 5, stream_density: float = 1.0,
                   policy_density: int = 100, traj_density: int = 0.4,
                   n_samples: int = 1000,
                   other_starts = None, n_reprod_trajs: int = 3):
    """ Plot a policy for given a DS model and trajectories.

    Args:
        ds (PlanningPolicyInterface): A dynamical system for motion generation task.
        trajectory (np.ndarray): Input trajectory array (n_samples, dim).
        title (str, optional): Title of the plot. Defaults to None.
        space_stretch (float, optional): How much of the entire space to show in vector map.
            Defaults to 1.

        stream_density (float, optional): Density of policy streams. Defaults to 1.0.
        policy_density (int, optional): Density of on-trajectory policy arrows. Defaults to 10.
        traj_density (int, optional): Density of expert's trajectories. Defaults to 0.4.
        file_name(str, optional): Name of the plot file. Defaults to "".
        save_dir(str, optional): Provide a save directory for the figure. Leave empty to
            skip saving. Defaults to "".
        n_samples (int, optional): Number of samples in each demonstration. Defaults to 1000.
        other_starts (List[np.ndarray], optional): Other starting points to show stability.
            Defaults to None.
        n_reprod_trajs (int, optional): Number of trajectories to reproduce. Defaults to 10.
        show_legends (bool, optional): Opt to show the legends. Defaults to True.
    """

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

        dimension = trajectory.shape[1]
        if dimension == 2:
            x_min = np.min(trajectory[:, 0])
            y_min = np.min(trajectory[:, 1])
            x_max = np.max(trajectory[:, 0])
            y_max = np.max(trajectory[:, 1])
            return x_min, x_max, y_min, y_max

        else:
            raise NotImplementedError('Dimension not supported')

    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(trajectory)

    # calibrate the axis
    plt.figure(figsize=(10,10))

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])

    # plot the trajectory
    start_points = np.array([trajectory[idx * n_samples] for  \
                             idx in range(int(len(trajectory) / n_samples))])
    goal_point = trajectory[-1]

    trimed_trajectory_idx = np.random.choice(a=len(trajectory),
                                             size=int(traj_density * len(trajectory)),
                                             replace=False)
    trimed_trajectory = np.array(trajectory[trimed_trajectory_idx])
    plt.scatter(trimed_trajectory[:, 0], trimed_trajectory[:, 1],
                color="blue", marker='o',
                s=5, label='Expert Demonstrations')

    # generate the grid data
    x = np.linspace(x_min - space_stretch, x_max + space_stretch, policy_density)
    y = np.linspace(y_min - space_stretch, y_max + space_stretch, policy_density)
    X, Y = np.meshgrid(x, y)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    Z = np.apply_along_axis(lambda x: ds(torch.from_numpy(np.array([x], dtype=np.float32))).detach().numpy(), 1, data)
    U, V = Z[:,:,0].reshape(policy_density, policy_density), \
        Z[:,:,1].reshape(policy_density, policy_density)

    # create streamplot
    plt.streamplot(X, Y, U, V, density=stream_density, color="black", linewidth=1)

    # on-trajectory policy-rollouts
    dt: float = 0.01

    if n_reprod_trajs > len(start_points):
        n_reprod_trajs = len(start_points)

    starts_idx = np.random.choice(a=len(start_points), size=n_reprod_trajs, replace=False)
    starts = start_points[starts_idx]
    starts = starts + other_starts if other_starts is not None else starts
    limit = np.linalg.norm([(x_max - x_min), (y_max - y_min)]) / 10
    for idx, start in enumerate(starts):
        simulated_traj = []
        simulated_traj.append(np.array([start]).reshape(1, 2))

        distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)
        while  distance_to_target > limit  and len(simulated_traj) < 2e3:
            vel = ds(torch.from_numpy(np.array(simulated_traj[-1], dtype=np.float32))).detach().numpy()
            simulated_traj.append(simulated_traj[-1] + dt * vel)
            distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)

        simulated_traj = np.array(simulated_traj)
        simulated_traj = simulated_traj.reshape(simulated_traj.shape[0],
                                                simulated_traj.shape[2])
        plt.plot(simulated_traj[:, 0], simulated_traj[:, 1],
                color='red', linewidth=2)

    plt.savefig('stream_nn.png')
