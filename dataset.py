import torch
import numpy as np

from typing import List
from torch.utils.data import DataLoader, TensorDataset


def lasa_expert(motion_shape: str, horizon: int, device: str, batch_size,
                state_only: bool = True, num_exp_trajectories: int = 7,
                num_aug_trajectories: int = 0, ic_noise_rate: float = 0.00):
    """ Dataset form the LASA handwriting dataset. This dataset is a good starting point
    but the state dimension is often 2 or 4.

    Args:
        motion_shape (str): Shape of the motion according to the handwriting dataset.
        Examples:

        ['GShape', 'Saeghe', 'Angle', 'DoubleBendedLine', 'JShape', 'CShape', 'WShape',
        'Spoon', 'BendedLine', 'NShape', 'Multi_Models_4', 'Sshape', 'Multi_Models_2',
        'Sine', 'Multi_Models_1', 'Khamesh', 'Leaf_1', 'Worm', 'RShape', 'Trapezoid',
        'Sharpc', 'Leaf_2', 'LShape', 'JShape_2', 'Multi_Models_3', 'Zshape', 'Snake',
        'PShape', 'Line']

        horizon (int): Length of the trajectory. This can be interpreted also as the
            granularity of imitation in a fixed horizon.

        device (str): Computational device, cuda:i or cpu.
        batch_size (int, optional): Batch size of the data loader. Defaults to 64.

        state_only (bool, optional): Return state only trajectories, without velocities.

        num_exp_trajectories (int, optional): Number of expert demonstrations picked from
            the LASA dataset.
        num_aug_trajectories (int, optional): Number of augmented demonstrations for EACH
            expert trajectory in the LASA dataset.


        noise_rate (float, optional): Added noise rate on entire trajectories.
            Defaults to 0.00.
        ic_noise_rate (float, optional): Noise rate for the initial condition distribution.
            Defaults to 0.00.

    Returns:
        torch.utils.data.DataLoader: Dataloader object ready for training
    """

    import pyLasaDataset as lasa

    # load motion data and normalize
    motion_data = getattr(lasa.DataSet, motion_shape).demos
    positions = [normalize(torch.Tensor(motion_data[idx].pos.T).T).T for idx in range(num_exp_trajectories)]
    velocities = [torch.Tensor(motion_data[idx].vel.T) for idx in range(num_exp_trajectories)] # TODO: normalize velocity?

    # load motion data into tensors
    x_train: torch.Tensor = torch.stack(positions, dim=0)
    y_train: torch.Tensor = torch.stack(velocities, dim=0)

    # send data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # down sample the original dataset
    num_samples: int = x_train.size(1)
    num_subsample: int = horizon

    # Create a range of indices to select elements at regular intervals
    step = num_samples // num_subsample
    indices = torch.arange(0, num_samples, step)[:num_subsample]

    x_train_ds = torch.from_numpy(x_train.cpu().numpy()[:, indices, :])
    x_train_ds = x_train_ds.to(device)

    # augmented trajectories
    x_train_ds_list: List[torch.Tensor] = []
    if num_aug_trajectories:
        for idx in range(num_aug_trajectories):
            horizon_idx = (idx % horizon) + 1

            shifted = x_train_ds[:, horizon_idx:, :]
            zero_col = torch.zeros(shifted.size(0), horizon_idx, shifted.size(2), device=device)  # Create a column of zeros
            shifted_with_zero = torch.cat((shifted, zero_col), dim=1)  # Append the zero column to the end

            # Add to the list
            x_train_ds_list.append(shifted_with_zero)

        x_train_ds = torch.cat(x_train_ds_list, dim=0)

    # dataset of initial conditions
    # TODO: Add noise with ic noise rate
    initial_conditions = x_train_ds[:, 0,:].unsqueeze(1)
    # y_init_noisy = stacked_y_init + ic_noise_rate * (2 * (torch.rand(batch_size, y_init.shape[1], y_init.shape[2], device=device) - 0.5))

    # state only data (y_train (velocities) not used)
    if state_only:
        dataset_state_only = TensorDataset(initial_conditions, x_train_ds)
        dataloader = DataLoader(dataset_state_only, batch_size=batch_size,
                                shuffle=True)

    # normal position, velocity data
    else:
        dataset_state_action = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset_state_action, batch_size=batch_size,
                                shuffle=True)

    return dataloader


def normalize(arr: torch.Tensor):
    """ Normalization of data in the form of array. Each row is first
    summed and elements are then divided by the sum.

    Args:
        arr (torch.Tensor): The input array to be normalized in the shape of (n_dim, n_samples).

    Returns:
        torch.Tensor: The normalized array.
    """

    assert arr.shape[0] < arr.shape[1]
    max_magnitude = torch.max(torch.norm(arr, dim=0))
    return arr / max_magnitude



def linear_expert(horizon, device, start_point=1.0):
    """ Generate synthetic linear data.

    NOTE: Toy example.

    Args:
        horizon (int): Length of the trajectory. This can be interpreted also as the granularity
            of imitation in a fixed horizon.

        device (str): Computational device, cuda:i or cpu.
        start_point (float, optional): Start point of the trajectory. Defaults to 1.0.

    Returns:
        torch.Tensor: Tensor dataset of the reference trajectory.
    """

    # create a 2D expert trajectory
    ref = torch.from_numpy(np.array([[i, i] for i in np.linspace(start_point, 0, horizon)], dtype=np.float32))
    ref = ref.unsqueeze(1)
    ref.to(device)
    return ref


def polynomial_expert(horizon, device, start_point=1.0, coefficients=[16, -16, 0.4, 0]):
    """ Generate synthetic polynomial data.

    NOTE: Toy example.

    Args:
        horizon (int): Length of the trajectory. This can be interpreted also as the granularity
            of imitation in a fixed horizon.

        device (str): Computational device, cuda:i or cpu.
        start_point (float, optional): Start point of the trajectory. Defaults to 1.0.. Defaults to 1.0.
        coefficients (list, optional): Polynomial coefficients with index 0 being the coefficient of the largest
            degree. Defaults to [16, -16, 0.4, 0].

    Returns:
        torch.Tensor: Tensor dataset of the reference trajectory.
    """

    # generate x values from start_point to 0
    x_values = np.linspace(start_point, 0, horizon, dtype=np.float32)

    # create the polynomial from the coefficients
    poly = np.poly1d(coefficients)

    y_values = poly(x_values)

    ref = torch.from_numpy(np.stack((y_values, x_values), axis=1))
    ref = ref.unsqueeze(1)  # add the extra dimension

    ref = ref.to(device)
    return ref