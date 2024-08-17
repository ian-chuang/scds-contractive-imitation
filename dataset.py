import os
import torch
import numpy as np

from enum import Enum
from typing import List

from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, TensorDataset

from robomimic.utils import file_utils as FileUtils
from robomimic.utils.dataset import SequenceDataset
from robomimic import DATASET_REGISTRY


class DatasetKeys(Enum):
    EEF_POS = "robot0_eef_pos"
    EEF_QUAT = "robot0_eef_quat"
    GRIPPER_QPOS = "robot0_gripper_qpos"
    JOINT_POS = "robot0_joint_pos"
    JOINT_VEL = "robot0_joint_vel"


def download_robomimic_data(data_root_dir: os.PathLike = os.path.join(os.getcwd(), 'data', 'robomimic'),
                            tasks: List[str] = ["lift", "can", "square", "transport"]):
    """

    Args:
        data_root (str, optional): _description_. Defaults to os.path.join(os.getcwd(), 'data', 'robomimic').
    """
    os.makedirs(data_root_dir, exist_ok=True)

    # download the dataset
    dataset_type = "ph"
    hdf5_type = "low_dim"
    for task in tasks:
        download_dir = os.path.join(data_root_dir, task)
        os.makedirs(download_dir)

        FileUtils.download_url(
            url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"],
            download_dir=download_dir,
        )


def robomimic_expert(task: str, horizon: int, device: str, batch_size,
                    state_only: bool = True, num_exp_trajectories: int = 7,
                    num_aug_trajectories: int = 0, ic_noise_rate: float = 0.00,
                    data_root_dir: os.PathLike = os.path.join(os.getcwd(), 'data', 'robomimic')):

    # enforce that the dataset exists
    data_dir = os.path.join(data_root_dir, task, "low_dim_v141.hdf5")
    if not os.path.exists(data_dir):
        print(f'No dataset in {data_dir}, downloading robomimic data...')
        download_robomimic_data(data_root_dir)

    dataset = SequenceDataset(
        hdf5_path=data_dir,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions",
        ),
        load_next_obs=False,
        frame_stack=1,
        seq_length=1,                   # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )

    print(f'\nRobomimic data loaded for {task}')
    print(dataset)
    print("")

    # def ic_trajectory_collate_fn(batch):

    #     out = []
    #     for sample in batch:
    #         out.append((sample["obs"][DatasetKeys.EEF_POS.value][0], sample["obs"][DatasetKeys.EEF_POS.value][1:]))

    #     return out

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,      # don't provide last batch in dataset pass if it's less than 100 in size,
        # collate_fn=ic_trajectory_collate_fn
    )

    return data_loader


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
                                shuffle=False)

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


# plot some trajectories
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for task in ["lift"]: # ["lift", "can", "square", "transport"]:
        robomimic_loader = robomimic_expert(task=task, horizon=50, device='cuda:0', batch_size=48)


        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        for sample in robomimic_loader:

            data = sample["obs"][DatasetKeys.EEF_POS.value]

            if data.size(2) == 4:
                # Convert to numpy array
                quat_np = data.squeeze().detach().cpu().numpy()

                # Create a scipy Rotation object
                r = R.from_quat(quat_np)

                # Convert to Euler angles
                euler_angles_np = r.as_euler('xyz', degrees=False)

                # Convert back to torch tensor
                data = torch.tensor(euler_angles_np, dtype=torch.float32).unsqueeze(1)

            # plot and investigate the waypoints: customized for 3D
            x = data[:50, 0, 0]
            y = data[:50, 0, 1]
            z = data[:50, 0, 2]

            ax.scatter(x, y, z, c='b', s=0.1, label='data')

            break

        plt.tick_params(axis='both', which='both', labelsize=16)
        plt.savefig(f'robomimic_{task}_ef_data.png', bbox_inches='tight')
