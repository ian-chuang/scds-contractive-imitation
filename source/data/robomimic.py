import os
import h5py

import torch
import numpy as np

from enum import Enum
from typing import Dict, List, Optional, Union

from scipy.spatial.transform import Rotation as R
import torch.utils
from torch.utils.data import DataLoader, Dataset

from robomimic.utils import file_utils as FileUtils
from robomimic import DATASET_REGISTRY
import torch.utils.data


class DatasetKeys(Enum):
    EEF_POS = "robot0_eef_pos"
    EEF_QUAT = "robot0_eef_quat"
    GRIPPER_QPOS = "robot0_gripper_qpos"
    JOINT_POS = "robot0_joint_pos"
    JOINT_VEL = "robot0_joint_vel"


class RobomimicDataset(Dataset):
    def __init__(self,
                 task: str,
                 data_root_dir: os.PathLike,
                 dataset_keys: List[DatasetKeys],
                 n_demos: Union[int, None],
                 device: Optional[str] = 'cuda:0'):
        """ Robomimic torch dataset, customized to return batches of initial
        conditions and full trajectories in state-only mode.

        # TODO: Move to a new file!!

        Args:
            task (str): Task name according to Robomimic dataset.
            data_root_dir (os.PathLike): Root directory to store the task data files.
            dataset_keys (List[DatasetKeys]): Specific keys to be included in the dataset.
                Examples are: [EF_POS, EF_QUAT] for full pose in task space.
                Note that Quaternion will be converted to Euler angles automatically.

            n_demos (int, None): Number of expert demonstrations. Use all if it's set to None.
        """

        assert task in ["lift", "can", "square", "transport"], \
            f'Task {task} is not supported!'

        # data properties
        self.data_root_dir = data_root_dir
        self.task = task
        self.obs_keys = dataset_keys
        self.device = device

        # check and download the data
        data_dir = os.path.join(data_root_dir, task, "low_dim_v141.hdf5")
        if not os.path.exists(data_dir):
            print(f'No dataset in {data_dir}, downloading robomimic data...')
            RobomimicDataset.download_robomimic_data(data_root_dir)

        # create data variables
        self.initial_conditions: List[torch.Tensor] = []
        self.expert_trajectories: List[torch.Tensor] = []

        # TODO: Quat to Euler and V.V.
        #     if data.size(2) == 4:
        # # Convert to numpy array
        # quat_np = data.squeeze().detach().cpu().numpy()

        # # Create a scipy Rotation object
        # r = R.from_quat(quat_np)

        # # Convert to Euler angles
        # euler_angles_np = r.as_euler('xyz', degrees=False)

        # # Convert back to torch tensor
        # data = torch.tensor(euler_angles_np, dtype=torch.float32).unsqueeze(1)

        # load the file
        with h5py.File(data_dir, 'r') as file:

            # build the demonstration ids
            if n_demos is None:
                n_demos = len(list(file["data"].keys()))
                print(f'Selecting the default number of demonstrations: {n_demos}')

            demo_ids = [f'demo_{idx}' for idx in range(n_demos)]

            # load the sample
            for demo_id in demo_ids:
                demo = file["data"][demo_id]["obs"]

                # iterate over observation keys
                obs = torch.cat([torch.Tensor(np.array(demo[obs_key])) for obs_key in self.obs_keys], dim=1)

                # store obs
                self.initial_conditions.append(obs[:1, :])
                self.expert_trajectories.append(obs[:, :])
                # print(obs[:1, :].shape, obs[:, :].shape)

        self.expert_trajectories = self.add_padding()
        print(f'Robomimic "{task}" dataset loaded with {n_demos} demos')

    def __len__(self):
        return len(self.expert_trajectories)

    def __getitem__(self, idx):
        return self.initial_conditions[idx].to(self.device), self.expert_trajectories[idx].to(self.device)

    def add_padding(self):
        # find the maximum length of the trajectories
        max_length = max(traj.shape[0] for traj in self.expert_trajectories)

        # pad each trajectory to the maximum length
        padded_trajectories = []
        for traj in self.expert_trajectories:
            length = traj.shape[0]
            if length < max_length: # pad with the last element (presumably the target)
                last_element = traj[-1, :].unsqueeze(0)
                padding = last_element.repeat(max_length - length, 1)
                padded_traj = torch.cat((traj, padding), dim=0)
            else:
                padded_traj = traj
            padded_trajectories.append(padded_traj)

        return padded_trajectories

    def download_robomimic_data(self):
        """

        Args:
            data_root (str, optional): _description_. Defaults to os.path.join(os.getcwd(), 'data', 'robomimic').
        """
        os.makedirs(self.data_root_dir, exist_ok=True)

        # download the dataset
        dataset_type = "ph"
        hdf5_type = "low_dim"

        download_dir = os.path.join(self.data_root_dir, self.task)
        os.makedirs(download_dir)

        FileUtils.download_url(
            url=DATASET_REGISTRY[self.task][dataset_type][hdf5_type]["url"],
            download_dir=download_dir,
        )


def robomimic_expert(task: str, device: str, batch_size: int,
                     dataset_keys: List[DatasetKeys], state_only: bool = True,
                     data_root_dir: Union[os.PathLike, None] = None,
                     n_demos: Union[int, None] = None):

    if data_root_dir is None:
        data_root_dir = os.path.join(os.getcwd(), 'data', 'robomimic')

    # load the data
    dataset = RobomimicDataset(task, data_root_dir, dataset_keys, n_demos, device=device)

    # build dataloader object
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=True, drop_last=False)

    return data_loader
