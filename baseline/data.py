import numpy as np

# dataset configs
N_SAMPLES_LASA_HANDWRITING = 1000
N_DEMONSTRATIONS_LASA_HANDWRITING = 7

def load_pylasa_data(motion_shape: str = "Angle", plot_data: bool = False,
    calibrated: bool = True, normalized: bool = True, n_dems: int = 10):
    """ Facilitate the handling of LASA handwriting dataset.

    Refer to https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt for
        more info about the dataset and attributes.

    In a quick glance, the dataset objects look like this:
        dt: the average time steps across all demonstrations
        demos: A structure variable containing necessary information
            about all demonstrations. The variable 'demos' has the following
            format:
            - demos{n}: Information related to the n-th demonstration.
            - demos{n}.pos (2 x 1000): matrix representing the motion in 2D
                space. The first and second rows correspond to
                x and y axes in the Cartesian space, respectively.
            - demons{n}.t (1 x 1000): vector indicating the corresponding time
                for each data point.
            - demos{n}.vel (2 x 1000): matrix representing the velocity of the motion.
            - demos{n}.acc (2 x 1000): matrix representing the acceleration of the motion.

    Args:
        motion_shape (str, optional): Choose a motion shape. A list of possible options
            may be found in this file. Defaults to "Angle". Possible options are [Angle, GShape, CShape,
            BendedLine, JShape, Multi_Models_1 to Multi_Models_4, Snake, Sine, Worm, PShape, etc.]. Complete list
            can be found in hw_data_module.dataset.NAMES_.

        plot_data (bool, optional): Whether to plot the designated motion or not. Defaults to False.

    Raises:
        NotImplementedError: Raised if the motion demonstrations are not available in dataset.

    Returns:
        Tuple(np.ndarray, np.ndarray): positions, velocities
    """
    # unusual import here to suppress the log
    import pyLasaDataset as hw_data_module

    lasa_selected_motions = ["GShape", "PShape", "Sine", "Worm", "Angle", "CShape", "NShape", "DoubleBendedLine"]
    lasa_dataset_motions = hw_data_module.dataset.NAMES_

    # list of tested motion data
    data = getattr(hw_data_module.DataSet, motion_shape)

    # extract pos and vel data
    pos_list = list()
    vel_list = list()

    for dem_index, demo in enumerate(data.demos):
        calibrated_pos = calibrate(demo.pos) if calibrated else demo.pos
        normalized_vel = normalize(demo.vel) if normalized else demo.vel
        normalized_pos = normalize(calibrated_pos) if normalized else calibrated_pos

        demo.pos = normalized_pos
        demo.vel = normalized_vel

        pos_list.append(normalized_pos)
        vel_list.append(normalized_vel)

        if dem_index + 1 == n_dems:
            print(f'Stopping at maximum {n_dems} demonstrations')
            break

    if plot_data:
        hw_data_module.utilities.plot_model(data)

    # concatenate the results
    concatenated_pos = np.concatenate(pos_list, axis=1)
    concatenated_vel = np.concatenate(vel_list, axis=1)

    return concatenated_pos.T, concatenated_vel.T


def calibrate(pos):
    """ Each dimension is shifted so that the last data point ends in the origin.

    Args:
        pos (np.ndarray): The positions array in the shape of (n_dim * n_samples).

    Returns:
        np.ndarray: The shifted positions array ending in origin.
    """

    return np.array([p - p[-1] for p in pos])


def normalize(arr: np.ndarray):
    """ Normalization of data in the form of array. Each row is first
    summed and elements are then divided by the sum.

    Args:
        arr (np.ndarray): The input array to be normalized in the shape of (n_dim, n_samples).

    Returns:
        np.ndarray: The normalized array.
    """

    assert arr.shape[0] < arr.shape[1]
    max_magnitude = np.max(np.linalg.norm(arr, axis=0))
    return arr / max_magnitude
