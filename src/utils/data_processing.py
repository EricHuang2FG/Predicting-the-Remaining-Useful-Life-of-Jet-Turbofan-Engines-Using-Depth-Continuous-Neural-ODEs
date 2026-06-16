import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.utils.constants import (
    NUM_SETTINGS_AND_SENSOR_READINGS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FIGURE_SIZE,
)


# reference paper for this processing:
# https://link.springer.com/article/10.1007/s44196-024-00639-w#data-availability
def preprocess_training_data(
    file_path: str, window_size: int = DEFAULT_WINDOW_SIZE, max_rul: int = 100
) -> tuple[torch.FloatTensor, torch.FloatTensor, StandardScaler]:
    """preprocess CMAPSS data for training

    Preprocess raw CMAPSS data into PyTorch tensors such that they are training-ready. Three main steps are carried out
    in the pre-processing. 1) z-score standardization, 2) RUL clipping, which clips all RUL to be less than or equal to 100,
    and, 3) sliding time window processing, which packages the data into time-series of size 40

    Args:
        file_path (str): path to the raw CMAPSS data.
        window_size (int): window size of the sliding time window. Set to 40 for this experiment.
        max_rul (int): threshold for RUL clipping. Set to 100 for this experiment

    Returns:
        tuple[torch.FloatTensor, torch.FloatTensor, StandardScalar]: a tuple of the processed input data tensor,
        label tensor, and a record of the scaling parameters used for z-score standardization
    """

    # data has the size (num_lines, 26)
    # column 1: unit number
    # column 2: cycle count
    # column 3 to 5: operational settings
    # column 6 to end: sensor measurements
    data: np.ndarray = np.loadtxt(file_path)

    sliding_window_sequences: list = []
    rul_labels: list = []

    # engine_num = [1, 2, 3, 4, ..., ]
    for engine_num in np.unique(data[:, 0]):
        # get data s.t. its first column is the engine_num
        engine_data: np.ndarray = data[data[:, 0] == engine_num]
        max_life: int = len(engine_data)

        # perform RUL segmentation
        # element-wise subtraction of max_num_cycles - engine_data array
        # then cap every element at max_rul (ref the paper)
        rul: np.ndarray = max_life - engine_data[:, 1]
        rul_segmented: np.ndarray = np.minimum(max_rul, rul)

        # do sliding window processing
        settings_and_sensor_readings: np.ndarray = engine_data[:, 2:26]
        for i in range(len(settings_and_sensor_readings) - window_size + 1):
            sliding_window_sequences.append(
                settings_and_sensor_readings[i : i + window_size]
            )
            rul_labels.append(
                rul_segmented[i + window_size - 1]
            )  # rul at the end of the window

    # shape: (num_sequences, window_size, 24)
    input_data: np.ndarray = np.array(sliding_window_sequences)
    expected_output: np.ndarray = np.array(rul_labels)

    # standardize using Z = (x_i - u) / sigma
    # documentation for this one: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # it does Z = (x_i - u) / sigma for you
    scaler: StandardScaler = StandardScaler()
    orig_shape: tuple = input_data.shape
    input_data_flattened: np.ndarray = input_data.reshape(
        -1, NUM_SETTINGS_AND_SENSOR_READINGS
    )
    input_data_flattened = scaler.fit_transform(input_data_flattened)
    input_data = input_data_flattened.reshape(orig_shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(expected_output), scaler


def preprocess_test_data(
    test_data_path: str,
    rul_path: str,
    scaler: StandardScaler,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_rul: int = 100,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """preprocess CMAPSS data for testing

    Preprocess raw CMAPSS data into PyTorch tensors for model evaluation. It follows the same processes as
    preprocess_training_data(). In particular, the z-score standardization is done with the same scaling
    parameters as those used to preprocess the training data

    Args:
        test_data_path (str): path to the raw CMAPSS test data.
        rul_path (str): path to the CMAPSS file holding the ground-truth RUL corresponding to the test data.
        scalar (StandardScaler): record of scalar used to preprocess the training data
        window_size (int): window size of the sliding time window. Set to 40 for this experiment.
        max_rul (int): threshold for RUL clipping. Set to 100 for this experiment

    Returns:
        tuple[torch.FloatTensor, torch.FloatTensor]: a tuple of the processed input data tensor, and the
        ground truth RUL tensor
    """

    # similar to preprocess_trainng_data() but runs on test data and real RUL
    # also only takes the last window of each engine test
    test_data: np.ndarray = np.loadtxt(test_data_path)
    rul_data: np.ndarray = np.loadtxt(rul_path)

    # perform rul segmentation
    rul_data = np.minimum(max_rul, rul_data)

    test_sequence: list = []

    for engine_num in np.unique(test_data[:, 0]):
        engine_data: np.ndarray = test_data[test_data[:, 0] == engine_num]

        engine_sequence: np.ndarray = engine_data[-window_size:, 2:26]
        engine_sequence_size: int = engine_sequence.shape[0]
        if engine_sequence_size < window_size:
            filler: np.ndarray = np.zeros(
                (window_size - engine_sequence_size, NUM_SETTINGS_AND_SENSOR_READINGS)
            )
            engine_sequence = np.concatenate((filler, engine_sequence), axis=0)

        test_sequence.append(engine_sequence)

    input_data: np.ndarray = np.array(test_sequence)

    # standardize using scaler from TRAIN DATA
    orig_shape: tuple = input_data.shape
    input_data_flattened: np.ndarray = input_data.reshape(
        -1, NUM_SETTINGS_AND_SENSOR_READINGS
    )
    input_data_flattened = scaler.transform(input_data_flattened)
    input_data = input_data_flattened.reshape(orig_shape)

    return torch.FloatTensor(input_data), torch.FloatTensor(rul_data)


def split_tensors_by_ratio(
    tensor_a: torch.FloatTensor, tensor_b: torch.FloatTensor, ratio: float = 0.7
) -> tuple[
    tuple[torch.FloatTensor, torch.FloatTensor],
    tuple[torch.FloatTensor, torch.FloatTensor],
]:
    """split and return two tensors by ratio

    Each of tensor_a and tensor_b are splitted into two tensors with the specified ratio. The splitted
    tensor_a and tensor_b are stored in distinct tuples, which are together packaged into one tuple and
    returned.

    Args:
        tensor_a (torch.FloatTensor): first tensor to be splitted.
        tensor_b (torch.FloatTensor): second tensor to be splitted.
        ratio (float): the ratio at which the tensors should be splitted

    Returns:
        tuple[
            tuple[
                torch.FloatTensor,
                torch.FloatTensor
            ],
            tuple[
                torch.FloatTensor,
                torch.FloatTensor
            ]
        ]:
            tuple of tuples containing the splitted tensor_a and tensor_b
    """

    # slice each tensor such that it gives a tuple
    # containing a tensor with the first ratio fraction of elements of the original tensor which will be used for training the model
    # and a tensor that contains the remaining elements of the origina training tensor used for validating said model
    # the same is performed on tensor_a and tensor_b, and the results are returned as a tuple

    # for our use case, tensor_a and tensor_b have the same on its first dimension
    num_data = tensor_a.shape[0]
    split_location = int(ratio * num_data)

    randomized_indicies = torch.randperm(num_data)

    left_index, right_index = (
        randomized_indicies[:split_location],
        randomized_indicies[split_location:],
    )

    a_left, b_left = tensor_a[left_index], tensor_b[left_index]
    a_right, b_right = tensor_a[right_index], tensor_b[right_index]

    return ((a_left, a_right), (b_left, b_right))


def visualize_fd001_data(file_path: str, figure_dest: str) -> None:
    """visualize a column of data of a particular file

    Plot a graph of the data stored in column column_number of the file located at file_path

    Args:
        file_path (str): path to the raw CMAPSS data.
        column_number (int): the number of the column of interest of the data
        figure_dest (str): destination path of the plotted graph

    Returns:
        None
    """
    data: np.ndarray = np.loadtxt(file_path)
    y_1: np.ndarray = data[:192, 6]
    y_2: np.ndarray = data[:192, 7]
    y_3: np.ndarray = data[:192, 8]

    _, (ax_1, ax_2, ax_3) = plt.subplots(
        3,
        1,
        figsize=DEFAULT_FIGURE_SIZE,
        sharex=True,
    )

    x = np.arange(len(y_1))

    ax_1.plot(x, y_1, color="red", label="Sensor 2", linewidth=1)
    ax_2.plot(x, y_2, color="blue", label="Sensor 3", linewidth=1)
    ax_3.plot(x, y_3, color="purple", label="Sensor 4", linewidth=1)

    for ax in (ax_1, ax_2, ax_3):
        ax.set_ylabel("Sensor Reading", fontweight="bold", fontsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.legend(loc="lower right", fontsize=16)

    plt.xlabel("Cycle", fontweight="bold", fontsize=22)

    plt.savefig(figure_dest, dpi=300)
    plt.show()


if __name__ == "__main__":
    visualize_fd001_data(
        "CMAPSS/test_FD001.txt", "figures/FD001_sample_sensor_data.pdf"
    )
