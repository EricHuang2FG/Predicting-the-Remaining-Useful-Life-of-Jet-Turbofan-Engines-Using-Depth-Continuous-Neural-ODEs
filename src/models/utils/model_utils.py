import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from src.models.node import ODE
from src.models.cnn_node import CNN_ODE
from src.models.cnn import CNN
from src.models.cnn_lstm_attention import CNN_LSTM_Attention
from src.models.lstm import LSTM
from src.utils.constants import (
    DATASET_ID_FD001,
    DATASET_ID_FD002,
    DATASET_ID_FD003,
    DATASET_ID_FD004,
    NUM_SETTINGS_AND_SENSOR_READINGS,
    MODEL_TYPE_CNN_NODE,
    MODEL_TYPE_NODE,
    MODEL_TYPE_CNN,
    MODEL_TYPE_CNN_LSTM_ATTENTION,
    MODEL_TYPE_LSTM,
    DEFAULT_NETWORK_SETTINGS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_FIGURE_SIZE,
    LINE_WIDTH,
)
from src.utils.data_processing import (
    preprocess_training_data,
    preprocess_test_data,
    split_tensors_by_ratio,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_from_file(
    model_class: str,
    path: str = "models/ode.model",
    settings: dict = DEFAULT_NETWORK_SETTINGS,
) -> nn.Module:
    """initialize a PyTorch model and load the weights stored in a file at path

    The PyTorch model is first initialized with passed-in settings using initialize_model(),
    then, it is loaded with weights stored as a file at path

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        path (str): file path of the file storing the weights.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.

    Returns:
        nn.Module: model loaded with weights stored in the file at path.
    """

    # load the saved weights from file path into a new instance of a model
    model: nn.Module = initialize_model(model_class, settings=settings)
    model.load_state_dict(torch.load(path, map_location="cpu"))

    return model


def initialize_model(
    model_class: str, settings: dict = DEFAULT_NETWORK_SETTINGS
) -> nn.Module:
    """initialize a PyTorch model with passed-in settings

    An empty PyTorch model (i.e. model with non-determined weights and biases) of model_class with
    passed-in settings is initialized and returned

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.

    Returns:
        nn.Module: initialized model of model_class.
    """

    if model_class not in [
        MODEL_TYPE_NODE,
        MODEL_TYPE_CNN_NODE,
        MODEL_TYPE_CNN,
        MODEL_TYPE_CNN_LSTM_ATTENTION,
        MODEL_TYPE_LSTM,
    ]:
        raise ValueError(f"Unknown model_class: {model_class}")
    # for the settings dictionary, we use [] access because
    # we want the code to crash if such setting is not provided

    # create instance of model with basic settings
    if model_class == MODEL_TYPE_NODE:
        return ODE(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            hidden_dimension=settings["hidden_dimension"],
            encoder_dimension=settings["encoder_dimension"],
            regressor_dimension=settings["regressor_dimension"],
            dropout=settings["dropout"],
            sequence_length=DEFAULT_WINDOW_SIZE,
        )

    if model_class == MODEL_TYPE_CNN_NODE:
        return CNN_ODE(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            cnn_num_kernals=settings["cnn_num_kernals"],
            cnn_kernal_size=settings["cnn_kernal_size"],
            cnn_stride=settings["cnn_stride"],
            cnn_padding=settings["cnn_padding"],
            hidden_dimension=settings["hidden_dimension"],
            encoder_dimension=settings["encoder_dimension"],
            regressor_dimension=settings["regressor_dimension"],
            dropout=settings["dropout"],
            sequence_length=settings["sequence_length"],
        )

    if model_class == MODEL_TYPE_CNN:
        return CNN(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            cnn_num_kernals=settings["cnn_num_kernals"],
            cnn_kernal_size=settings["cnn_kernal_size"],
            cnn_stride=settings["cnn_stride"],
            cnn_padding=settings["cnn_padding"],
            dropout=settings["dropout"],
        )

    if model_class == MODEL_TYPE_CNN_LSTM_ATTENTION:
        return CNN_LSTM_Attention(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            cnn_num_kernals=settings["cnn_num_kernals"],
            cnn_kernal_size=settings["cnn_kernal_size"],
            cnn_stride=settings["cnn_stride"],
            cnn_padding=settings["cnn_padding"],
            pooling_kernal_size=settings["pooling_kernal_size"],
            pooling_stride=settings["pooling_stride"],
            pooling_padding=settings["pooling_padding"],
            lstm_hidden_dimension=settings["lstm_hidden_dimension"],
            lstm_num_layers=settings["lstm_num_layers"],
            attention_dimension=settings["attention_dimension"],
            dropout=settings["dropout"],
        )

    if model_class == MODEL_TYPE_LSTM:
        return LSTM(
            input_dimension=NUM_SETTINGS_AND_SENSOR_READINGS,
            lstm_layer_1_hidden_dimension=settings["lstm_layer_1_hidden_dimension"],
            lstm_layer_2_hidden_dimension=settings["lstm_layer_2_hidden_dimension"],
            dropout=settings["dropout"],
        )


def train_model(
    model_class: str,
    dest_path: str,
    input_data: torch.FloatTensor,
    expected_output: torch.FloatTensor,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    return_loss: bool = False,
    input_validation_data: torch.FloatTensor = None,  # pass in if early stopping is desired
    expected_validation_output: torch.FloatTensor = None,
    model_is_lstm: bool = False,
):
    """train a PyTorch model of model_class

    Train a PyTorch model of model_class with passed-in settings. The weights of the trained model is
    saved at dest_path. If input_validation_data and expected_validation_output are not None,
    then early stopping is employed.

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        dest_path (str): destination location where the trained weights are to be saved
        input_data (torch.FloatTensor): data for training
        expected_output (torch.FloatTensor): labels for training data
        settings (dict): a dictionary of settings of the particular model belonging to model_class.
        return_loss (bool): whether or not validation loss should be returned by the function.
        input_validation_data (torch.FloatTensor): validation data.
        expected_validation_output (torch.FloatTensor): labels for validation data

    Returns:
        nn.Module: trained model. If return_loss is true, the average training loss for the last epoch is returned in a tuple
    """
    model = initialize_model(model_class, settings=settings)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Mean Squared Error Loss, difference between prediction and true values
    loss_function = nn.MSELoss()
    # use Adam Algorithm to update weights to minimize loss
    # optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    if model_is_lstm:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings["lr"],
            weight_decay=settings["l2_regularization"],
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])

    dataset = TensorDataset(input_data, expected_output)
    dataloader = DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=True
    )  # feeds batches of data to train,
    # data is randomly ordered at beginning of each epoch
    model.train()

    lowest_validation_loss = float("inf")
    num_no_improvement_epochs = 0
    optimal_state = None

    # going through epochs one at a time
    epochs = settings["epochs"]
    for epoch in range(epochs):
        model.train()
        total_loss_per_epoch = 0
        # going through each pass of the training set (epoch) in batches
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()  # zero all the gradients first to reset model weights, only want the error associated with current batch
            predictions = model(x)
            # calls model to get the mode's prediction for the given input values
            # goes through a forward pass (automatically calls forward fucntion of the model type)
            loss = loss_function(predictions, y)
            loss.backward()
            # back propagation: figures out how much each parameter contributes to the final loss, take derivatives as look at changes
            # calculates gradient of the loss function wrt each parameter
            # hidden layers creates composite functions, thus require chain rule to find impact of earlier parameters on loss
            # chain rule takes into account gradient calculated at a later layer to find the gradient for an earlier layer

            # for Neural ODEs, this is too complicated due to its infinite dimensions
            # thus, use Adjoint Method that involves integrating another ODE backwards in time to find the assocaiated gradients with each parameter
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # updates the parameters/weights taking into account gradient values
            # size of each step is determined by learning rate
            total_loss_per_epoch += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}], loss: {(total_loss_per_epoch / len(dataloader)):.6f}"
        )

        # check using validation set to see if early stopping is required
        if input_validation_data is not None and expected_validation_output is not None:
            model.eval()
            with torch.no_grad():  # autograd disabled
                input_validation_data = input_validation_data.to(device)
                expected_validation_output = expected_validation_output.to(device)
                validation_predictions = model(input_validation_data)
                validation_loss = loss_function(
                    validation_predictions, expected_validation_output
                ).item()
                print(f"Current validation loss: {validation_loss}")

                if model_is_lstm:
                    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

            if (
                validation_loss <= lowest_validation_loss - 1e-4
            ):  # if it's better, to a 4 decimal places tolerance
                lowest_validation_loss = validation_loss
                num_no_improvement_epochs = 0
                optimal_state = (
                    model.state_dict()
                )  # saves a copy of best parameters at this point
            else:
                num_no_improvement_epochs += 1

            if num_no_improvement_epochs >= 5:
                print(
                    f"Early stopping at [{epoch + 1}/{epochs}], validation loss: {(lowest_validation_loss):.6f}"
                )
                model.load_state_dict(
                    optimal_state
                )  # restores the best model parameters
                break

    if optimal_state is not None:
        model.load_state_dict(optimal_state)

    torch.save(model.state_dict(), dest_path)

    if return_loss:
        return model, total_loss_per_epoch / len(
            dataloader
        )  # average training loss per batch for last epoch
    return model


def train_model_repeated(
    num_repeats: int,
    model_class: str,
    data_path: str,
    dest_path_template: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    custom_input_data: torch.FloatTensor = None,
    custom_expected_output: torch.FloatTensor = None,
    do_validation: bool = True,
    model_is_lstm: bool = False,
):
    if "{{ repeat_index }}" not in dest_path_template:
        raise ValueError(f"Invalid template: {dest_path_template}")

    x, y = custom_input_data, custom_expected_output
    if custom_input_data is None or custom_expected_output is None:
        x, y, _ = preprocess_training_data(data_path, window_size=window_size)

    if do_validation:
        (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
            x, y, ratio=0.9
        )
    else:
        x_train, y_train = x, y
        x_validation, y_validation = None, None

    for repeat_index in range(num_repeats):
        set_seed(42 + repeat_index)
        train_model(
            model_class,
            dest_path_template.replace("{{ repeat_index }}", str(repeat_index)),
            x_train,
            y_train,
            settings=settings,
            input_validation_data=x_validation,
            expected_validation_output=y_validation,
            model_is_lstm=model_is_lstm,
        )


def plot_model_evaluation_graph(
    model_path: str,
    prediction_array_cleaned_sorted: np.ndarray,
    ground_truth_sorted: np.ndarray,
    settings: dict,
    figure_dest: str,
    rmse: float,
    score: float,
):
    # extract the dataset id by inferring from the model names
    # if such extraction fails, the dataset id will not be included in the graph title
    dataset_id = ""
    if model_path:
        if "001" in model_path:
            dataset_id = DATASET_ID_FD001
        elif "002" in model_path:
            dataset_id = DATASET_ID_FD002
        elif "003" in model_path:
            dataset_id = DATASET_ID_FD003
        elif "004" in model_path:
            dataset_id = DATASET_ID_FD004

    # plotting
    x = list(range(len(ground_truth_sorted)))

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.plot(
        x,
        ground_truth_sorted,
        color="red",
        label="Actual RUL",
        linewidth=LINE_WIDTH,
    )
    plt.scatter(
        x,
        prediction_array_cleaned_sorted,
        color="purple",
        # label=f"Predicted RUL, RMSE: {rmse:.4f}, Score: {score:.4f}",
        label=f"Predicted RUL",
        s=16,
        # linewidth=LINE_WIDTH,
    )

    plt.xlabel("Engine Units", fontweight="bold", fontsize=22)
    plt.ylabel("RUL", fontweight="bold", fontsize=22)
    # plt.title(
    #     f"Actual and Predicted RUL of {settings["name"]}"
    #     + ("" if not dataset_id else f", {dataset_id}"),
    #     fontweight="bold",
    #     fontsize=25,
    # )
    plt.legend(loc="lower right", fontsize=18)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(True)
    plt.tight_layout(pad=2.8)
    plt.tick_params(axis="both", which="major", labelsize=22)
    plt.savefig(figure_dest, dpi=300)

    plt.show()


def score_function(predicted_output: np.ndarray, expected_output: np.ndarray) -> float:
    diff = predicted_output - expected_output

    less_than_mask = diff < 0
    greater_than_equal_to_mask = diff >= 0

    score_components = np.zeros_like(predicted_output, dtype=float)

    # e^((y - y_pred) / 13) - 1
    score_components[less_than_mask] = np.exp(-diff[less_than_mask] / 13) - 1

    # e^((y_pred - y) / 10) - 1
    score_components[greater_than_equal_to_mask] = (
        np.exp(diff[greater_than_equal_to_mask] / 10) - 1
    )

    return np.sum(score_components)


def evaluate_model(
    input_data: torch.FloatTensor,
    expected_output: torch.FloatTensor,
    model_class: str,
    model: nn.Module,  # pass this in directly, or, pass in None here but give the model_path
    model_path: str = "models/ode.model",
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    plot: bool = True,
    figure_dest: str = "figures/untitled_figure.pdf",
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """evaluates a PyTorch model

    Evaluates a PyTorch model of model_class by computing the RMSE and MAPE of the predicted RUL with the
    expected_output. The model is either directly passed in as an argument, or loaded as a file.
    If plot is True, graphs that visualizes predicted RULs and the corresponding expected_output will be plotted,
    and saved at figure_dest

    Args:
        input_data (torch.FloatTensor): data for evaluation.
        expected_output (torch.FloatTensor): ground-truth RULs for evaluation.
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        model (nn.Module): PyTorch model of the model to be evaluated.
        model_path (str): file path containing the weights of the model to be evaluated.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.
        plot (bool): whether or not graphs visualizing the predicted and actual RULs should be produced
        figure_dest (str): if plot is True, the destination location where the plotted graph should be saved

    Returns:
        tuple[float, float]: a tuple of the RMSE and MAPE values of the model when evaluated on input_data
    """
    if not model:
        model = load_model_from_file(model_class, path=model_path, settings=settings)
    model.eval()

    with torch.no_grad():
        predictions: torch.FloatTensor = model(input_data)

    prediction_array: np.ndarray = predictions.numpy()
    ground_truth_array: np.ndarray = expected_output.numpy()

    # clip to be below 100
    # prediction_array = np.minimum(100, prediction_array)
    ground_truth_array = np.minimum(100, ground_truth_array)

    # clip the value in the array to be meaningful values between 0 and 150
    # for bad values like -1500 that do not correspond with physical meaning (~4 of them out of 100)
    # replace with a default of 100
    prediction_array_cleaned: list = [
        value if 0 <= value <= 150 else 100 for value in prediction_array
    ]
    prediction_array_cleaned: np.ndarray = np.array(prediction_array_cleaned)

    sort_indicies: np.ndarray = np.argsort(ground_truth_array)
    ground_truth_sorted: np.ndarray = ground_truth_array[sort_indicies]
    prediction_array_cleaned_sorted: np.ndarray = prediction_array_cleaned[
        sort_indicies
    ]

    rmse = np.sqrt(np.mean((prediction_array_cleaned - ground_truth_array) ** 2))

    # avoid division by 0
    non_zero_mask = ground_truth_array != 0
    mape = np.mean(
        np.abs(
            (
                prediction_array_cleaned[non_zero_mask]
                - ground_truth_array[non_zero_mask]
            )
            / ground_truth_array[non_zero_mask]
        )
    )

    score = score_function(prediction_array_cleaned, ground_truth_array)

    print(f"RMSE: {rmse}, MAPE: {mape}, SCORE: {score}")

    if plot:
        plot_model_evaluation_graph(
            model_path,
            prediction_array_cleaned_sorted,
            ground_truth_sorted,
            settings,
            figure_dest,
            rmse,
            score,
        )

    return (
        rmse,
        mape,
        score,
        prediction_array_cleaned,
        ground_truth_sorted,
        sort_indicies,
    )


def evaluate_model_repeated(
    num_repeats: int,
    model_class: str,
    training_data_path: str,
    testing_data_path: str,
    ground_truth_path: str,
    model_path_template: str,
    figure_dest: str,
    settings: dict,
    custom_input_data: torch.FloatTensor = None,
    custom_expected_output: torch.FloatTensor = None,
    plot: bool = True,
) -> tuple[float, float, float]:
    if num_repeats <= 0:
        raise ValueError(f"Invalid number of repeats: {num_repeats}")

    if "{{ repeat_index }}" not in model_path_template:
        raise ValueError(f"Invalid template: {model_path_template}")

    x_test, y_test = custom_input_data, custom_expected_output
    if custom_input_data is None or custom_expected_output is None:
        _, _, scaler = preprocess_training_data(training_data_path)
        x_test, y_test = preprocess_test_data(
            testing_data_path, ground_truth_path, scaler
        )

    all_rmse, all_mape, all_score = [], [], []
    all_predictions, ground_truth_sorted = [], None
    sort_indicies = None

    for repeat_index in range(num_repeats):
        (
            curr_rmse,
            curr_mape,
            curr_score,
            curr_predictions,
            ground_truth_sorted,
            sort_indicies,
        ) = evaluate_model(
            x_test,
            y_test,
            model_class,
            None,
            model_path_template.replace("{{ repeat_index }}", str(repeat_index)),
            settings=settings,
            plot=False,
            figure_dest=None,
        )
        all_rmse.append(curr_rmse)
        all_mape.append(curr_mape)
        all_score.append(curr_score)

        if repeat_index == 0:
            all_predictions = [[val] for val in curr_predictions]
        else:
            for index, prediction in enumerate(curr_predictions):
                all_predictions[index].append(prediction)

    mean_rmse = np.mean(all_rmse)
    std_rmse = np.std(all_rmse)

    mean_mape = np.mean(all_mape)
    std_mape = np.std(all_mape)

    mean_score = np.mean(all_score)
    std_score = np.std(all_score)

    mean_predictions = np.array(
        [np.mean(predictions) for predictions in all_predictions]
    )

    # sort it identically as how we sorted the array of ground truth
    # so that the mapping between each ground truth and each prediction is preserved
    mean_predictions_sorted = mean_predictions[sort_indicies]

    if plot:
        plot_model_evaluation_graph(
            model_path_template,
            mean_predictions_sorted,
            ground_truth_sorted,
            settings,
            figure_dest,
            mean_rmse,
            mean_score,
        )

    print(
        f"Mean RMSE: {mean_rmse}, std {std_rmse}\nMean MAPE: {mean_mape}, std {std_mape}\nMean SCORE: {mean_score}, std {std_score}\n\n"
    )

    return mean_rmse, mean_mape, mean_score


def visualize_outliers(figure_dest: str) -> None:
    # x = np.array([42, 43, 44, 45, 46])
    x = ["FD001", "FD002", "FD003", "FD004"]
    scores_fd001 = np.array(
        [
            295.0284453740695,
            194.42733492583935,
            427.8466595521765,
            117.72722772418423,
            141.37437134937437,
        ]
    )
    scores_fd002 = np.array(
        [
            400.6805398460523,
            382.4040293510642,
            21171.391550779343,
            353.5892170340643,
            497.1350216088303,
        ]
    )
    scores_fd003 = np.array(
        [
            123.26766192913055,
            125.19246143861052,
            120.56476700618092,
            149.31516570832184,
            121.12721052419984,
        ]
    )
    scores_fd004 = np.array(
        [
            4620.340104706469,
            2223.2988111529303,
            2715.700261718732,
            3524.663500905037,
            10925.05218530685,
        ]
    )

    all_dataset_scores = [
        scores_fd001,
        scores_fd002,
        scores_fd003,
        scores_fd004,
    ]

    _, axes = plt.subplots(1, 4, figsize=DEFAULT_FIGURE_SIZE, sharey=False)

    for i, (ax, scores) in enumerate(zip(axes, all_dataset_scores)):
        # Horizontal jitter for clarity
        x_jitter = np.random.normal(1 if i != 3 else 0.9, 0.05, size=len(scores))
        ax.scatter(x_jitter, scores, color="blue", s=50)

        # Median line
        median = np.median(scores)
        ax.hlines(median, 0.8, 1.2, colors="red", linestyles=":", linewidth=3, label="Median")

        # Mean line
        mean = np.mean(scores)
        ax.hlines(mean, 0.8, 1.2, colors="purple", linestyles="--", linewidth=3, alpha=0.5, label="Mean")

        if x[i] == "FD002":
            plt.text(
                i,  # slightly to the right of the line
                median,    # vertical position at the median
                f"Median: {median:.0f}",
                color="blue",
                fontsize=12,
                va="center"  # vertical alignment
            )

        # Formatting
        ax.set_xticks([1])
        ax.set_xticklabels([x[i]], fontsize=18, fontweight="bold")
        # ax.set_title(x[i], fontsize=16, fontweight="bold")
        # ax.set_xlabel("Seed", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Optional: linear scale to highlight outliers
        ax.set_yscale("linear")
        ax.tick_params(axis="y", labelsize=16)
        # ax.tick_params(axis="x", labelsize=12)

        # Show legend only on the first subplot
        if i == 3:
            ax.legend(fontsize=16)

    axes[0].set_ylabel("Score", fontsize=24, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figure_dest, dpi=300)
    plt.show()

    # plt.figure(figsize=DEFAULT_FIGURE_SIZE)

    # for i, scores in enumerate(all_dataset_scores):
    #     # Add small horizontal jitter for strip plot
    #     x_jitter = np.random.normal(i, 0.05, size=len(scores))
    #     plt.scatter(
    #         x_jitter, scores, color="blue", s=50, label="Score" if i == 1 else ""
    #     )

    #     # Median line
    #     median = np.median(scores)
    #     plt.hlines(
    #         median,
    #         i - 0.3,
    #         i + 0.3,
    #         colors="blue",
    #         linestyles=":",
    #         linewidth=2,
    #         label="Median" if i == 1 else "",
    #     )

    #     # Mean line
    #     mean = np.mean(scores)
    #     plt.hlines(
    #         mean,
    #         i - 0.3,
    #         i + 0.3,
    #         colors="purple",
    #         linestyles="--",
    #         linewidth=2,
    #         alpha=0.5,
    #         label="Mean" if i == 1 else "",
    #     )

    # print(f"Mean: {mean}")
    # print(f"Median: {median}")

    # mean = scores.mean()
    # median = np.median(scores)

    # plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    # plt.plot(x, scores, "o", markersize=12, label="Score", color="red")

    # plt.axhline(
    #     median,
    #     linestyle=":",
    #     linewidth=LINE_WIDTH,
    #     label=f"Median: {median:.0f}",
    #     color="blue",
    # )
    # plt.axhline(
    #     mean,
    #     linestyle="--",
    #     linewidth=LINE_WIDTH,
    #     label=f"Mean: {mean:.0f}",
    #     color="purple",
    #     alpha=0.5,
    # )

    # plt.xlabel("Random Seed", fontweight="bold", fontsize=22)
    # plt.ylabel("SCORE (Log Scale)", fontweight="bold", fontsize=22)
    # plt.yscale("log")
    # plt.xticks(range(len(x)), x)
    # # plt.xticks(x)
    # plt.legend(fontsize=22)
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    # plt.grid(True)
    # plt.tick_params(axis="both", which="major", labelsize=24)
    # plt.tight_layout()

    # plt.savefig(figure_dest, dpi=300)
    # plt.show()


if __name__ == "__main__":
    visualize_outliers("figures/outliers.pdf")
