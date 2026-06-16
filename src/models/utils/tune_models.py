import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from src.utils.data_processing import (
    preprocess_training_data,
    split_tensors_by_ratio,
)
from src.models.utils.model_utils import (
    train_model,
    train_model_repeated,
    evaluate_model,
    evaluate_model_repeated,
)
from src.utils.constants import (
    DEFAULT_NETWORK_SETTINGS,
    LINE_WIDTH,
    OPTIMIZED_CNN_NODE_SETTINGS,
    MODEL_TYPE_NODE,
    MODEL_TYPE_CNN_NODE,
    MODEL_TYPE_CNN,
    DIMENSION_TYPE_ENCODER,
    DIMENSION_TYPE_HIDDEN,
    DIMENSION_TYPE_REGRESSOR,
    DIMENSION_TYPE_CNN_NUM_KERNALS,
    DATASET_ID_FD001,
    DATASET_ID_FD002,
    DATASET_ID_FD003,
    DATASET_ID_FD004,
)


def window_size_sweep(
    model_class: str,
    training_data_directory_template: str,
    figure_dest: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
) -> None:
    if "{{ dataset_id }}" not in training_data_directory_template:
        raise ValueError(
            f"Invalid training_data_directory_template: {training_data_directory_template}"
        )

    all_rmse, all_score = [], []
    candidate_window_sizes = [30, 40, 50, 60]
    for window_size in candidate_window_sizes:
        curr_settings = settings.copy()
        curr_settings["epochs"] = 25
        curr_settings["sequence_length"] = window_size

        all_rmse_per_window_size = []
        all_score_per_window_size = []
        for dataset_id in [
            DATASET_ID_FD001,
            DATASET_ID_FD002,
            DATASET_ID_FD003,
            DATASET_ID_FD004,
        ]:
            training_data_directory = training_data_directory_template.replace(
                "{{ dataset_id }}", str(dataset_id)
            )
            x, y, _ = preprocess_training_data(
                training_data_directory, window_size=window_size
            )
            (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
                x, y, ratio=0.9
            )

            model_path_template = (
                f"models/tune/tune.{dataset_id}.w{window_size}"
                + ".r{{ repeat_index }}.model"
            )
            train_model_repeated(
                5,
                model_class,
                None,
                model_path_template,
                settings=curr_settings,
                window_size=window_size,
                custom_input_data=x_train,
                custom_expected_output=y_train,
                do_validation=False
            )
            rmse, _, score = evaluate_model_repeated(
                5,
                model_class,
                None,
                None,
                None,
                model_path_template,
                None,
                curr_settings,
                custom_input_data=x_validation,
                custom_expected_output=y_validation,
                plot=False,
            )

            print(
                f"Window size {window_size}, {dataset_id} RMSE: {rmse}, SCORE: {score}"
            )

            all_rmse_per_window_size.append(rmse)
            all_score_per_window_size.append(score)

        average_rmse_per_window_size = np.mean(all_rmse_per_window_size)
        average_score_per_window_size = np.mean(all_score_per_window_size)

        print(
            f"WINDOW SIZE {window_size}, RMSE {average_rmse_per_window_size}, SCORE {average_score_per_window_size}"
        )

        all_rmse.append(average_rmse_per_window_size)
        all_score.append(average_score_per_window_size)

    _, (ax_1, ax_2) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    ax_1.plot(
        candidate_window_sizes,
        all_rmse,
        color="red",
        linewidth=LINE_WIDTH,
        marker="x",
        markersize=15,
    )
    ax_1.set_ylabel("RMSE", fontweight="bold", fontsize=20)
    ax_1.spines["top"].set_visible(False)
    ax_1.spines["right"].set_visible(False)
    ax_1.grid(True)
    ax_1.tick_params(axis="both", which="major", labelsize=22)

    ax_2.plot(
        candidate_window_sizes,
        all_score,
        color="blue",
        linewidth=LINE_WIDTH,
        marker="x",
        markersize=15,
    )
    ax_2.set_ylabel("SCORE", fontweight="bold", fontsize=20)
    ax_2.spines["top"].set_visible(False)
    ax_2.spines["right"].set_visible(False)
    ax_2.grid(True)
    ax_2.tick_params(axis="both", which="major", labelsize=22)

    plt.xlabel("Window Size", fontweight="bold", fontsize=20)

    plt.savefig(figure_dest, dpi=300)
    plt.show()


def learning_rate_sweep(
    model_class: str,
    training_data_directory: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
):
    """sweep through candidate learning rates and produce a graph of validation loss across each learning rate

    Defines a list of candidate learning rates and train a model using each. Then, evaluate the model using
    the validation set, compute the validation RMSE loss, and plot a graph comparing validation loss with
    the candidate learning rate.

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        training_data_directory (str): file path of the training data.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.

    Returns:
        None
    """
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.9
    )

    curr_settings = settings.copy()

    candidate_lrs = [0.1, 0.01, 0.005, 0.001, 0.0003]
    losses = []
    # for the sake of speed of training, evaluate based on 0.3 of train data
    # simply use the weights from the last epoch
    for lr in candidate_lrs:
        curr_settings["lr"] = lr
        model = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
        )
        losses.append(
            # only look at rmse for evaluation of model for specific parameter value
            evaluate_model(
                x_validation,
                y_validation,
                model_class,
                model,
                None,
                settings=curr_settings,
                plot=False,
            )[0]
        )

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_lrs, losses, marker="x")
    plt.xscale("log")
    plt.xlabel("Learning Rates")
    plt.ylabel("Validation RMSE")
    plt.grid(True)
    plt.show()


def hidden_dimensions_sweep(
    model_class: str,
    training_data_directory: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
    dimension_type: str = DIMENSION_TYPE_HIDDEN,
):
    """sweep through candidate hidden dimensions and produce a graph of validation loss across each hidden dimension

    Defines a list of candidate hidden dimensions depending on dimension_type, and train a model using each.
    Then, evaluate the model using the validation set, compute the validation RMSE loss, and plot a graph
    comparing validation loss with the candidate hidden dimension.

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        training_data_directory (str): file path of the training data.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.
        dimension_type (str): the type of hidden dimension to be sweeped

    Returns:
        None
    """
    if dimension_type not in [
        DIMENSION_TYPE_HIDDEN,
        DIMENSION_TYPE_ENCODER,
        DIMENSION_TYPE_REGRESSOR,
        DIMENSION_TYPE_CNN_NUM_KERNALS,
    ]:
        raise ValueError(f"Unexpected dimension type given: {dimension_type}")
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.9
    )

    curr_settings = settings.copy()

    candidate_hds = (
        [32, 64, 128]
        if dimension_type != DIMENSION_TYPE_CNN_NUM_KERNALS
        else [2, 3, 4]
        # else [20, 24, 28]
        # else [4, 8, 12, 16, 20]
    )
    losses = []
    for hd in candidate_hds:
        curr_settings[dimension_type] = hd
        model = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
        )
        losses.append(
            evaluate_model(
                x_validation,
                y_validation,
                model_class,
                model,
                None,
                settings=curr_settings,
                plot=False,
            )[0]
        )

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_hds, losses, marker="x")
    plt.xlabel(f"{dimension_type} for ODE")
    plt.ylabel("Validation RMSE")
    plt.grid(True)
    plt.show()


def dropout_rate_sweep(
    model_class: str,
    training_data_directory: str,
    settings: dict = DEFAULT_NETWORK_SETTINGS,
):
    """sweep through candidate dropout rates and produce a graph of validation loss across each dropout rate

    Defines a list of candidate dropout rates and train a model using each. Then, evaluate the model using
    the validation set, compute the validation RMSE loss, and plot a graph comparing validation loss with
    the candidate dropout rate.

    Args:
        model_class (str): class of the model. Either MODEL_TYPE_NODE or MODEL_TYPE_CNN_NODE.
        training_data_directory (str): file path of the training data.
        settings (dict): a dictionary of settings of the particular model belonging to model_class.

    Returns:
        None
    """
    x, y, _ = preprocess_training_data(training_data_directory)

    (x_train, x_validation), (y_train, y_validation) = split_tensors_by_ratio(
        x, y, ratio=0.9
    )

    curr_settings = settings.copy()

    # tuning dropout rate to see if overfit to training data
    # and need to remove some neurons so others dont become too generalized and overall more generalizable
    candidate_dor = [0, 0.05, 0.1, 0.2, 0.3]
    losses = (
        []
    )  # error/difference between the predicted value and the correct RUL values
    total_loss = (
        []
    )  # error/difference between the predicted value and the corresponding training data value
    for dor in candidate_dor:
        curr_settings["dropout"] = dor
        # trained_loss returns average loss per batch for last epoch
        trained_model, trained_loss = train_model(
            model_class,
            "models/tunning_dummy.model",
            x_train,
            y_train,
            settings=curr_settings,
            return_loss=True,
        )
        losses.append(
            evaluate_model(
                x_validation,
                y_validation,
                model_class,
                trained_model,
                None,
                settings=curr_settings,
                plot=False,
            )[0]
        )
        total_loss.append(trained_loss)

    plt.figure(figsize=(12, 9))
    plt.plot(candidate_dor, losses, color="red", marker="x")
    plt.plot(candidate_dor, total_loss, color="blue", marker="o")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Validation/Training RMSE")
    plt.grid(True)
    plt.show()

    # general rule of thumb: if training loss << validation loss, increase dropout because overfitting, vice versa
    print(
        f"total losses from training set after 25 epochs {[round(l, 6) for l in total_loss]}"
    )


if __name__ == "__main__":
    # general order of tuning: learning rate, hidden dimensions, dropout rate
    # important to hold seed constant as then can compare between different neural networks with variations in their parameters
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    settings: dict = OPTIMIZED_CNN_NODE_SETTINGS
    settings["epochs"] = 25  # 10 epochs for tuning suffice

    window_size_sweep(
        MODEL_TYPE_CNN_NODE,
        "CMAPSS/train_{{ dataset_id }}.txt",
        "figures/tune_window_size.pdf",
        settings=settings,
    )
