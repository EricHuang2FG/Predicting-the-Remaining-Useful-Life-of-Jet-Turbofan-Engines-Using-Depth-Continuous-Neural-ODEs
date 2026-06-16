import torch
from src.models.utils.model_utils import (
    evaluate_model,
    evaluate_model_repeated,
    train_model,
    train_model_repeated,
    set_seed,
)
from src.utils.data_processing import (
    preprocess_training_data,
    preprocess_test_data,
    split_tensors_by_ratio,
)
from src.utils.constants import (
    MODEL_TYPE_NODE,
    MODEL_TYPE_CNN_NODE,
    MODEL_TYPE_CNN,
    MODEL_TYPE_CNN_LSTM_ATTENTION,
    MODEL_TYPE_LSTM,
    OPTIMIZED_NODE_SETTINGS,
    OPTIMIZED_CNN_NODE_SETTINGS,
    OPTIMIZED_CNN_SETTINGS,
    OPTIMIZED_CNN_LSTM_ATTENTION_SETTINGS,
    OPTIMIZED_LSTM_SETTINGS,
)


def main() -> None:
    """execute this function to reproduce figures in the paper

    This function is currently setup to reproduce the figures presented in the paper
    that graphically compares model-predicted and ground-truth RULs. It also contains example code
    for model training, if desired by the user.
    """

    set_seed(42)

    # example code for training models
    train_model_repeated(
        5,
        MODEL_TYPE_NODE,
        "CMAPSS/train_FD002.txt",
        "models/node/node.FD002.r{{ repeat_index }}.v4.model",
        settings=OPTIMIZED_NODE_SETTINGS,
        model_is_lstm=False,
    )

    # example code for testing models
    evaluate_model_repeated(
        5,
        MODEL_TYPE_CNN_NODE,
        "CMAPSS/train_FD004.txt",
        "CMAPSS/test_FD004.txt",
        "CMAPSS/RUL_FD004.txt",
        "models/cnn_node/cnn_node.FD004.r{{ repeat_index }}.v3.model",
        "figures/cnn_node/plot_average_cnn_node_FD004.pdf",
        settings=OPTIMIZED_CNN_NODE_SETTINGS,
    )


if __name__ == "__main__":
    main()
