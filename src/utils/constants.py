DATASET_ID_FD001 = "FD001"
DATASET_ID_FD002 = "FD002"
DATASET_ID_FD003 = "FD003"
DATASET_ID_FD004 = "FD004"

DEFAULT_FIGURE_SIZE: tuple = (12, 9)
LINE_WIDTH: int = 3

NUM_SETTINGS_AND_SENSOR_READINGS: int = 24

MODEL_TYPE_NODE: str = "node"
MODEL_TYPE_CNN_NODE: str = "cnn_node"
MODEL_TYPE_CNN: str = "cnn"
MODEL_TYPE_CNN_LSTM_ATTENTION: str = "cnn_lstm_attention"
MODEL_TYPE_LSTM: str = "lstm"

DIMENSION_TYPE_HIDDEN: str = "hidden_dimension"
DIMENSION_TYPE_ENCODER: str = "encoder_dimension"
DIMENSION_TYPE_REGRESSOR: str = "regressor_dimension"
DIMENSION_TYPE_CNN_NUM_KERNALS: str = "cnn_num_kernals"

DEFAULT_WINDOW_SIZE: int = 40
DEFAULT_NETWORK_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "hidden_dimension": 64,
    "encoder_dimension": 128,
    "regressor_dimension": 32,
    "dropout": 0.2,
    "name": "NODE",
}

OPTIMIZED_NODE_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "hidden_dimension": 128,
    "encoder_dimension": 128,
    "regressor_dimension": 128,
    "dropout": 0.2,
    "name": "NODE",
}

OPTIMIZED_CNN_NODE_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "cnn_num_kernals": 20,
    "cnn_kernal_size": 3,
    "cnn_stride": 1,
    "cnn_padding": 1,
    # "pooling_kernal_size": 3,
    # "pooling_stride": 1,
    "hidden_dimension": 128,
    "encoder_dimension": 128,
    "regressor_dimension": 128,
    "dropout": 0.2,
    "sequence_length": DEFAULT_WINDOW_SIZE,
    "name": "CNN-NODE",
}

# pooling does not help, so we removed pooling here
OPTIMIZED_CNN_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "cnn_num_kernals": 20,
    "cnn_kernal_size": 3,
    "cnn_stride": 1,
    "cnn_padding": 1,
    # "pooling_kernal_size": 3,
    # "pooling_stride": 1,
    "dropout": 0.2,
    "name": "CNN",
}

OPTIMIZED_CNN_LSTM_ATTENTION_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "cnn_num_kernals": 36,
    "cnn_kernal_size": 3,
    "cnn_stride": 1,
    "cnn_padding": 1,
    "pooling_kernal_size": 3,
    "pooling_stride": 1,
    "pooling_padding": 1,
    "lstm_hidden_dimension": 64,
    "lstm_num_layers": 1,
    "attention_dimension": 32,
    "dropout": 0.0,
    "name": "CNN-LSTM-Attention",
}

OPTIMIZED_LSTM_SETTINGS: dict = {
    "batch_size": 128,
    "epochs": 25,
    "lr": 0.001,
    "l2_regularization": 0.01,
    "lstm_layer_1_hidden_dimension": 50,
    "lstm_layer_2_hidden_dimension": 25,
    "dropout": 0.4,
    "name": "LSTM",
}
