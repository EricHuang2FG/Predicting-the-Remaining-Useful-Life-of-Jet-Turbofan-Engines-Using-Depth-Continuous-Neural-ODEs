import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.node import NeuralODE


class CNN_ODE(nn.Module):
    """CNN-NODE model architecture for remaining useful life (RUL) prediction.

    This model combines a one-dimensional convolutional neural network (CNN) with
    a Neural Ordinary Differential Equation (NODE) solved numerically using the
    torchdiffeq library with the Dopri-5 method. The output of the model is a single integer
    value that represents the the predicted RUL of the input engine sensor reading time-series.
    """

    def __init__(
        self,
        input_dimension: int = 24,
        cnn_num_kernals: int = 36,  # 36 convolution filters
        cnn_kernal_size: int = 3,  # sliding window analyzing 3 time steps at once
        cnn_stride: int = 1,  # slides window by 1 element each time, no skipping
        cnn_padding: int = 1,  # pad both sides with 0 so maintain sequence length
        hidden_dimension: int = 64,
        encoder_dimension: int = 128,
        regressor_dimension: int = 32,
        dropout: float = 0.2,
        sequence_length: int = 40,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            # create 1D convolution layer
            nn.Conv1d(
                in_channels=input_dimension,  # number of features per time step to analyze
                out_channels=cnn_num_kernals,  # number of out channels corresponds with number of filters
                kernel_size=cnn_kernal_size,
                stride=cnn_stride,
                padding=cnn_padding,
            ),
            nn.SiLU(),  # sigmoid/logistic function inctrouduces nonlinearity
            nn.Dropout(dropout),
        )

        # transforms the CNN convolution results into one long vector theta_0
        # used by the Neural ODE as the initial state
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * cnn_num_kernals, encoder_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dimension, hidden_dimension),
        )

        self.ode = NeuralODE(hidden_dimension)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dimension, regressor_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_dimension, 1),
        )

    def forward(self, x):
        # get initial state
        t_span = torch.tensor([0.0, 1.0], device=x.device)

        cnn_in = x.transpose(1, 2)  # swap second and third dimensions
        cnn_out = self.cnn(cnn_in)
        theta_0 = self.encoder(cnn_out)
        # numerically integrates the derivative to get the predicted value
        theta_final = odeint(self.ode, theta_0, t_span, method="dopri5")[-1]
        prediction = self.regressor(theta_final).squeeze(-1)

        return prediction
