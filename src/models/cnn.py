import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        input_dimension: int = 24,
        cnn_num_kernals: int = 36,  # 36 convolution filters
        cnn_kernal_size: int = 3,  # sliding window analyzing 3 time steps at once
        cnn_stride: int = 1,  # slides window by 1 element each time, no skipping
        cnn_padding: int = 1,  # pad both sides with 0 so maintain sequence length
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
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_num_kernals * sequence_length, 1),
        )

    def forward(self, x):
        # get initial state
        cnn_in = x.transpose(1, 2)  # swap second and third dimensions
        cnn_out = self.cnn(cnn_in)
        prediction = cnn_out.squeeze(-1)

        return prediction
