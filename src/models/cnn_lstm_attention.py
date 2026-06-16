import torch
import torch.nn as nn


class CNN_LSTM_Attention(nn.Module):
    def __init__(
        self,
        input_dimension: int = 24,
        cnn_num_kernals: int = 36,  # 36 convolution filters
        cnn_kernal_size: int = 3,  # sliding window analyzing 3 time steps at once
        cnn_stride: int = 1,  # slides window by 1 element each time, no skipping
        cnn_padding: int = 1,  # pad both sides with 0 so maintain sequence length
        pooling_kernal_size: int = 3,
        pooling_stride: int = 1,
        pooling_padding: int = 1,
        lstm_hidden_dimension: int = 64,
        lstm_num_layers: int = 1,
        attention_dimension: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dimension,
                out_channels=cnn_num_kernals,
                kernel_size=cnn_kernal_size,
                stride=cnn_stride,
                padding=cnn_padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=pooling_kernal_size,
                stride=pooling_stride,
                padding=pooling_padding,
            ),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_num_kernals,
            hidden_size=lstm_hidden_dimension,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.dk = attention_dimension

        self.q = nn.Linear(lstm_hidden_dimension, attention_dimension)
        self.k = nn.Linear(lstm_hidden_dimension, attention_dimension)
        self.v = nn.Linear(lstm_hidden_dimension, attention_dimension)

        self.regressor = nn.Linear(attention_dimension, 1)

    def forward(self, x):
        cnn_in = x.transpose(1, 2)
        cnn_out = self.cnn(cnn_in)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = self.dropout(lstm_out)

        queries = self.q(lstm_out)
        keys = self.k(lstm_out)
        values = self.v(lstm_out)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (
            self.dk**0.5
        )  # (Q * K^T) / sqrt(dk)
        attention_weights = torch.softmax(
            attention_scores, dim=-1
        )  # F(Q, K) = softmax(Q * K^T / sqrt(dk))
        attention_out = torch.matmul(
            attention_weights, values
        )  # Attention(Q, K, V) = F(Q, K) * V
        attention_out = torch.mean(attention_out, dim=1)
        # attention_out = attention_out[:, -1, :]

        prediction = self.regressor(attention_out).squeeze(-1)

        return prediction
