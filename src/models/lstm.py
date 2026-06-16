import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_dimension: int = 24,
        lstm_layer_1_hidden_dimension: int = 50,
        lstm_layer_2_hidden_dimension: int = 25,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.lstm_layer_1 = nn.LSTM(
            input_size=input_dimension,
            hidden_size=lstm_layer_1_hidden_dimension,
            batch_first=True,
        )

        self.dropout_1 = nn.Dropout(dropout)

        self.lstm_layer_2 = nn.LSTM(
            input_size=lstm_layer_1_hidden_dimension,
            hidden_size=lstm_layer_2_hidden_dimension,
            batch_first=True,
        )
        self.dropout_2 = nn.Dropout(dropout)

        self.regressor = nn.Linear(lstm_layer_2_hidden_dimension, 1)

    def forward(self, x):
        lstm_1_out, _ = self.lstm_layer_1(x)
        lstm_1_out = self.dropout_1(lstm_1_out)

        _, (hidden_state, _) = self.lstm_layer_2(lstm_1_out)
        last_hidden_state = self.dropout_2(hidden_state[-1])

        prediction = self.regressor(last_hidden_state)

        return prediction.squeeze(-1)
