import torch
import torch.nn as nn
from torchdiffeq import odeint


# inherits from the nn.Module base class, forms foundational outline of neural network
# this neural network defines the derivative function f(θ,t)
# models how the state θ changes infinitessimally over time
# the hidden layers represent the continuous evolution of state overtime
# continuously updates the weights of the model in order to tune the f(θ,t) function so inputs match ot predicted values
class NeuralODE(nn.Module):

    def __init__(self, dimension: int = 64) -> None:
        # initializing simple feed forward neural network
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, dimension)
        )

        # force some weights at initialization
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, t, theta):
        output = self.linear_stack(
            theta
        )  # passes theta through the sequential stack to determine the derivative at that state
        output = torch.clamp(output, -10.0, 10.0)
        return output


class ODE(nn.Module):
    """NODE model architecture for remaining useful life (RUL) prediction.

    This model defines a Neural Ordinary Differential Equation (NODE) solved numerically using the
    torchdiffeq library with the Dopri-5 method. The model takes in engine sensor reading time-series
    as the input, and outputs a single integer value that represents the the predicted RUL of that engine.
    """

    def __init__(
        self,
        input_dimension: int = 24,
        hidden_dimension: int = 64,
        encoder_dimension: int = 128,
        regressor_dimension: int = 32,
        dropout: float = 0.2,
        sequence_length: int = 40,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dimension * sequence_length, encoder_dimension),
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
        theta_0 = self.encoder(x)

        # Using the DOPRI-5 solver, the ODE solution is numerically integrated over the time span
        # by going through many forward passes of the Neural ODE with adaptive step sizes
        theta_final = odeint(self.ode, theta_0, t_span, method="dopri5")[-1]
        prediction = self.regressor(theta_final).squeeze(-1)

        return prediction
