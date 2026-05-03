import torch.nn as nn

from models.base_model import LaserBaseModule


class LaserGRU(LaserBaseModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # output: the hidden states of the last GRU layer at all time steps.
        # h_n: the hidden states of each GRU layer at the final time step.
        output, h_n = self.gru(x)
        last_hidden = output[:, -1, :]
        y = self.fc(last_hidden)
        return y
