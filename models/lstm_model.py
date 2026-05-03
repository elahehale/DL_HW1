import numpy as np
import torch
import torch.nn as nn


class LaserLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)

        # take last time step
        out = out[:, -1, :]

        out = self.fc(out)
        return out
