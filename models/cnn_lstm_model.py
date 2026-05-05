import torch.nn as nn
from models.base_model import LaserBaseModule


class LaserCNNLSTM(LaserBaseModule):
    def __init__(
        self,
        seq_length=20,
        input_channels=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        num_filters=64,
        kernel_size=3,
        dropout=0.1,
        fc_dropout=0.1,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        conv_len = seq_length - 2 * (kernel_size - 1)
        if conv_len < 1:
            raise ValueError("seq_length too small for this kernel_size")

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=num_filters * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)
