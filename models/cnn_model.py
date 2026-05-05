import torch.nn as nn
from models.base_model import LaserBaseModule


class LaserCNN(LaserBaseModule):
    def __init__(
        self,
        seq_length=20,
        input_channels=1,
        output_size=1,
        num_filters=32,
        kernel_size=3,
        linear_size=32,
        dropout=0.1,
        fc_dropout=0.1,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.input_channels = input_channels
        self.output_size = output_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.linear_size = linear_size
        self.dropout = dropout
        self.fc_dropout = fc_dropout

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Flatten(),
        )

        self.fc_dropout_layer = nn.Dropout(fc_dropout)

        conv_output_length = seq_length - 2 * (kernel_size - 1)
        self.fc = nn.Sequential(
            nn.Linear((num_filters * 2) * conv_output_length, self.linear_size),
            nn.BatchNorm1d(self.linear_size),
            nn.ReLU(),
            nn.Dropout1d(fc_dropout),
            nn.Linear(self.linear_size, output_size),
        )

    def forward(self, x):
        # needed: [batch_size, 1 seq_length]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.fc_dropout_layer(x)
        y = self.fc(x)
        return y
