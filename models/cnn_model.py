import torch.nn as nn
from models.base_model import LaserBaseModule

class LaserCNN(LaserBaseModule):
    def __init__(self, seq_length=20, input_channels=1,  
                 output_size=1, num_filters=32, kernel_size=3):
        super().__init__()

        self.seq_length = seq_length
        self.input_channels = input_channels
        self.output_size = output_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size


        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=num_filters,
                kernel_size=kernel_size
            ),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters*2, kernel_size),
            nn.Flatten()
        )
        conv_output_length = seq_length - 2*(kernel_size -1)

        self.fc = nn.Sequential(
            nn.Linear((num_filters*2)* conv_output_length, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # needed: [batch_size, 1 seq_length]
        x = x.permute(0,2,1)
        x = self.conv(x)
        y = self.fc(x)
        return y