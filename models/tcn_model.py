import torch.nn as nn
import torch.nn.functional as F

from models.base_model import LaserBaseModule


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.proj(x)
        out = F.pad(x, (self.pad, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.drop(out)

        out = F.pad(out, (self.pad, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.drop(out)

        return F.relu(out + res)


class LaserTCN(LaserBaseModule):
    def __init__(
        self,
        input_channels=1,
        hidden_channels=48,
        kernel_size=3,
        num_levels=4,
        dropout=0.2,
        output_size=1,
    ):
        super().__init__()
        self.hidden_size = hidden_channels
        self.num_layers = num_levels

        dilations = [2 ** i for i in range(num_levels)]
        layers = []
        in_ch = input_channels
        for d in dilations:
            layers.append(
                TemporalBlock(in_ch, hidden_channels, kernel_size, d, dropout)
            )
            in_ch = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.drop(x)
        return self.fc(x)
