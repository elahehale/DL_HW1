import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models.base_model import LaserBaseModule


class LaserCNNGRU(LaserBaseModule):
    def __init__(
        self,
        seq_length=20,
        input_channels=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        num_filters=64,
        kernel_size=3,
        cnn_dropout=0,
        lstm_dropout=0.1,
        fc_dropout=0.1,
        layer_norm=False,
        grad_clip_norm=1.0,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.grad_clip_norm = grad_clip_norm

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, num_filters, kernel_size),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout1d(cnn_dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout1d(cnn_dropout),
        )

        self.gru = nn.GRU(
            input_size=num_filters * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size) if layer_norm else None
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # [batch_size, seq_len, channels] => [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        out, _ = self.gru(x)
        out = out[:, -1, :]
        if self.layer_norm:
            out = self.layer_norm(out)
        out = self.dropout(out)
        return self.fc(out)

    def clip_gradients(self):
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self.gru.parameters(), max_norm=self.grad_clip_norm)
