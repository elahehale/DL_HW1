import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from models.base_model import LaserBaseModule


class LaserGRU(LaserBaseModule):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        output_size=1,
        dropout=0.2,
        fc_dropout=0.1,
        layer_norm=False,
        grad_clip_norm=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.grad_clip_norm = grad_clip_norm

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size) if layer_norm else None
        self.fc_dropout_layer = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # output: the hidden states of the last GRU layer at all time steps.
        # h_n: the hidden states of each GRU layer at the final time step.
        output, h_n = self.gru(x)
        x = output[:, -1, :]
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.fc_dropout_layer(x)
        y = self.fc(x)
        return y

    def clip_gradients(self):
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self.gru.parameters(), max_norm=self.grad_clip_norm)
