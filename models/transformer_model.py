import math

import torch
import torch.nn as nn

from models.base_model import LaserBaseModule


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        n = x.size(1)
        return x + self.pe[:, :n, :].to(dtype=x.dtype, device=x.device)


class LaserTimeSeriesTransformer(LaserBaseModule):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_size=1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must divide nhead")
        self.hidden_size = d_model
        self.num_layers = num_encoder_layers

        self.in_proj = nn.Linear(1, d_model)
        self.pe = SinusoidalPE(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_encoder_layers)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = self.in_proj(x)
        x = self.pe(x)

        seq = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(
            seq, device=x.device, dtype=torch.float32
        )
        x = self.encoder(x, mask=causal)

        x = self.drop(x[:, -1, :])
        return self.fc(x)
