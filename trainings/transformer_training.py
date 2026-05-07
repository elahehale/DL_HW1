import sys
from pathlib import Path

# same hack as other scripts in this folder — run from project root
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch import nn

from dataset import LaserData
from evaluate import evaluate_model
from models.transformer_model import LaserTimeSeriesTransformer
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 60
lr = 1e-3
weight_decay = 0.0
seq_len = 30
train_split = 0.8
clip_norm = None

save_weights = True
run_tag = ""

d_model = 64
nhead = 4
enc_layers = 2
ff_dim = 128
attn_dropout = 0.1

dataset = LaserData(
    "data/Xtrain.mat",
    split_ratio=train_split,
    sequence_length=seq_len,
    key="Xtrain",
)
train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)

model = LaserTimeSeriesTransformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=enc_layers,
    dim_feedforward=ff_dim,
    dropout=attn_dropout,
    output_size=1,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, weight_decay=weight_decay
)

train_model(
    epochs,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    clip_grad_norm=clip_norm,
    version=run_tag,
    save_model=save_weights,
)

evaluate_model(model, val_loader, device, dataset)
