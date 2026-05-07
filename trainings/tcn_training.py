import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch import nn

from dataset import LaserData
from evaluate import evaluate_model
from models.tcn_model import LaserTCN
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

dataset = LaserData(
    "data/Xtrain.mat",
    split_ratio=train_split,
    sequence_length=seq_len,
    key="Xtrain",
)
train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)

hidden_ch = 48
kernel_sz = 3
stack_depth = 4
drop_p = 0.2

model = LaserTCN(
    input_channels=1,
    hidden_channels=hidden_ch,
    kernel_size=kernel_sz,
    num_levels=stack_depth,
    dropout=drop_p,
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
