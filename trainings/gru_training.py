import torch
from torch import nn

from dataset import LaserData
from evaluate import evaluate_model
from models.gru_model import LaserGRU
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LaserData("data/Xtrain.mat", sequence_length=20)
train_loader, val_loader = dataset.get_loaders(batch_size=32)

model = LaserGRU(input_size=1, hidden_size=16, num_layers=2, output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

epochs = 60

train_model(
    epochs,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    scheduler=scheduler,
)
# this should be test loader
evaluate_model(model, val_loader, device, dataset)
