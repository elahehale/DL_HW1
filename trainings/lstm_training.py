import torch
from torch import nn

from dataset import LaserData
from evaluate import evaluate_model
from models.lstm_model import LaserLSTM
from train import train_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LaserData("data/Xtrain.mat", sequence_length=30)
train_loader, val_loader = dataset.get_loaders(batch_size=32)

model = LaserLSTM(
    input_size=1,
    hidden_size=32,
    num_layers=2,
    output_size=1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 60

train_model(epochs, model, optimizer, criterion, train_loader, val_loader, device)
# this should be test loader
evaluate_model(model, val_loader, device, dataset)
