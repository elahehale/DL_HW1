import torch
from torch import nn

from dataset import LaserData
from evaluate import evaluate_model
from models.cnn_model import LaserCNN
from train import train_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LaserData("data/Xtrain.mat", sequence_length=20)
train_loader, val_loader = dataset.get_loaders(batch_size=32)

model = LaserCNN(seq_length=20, input_channels=1, output_size=1,
                num_filters=32, kernel_size=3).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

epochs = 60

train_model(epochs, model, optimizer, criterion, train_loader, val_loader, 
            device, scheduler=None, clip_grad_norm=None, version="cnn_", save_model=True )
  

evaluate_model(model, val_loader, device, dataset)
