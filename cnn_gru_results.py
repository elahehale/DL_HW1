from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from dataset import LaserData
from evaluate import evaluate_model
from models.cnn_gru_model import LaserCNNGRU
from models.cnn_lstm_model import LaserCNNLSTM
import torch

from models.cnn_model import LaserCNN
from models.gru_model import LaserGRU
from models.lstm_model import LaserLSTM
from prediction import iterative_predicting_next_points
from train import train_model
from util import set_seed


def reset_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def get_data_loaders(seq, seed=100):
    set_seed(seed)
    dataset_80 = LaserData("data/Xtrain.mat", split_ratio=0.8, sequence_length=seq)
    scaler_80 = dataset_80.get_scaler()
    train_loader_80, val_loader_80 = dataset_80.get_loaders(seed = seed)
    dataset_test_80 = LaserData(
        "data/Xtest.mat",
        split_ratio=0.8,
        sequence_length=seq,
        scaler=scaler_80,
        mode="test",
        key="Xtest",
    )
    test_loader_80 = dataset_test_80.get_loaders(seed = seed)
    set_seed(seed)
    dataset_100 = LaserData("data/Xtrain.mat", split_ratio=1, sequence_length=seq)
    scaler_100 = dataset_100.get_scaler()
    dataset_test_100 = LaserData(
        "data/Xtest.mat",
        split_ratio=1,
        sequence_length=seq,
        scaler=scaler_100,
        mode="test",
        key="Xtest",
    )
    test_loader_100 = dataset_test_100.get_loaders(seed = seed)
    train_loader_100 = dataset_100.get_loaders(seed = seed)
    set_seed(seed)
    dataset_90 = LaserData("data/Xtrain.mat", split_ratio=0.9, sequence_length=seq)
    scaler_90 = dataset_90.get_scaler()
    dataset_test_90 = LaserData(
        "data/Xtest.mat",
        split_ratio=0.9,
        sequence_length=seq,
        scaler=scaler_90,
        mode="test",
        key="Xtest",
    )
    test_loader_90 = dataset_test_90.get_loaders(seed = seed)
    train_loader_90, val_loader_90 = dataset_90.get_loaders(seed = seed)

    return (
        test_loader_80,
        test_loader_90,
        test_loader_100,
        train_loader_80,
        train_loader_90,
        train_loader_100,
        val_loader_80,
        val_loader_90,
        dataset_80,
        dataset_90,
        dataset_100,
    )


models = {
    "CNNGRU": {
        "model": "CNNGRU",
        "seq": 45,
        "cnndrop": 0,
        "lstmdrop": 0,
        "fcdrop": 0,
        "ln": False,
        "ep": 100,
        "opt": AdamW,
        "scale": StandardScaler,
        "lr": 0.001,
        "wd": 0.0001,
        "hid": 64,
        "layers": 3,
        "kernel": 5,
        "filters": 64,
        "schedf": 0.75,
        "schedp": 5,
        "mse": 0.9875,
    },
}

def initialize_cnngru_model(params, device, seed=100):
    set_seed(seed)
    model = LaserCNNGRU(
        seq_length=params["seq"],
        input_channels=1,
        hidden_size=params["hid"],
        num_layers=params["layers"],
        output_size=1,
        num_filters=params["filters"],
        kernel_size=params["kernel"],
        cnn_dropout=params["cnndrop"],
        lstm_dropout=params["lstmdrop"],
        fc_dropout=params["fcdrop"],
        layer_norm=params["ln"],
    ).to(device)
    return model

def initialize_optimizer_scheduler(params, mod):
    optimizer = params["opt"](
        mod.parameters(), lr=params["lr"], weight_decay=params["wd"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params["schedf"],
        patience=params["schedp"],
    )
    return optimizer, scheduler

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CNNGRU_model_path = "out/models/Best/LaserCNNGRU_h64_l3100epochs_45seq_AdamW_lr0.00003.pth"
    TRAIN = True
    TEST = True
    print("========= Evaluate CNN-GRU on Test Set =========")
    seed = 42
    data = models["CNNGRU"]
    (
        test_loader_80,
        test_loader_90,
        test_loader_100,
        train_loader_80,
        train_loader_90,
        train_loader_100,
        val_loader_80,
        val_loader_90,
        dataset_80,
        dataset_90,
        dataset_100,

    ) = get_data_loaders(data["seq"], seed)
    model = initialize_cnngru_model(data, device, seed)
    criterion = nn.MSELoss()
    if TRAIN:
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        epochs = data["ep"]
        print("Start Training CNN-GRU with the best hyperparameters.")
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_80,
            val_loader_80,
            device,
            scheduler=scheduler,
            version=f"{seed}seed_80_data",
            save_model=True,
            plot_losses=True
        )

        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80, draw_predicted_vs_true=True)
        print(f" seed = {seed} - Trained CNNGRU (80/20 train/val) results on test set for the best hyperparams")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

    if TEST:
        state_dict = torch.load(CNNGRU_model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80, draw_predicted_vs_true=True)
        print(f" seed = {seed} - saved model of CNNGRU results on test set")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")
        iterative_predicting_next_points(model, dataset_80, test_loader_80, 200, device)
