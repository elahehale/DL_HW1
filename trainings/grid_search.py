import torch
import os
from torch import nn
from dataset import LaserData
from models.cnn_lstm_model import LaserCNNLSTM
from models.gru_model import LaserGRU
from models.lstm_model import LaserLSTM
from itertools import product
from models.cnn_model import LaserCNN
from train import train_model
from evaluate import evaluate_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import save_model

############################ Hyperparameter options ###########################

# ======================= common model hyperparameters =========================
sequence_lengths = [25, 50]
print(f"{sequence_lengths=}")

dropouts = [0, 0.1, 0.3]
print(f"{dropouts=}")

# fc_dropouts = [0, 0.1, 0.2, 0.3]
# print(f"{fc_dropouts=}")

##========================= gru & lstm hyperparameters ========================
hidden_sizes = [16, 32]
print(f"{hidden_sizes=}")

num_layers = [1, 2, 3]
print(f"{num_layers=}")

# layernorm_options = [True, False]
# print(f"{layernorm_options=}")

# ============================ CNN hyperparameters =============================
kernel_sizes = [3, 5]
print(f"{kernel_sizes=}")

num_filters = [16, 32]
print(f"{num_filters=}")

# ========================= training hyperparameters ===========================
# epochs = [40, 60]
# print(f"{epochs=}")

# optimizers = ["Adam", "AdamW"]
# print(f"{optimizers=}")

# learning_rates = [1e-4, 5e-4, 1e-3]
# print(f"{learning_rates=}")

# weight_decays = [0, 1e-6, 1e-4]
# print(f"{weight_decays=}")

scheduler_factors = [0.5, 0.75, 1]  # 1 is equivalent to no scheduler
print(f"{scheduler_factors=}")

# scheduler_patiences = [5, 7]
# print(f"{scheduler_patiences=}")

print(f"Total combinations: {
        len(sequence_lengths) 
        * len(dropouts) 
        # * len(fc_dropouts) 
        * len(hidden_sizes) 
        * len(num_layers) 
        # * len(layernorm_options) 
        * len(kernel_sizes) * len(num_filters) 
        # * len(epochs) 
        # * len(optimizers) 
        # * len(learning_rates) 
        # * len(weight_decays) 
        * len(scheduler_factors) 
        # * len(scheduler_patiences)
        }")

################################ grid search ##################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 60
criterion = nn.MSELoss()

common_total = (
    len(sequence_lengths)
    * len(dropouts)
    * len(scheduler_factors)
    # * len(scheduler_patiences)
)

cnn_count = 0
cnn_total = common_total * len(kernel_sizes) * len(num_filters)

lstm_count = 0
lstm_total = common_total * len(hidden_sizes) * len(num_layers)

gru_count = 0
gru_total = common_total * len(hidden_sizes) * len(num_layers)

cnn_lstm_count = 0
cnn_lstm_total = (
    common_total
    * len(hidden_sizes)
    * len(num_layers)
    * len(kernel_sizes)
    * len(num_filters)
)

best_cnn_mse = float("inf")
best_cnn_model = None
best_cnn_model_name = ""

best_lstm_mse = float("inf")
best_lstm_model = None
best_lstm_model_name = ""

best_gru_mse = float("inf")
best_gru_model = None
best_gru_model_name = ""

best_cnn_lstm_mse = float("inf")
best_cnn_lstm_model = None
best_cnn_lstm_model_name = ""

for seq_len in sequence_lengths:
    dataset = LaserData("data/Xtrain.mat", sequence_length=seq_len)
    train_loader, val_loader = dataset.get_loaders(batch_size=32)

    # common hyperparameters
    for (
        dropout,
        scheduler_factor,
        # scheduler_patience,
    ) in product(
        dropouts,
        scheduler_factors,
        # scheduler_patiences,
    ):
        # CNN hyperparameters
        for kernel_size, num_filter in product(kernel_sizes, num_filters):
            model = LaserCNN(
                seq_length=seq_len,
                num_filters=num_filter,
                kernel_size=kernel_size,
                dropout=dropout,
                fc_dropout=dropout,
            ).to(device)
            cnn_count += 1
            model_name = f"CNN_seq{seq_len}_drop{dropout}_kern{kernel_size}_filt{num_filter}_schedf{scheduler_factor}"
            print(f"Training CNN model ({cnn_count}/{cnn_total}) {model_name}")

            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = (
                ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=5,
                )
                if scheduler_factor < 1
                else None
            )

            model = train_model(
                epochs,
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                device,
                scheduler=scheduler,
                clip_grad_norm=None,
                save_model=False,
            )

            mae, mse = evaluate_model(model, val_loader, device, dataset)
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

            if mse < best_cnn_mse:
                best_cnn_mse = mse
                best_cnn_model = model
                best_cnn_model_name = model_name

        # lstm hyperparameters
        for hidden_size, num_layer in product(hidden_sizes, num_layers):
            model = LaserLSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layer,
                output_size=1,
                dropout=dropout,
                fc_dropout=dropout,
                layer_norm=True,
            ).to(device)
            lstm_count += 1
            model_name = f"LSTM_seq{seq_len}_drop{dropout}_hid{hidden_size}_layers{num_layer}_schedf{scheduler_factor}"
            print(f"Training LSTM model ({lstm_count}/{lstm_total}) {model_name}")

            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = (
                ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=5,
                )
                if scheduler_factor < 1
                else None
            )

            model = train_model(
                epochs,
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                device,
                scheduler=scheduler,
                clip_grad_norm=1.0,
                save_model=False,
            )

            mae, mse = evaluate_model(model, val_loader, device, dataset)
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

            if mse < best_lstm_mse:
                best_lstm_mse = mse
                best_lstm_model = model
                best_lstm_model_name = model_name

        # gru hyperparameters
        for hidden_size, num_layer in product(hidden_sizes, num_layers):
            model = LaserGRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layer,
                output_size=1,
                dropout=dropout,
                fc_dropout=dropout,
                layer_norm=True,
            ).to(device)
            gru_count += 1
            model_name = f"GRU_seq{seq_len}_drop{dropout}_hid{hidden_size}_layers{num_layer}_schedf{scheduler_factor}"
            print(f"Training GRU model ({gru_count}/{gru_total}) {model_name}")

            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = (
                ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=5,
                )
                if scheduler_factor < 1
                else None
            )

            model = train_model(
                epochs,
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                device,
                scheduler=scheduler,
                clip_grad_norm=1.0,
                save_model=False,
            )

            mae, mse = evaluate_model(model, val_loader, device, dataset)
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

            if mse < best_gru_mse:
                best_gru_mse = mse
                best_gru_model = model
                best_gru_model_name = model_name

        # cnn-lstm hyperparameters
        for hidden_size, num_layer, kernel_size, num_filter in product(
            hidden_sizes, num_layers, kernel_sizes, num_filters
        ):
            model = LaserCNNLSTM(
                seq_length=seq_len,
                input_channels=1,
                hidden_size=hidden_size,
                num_layers=num_layer,
                output_size=1,
                num_filters=num_filter,
                kernel_size=kernel_size,
                dropout=dropout,
                fc_dropout=dropout,
            ).to(device)
            cnn_lstm_count += 1
            model_name = f"CNNLSTM_seq{seq_len}_drop{dropout}_hid{hidden_size}_layers{num_layer}_kernel{kernel_size}_filters{num_filter}_schedf{scheduler_factor}"
            print(
                f"Training CNNLSTM model ({cnn_lstm_count}/{cnn_lstm_total}) {model_name}"
            )

            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = (
                ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=5,
                )
                if scheduler_factor < 1
                else None
            )

            model = train_model(
                epochs,
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                device,
                scheduler=scheduler,
                clip_grad_norm=1.0,
                save_model=False,
            )

            mae, mse = evaluate_model(model, val_loader, device, dataset)
            print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

            if mse < best_cnn_lstm_mse:
                best_cnn_lstm_mse = mse
                best_cnn_lstm_model = model
                best_cnn_lstm_model_name = model_name


print(f"Best CNN model: {best_cnn_model_name}, MSE: {best_cnn_mse:.4f}")
best_cnn_model_name += f"_mse{best_cnn_mse:.4f}"
save_model(best_cnn_model, best_cnn_model_name)

print(f"Best LSTM model: {best_lstm_model_name}, MSE: {best_lstm_mse:.4f}")
best_lstm_model_name += f"_mse{best_lstm_mse:.4f}"
save_model(best_lstm_model, best_lstm_model_name)

print(f"Best GRU model: {best_gru_model_name}, MSE: {best_gru_mse:.4f}")
best_gru_model_name += f"_mse{best_gru_mse:.4f}"
save_model(best_gru_model, best_gru_model_name)

print(f"Best CNNLSTM model: {best_cnn_lstm_model_name}, MSE: {best_cnn_lstm_mse:.4f}")
best_cnn_lstm_model_name += f"_mse{best_cnn_lstm_mse:.4f}"
save_model(best_cnn_lstm_model, best_cnn_lstm_model_name)
