import torch
import time
from torch import nn
from dataset import LaserData
from models.cnn_lstm_model import LaserCNNLSTM
from models.gru_model import LaserGRU
from models.lstm_model import LaserLSTM
from itertools import product
from models.cnn_model import LaserCNN
from train import train_model
from evaluate import evaluate_model
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

############################ Hyperparameter options ###########################

# ======================= common model hyperparameters =========================
sequence_lengths = [25, 30, 35]
print(f"{sequence_lengths=}")

fc_dropouts = [0, 0.1]
print(f"{fc_dropouts=}")

##========================= gru & lstm hyperparameters ========================
hidden_sizes = [32, 64, 128]
print(f"{hidden_sizes=}")

num_layers = [3, 4, 5]
print(f"{num_layers=}")

layernorm_options = [False]
print(f"{layernorm_options=}")

lstm_dropouts = [0, 0.1]
print(f"{lstm_dropouts=}")

# ========================= training hyperparameters ===========================
epochs = [30]
print(f"{epochs=}")

optimizers = [AdamW, RMSprop]
print(f"{optimizers=}")

learning_rates = [5e-4]
print(f"{learning_rates=}")

weight_decays = [1e-5]
print(f"{weight_decays=}")

scheduler_factors = [0.5, 0.75]  # 1 is equivalent to no scheduler
print(f"{scheduler_factors=}")

scheduler_patiences = [5]
print(f"{scheduler_patiences=}")

scalers =[StandardScaler]
print(f"{scalers=}")

################################ grid search ##################################

set_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
batch_size = 32

common_total = (
    len(sequence_lengths)
    * len(fc_dropouts)
    * len(epochs)
    * len(optimizers)
    * len(learning_rates)
    * len(weight_decays)
    * len(scheduler_factors)
    * len(scheduler_patiences)
    * len(scalers)

)

lstm_count = 0
lstm_total = (
    common_total
    * len(lstm_dropouts)
    * len(hidden_sizes)
    * len(num_layers)
    * len(layernorm_options)
)


total_runs = lstm_total
print(f"Total model trainings: {total_runs}")


def total_count():
    return lstm_count


best_lstm_mse = float("inf")
best_lstm_model = None
best_lstm_model_name = ""

start_time = time.time()


def print_progress():
    elapsed_seconds = time.time() - start_time
    remaining_seconds = estimate_remaining_time(
        elapsed_seconds, total_count(), total_runs
    )

    print(f"Best LSTM MSE: {best_lstm_mse:.4f}")
    print(
        f"Progress: {total_count()}/{total_runs} | Elapsed: {format_seconds(elapsed_seconds)} | Remaining: {format_seconds(remaining_seconds)}"
    )


for seq_len in sequence_lengths:
    for scaler in scalers:
        set_seed(100)
        dataset = LaserData("data/Xtrain.mat", sequence_length=seq_len, scaler=scaler())
        train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)

        # common hyperparameters
        for (
            fc_dropout,
            epoch_count,
            optimizer_cls,
            learning_rate,
            weight_decay,
            scheduler_factor,
            scheduler_patience,
        ) in product(
            fc_dropouts,
            epochs,
            optimizers,
            learning_rates,
            weight_decays,
            scheduler_factors,
            scheduler_patiences,
        ):
            optimizer_name = optimizer_cls.__name__

            # lstm hyperparameters
            for lstm_dropout, hidden_size, num_layer, layer_norm in product(
                lstm_dropouts, hidden_sizes, num_layers, layernorm_options
            ):
                set_seed(100)
                model = LaserLSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layer,
                    output_size=1,
                    dropout=lstm_dropout,
                    fc_dropout=fc_dropout,
                    layer_norm=layer_norm,
                ).to(device)
                lstm_count += 1
                model_name = f"LSTM_seq{seq_len}_lstmdrop{lstm_dropout}_fcdrop{fc_dropout}_ln{layer_norm}_ep{epoch_count}_opt{optimizer_name}_scale{scaler.__name__}_lr{learning_rate}_wd{weight_decay}_hid{hidden_size}_layers{num_layer}_schedf{scheduler_factor}_schedp{scheduler_patience}"
                print(
                    f"{'='*100}\nTraining LSTM model ({lstm_count}/{lstm_total}) {model_name}"
                )

                optimizer = optimizer_cls(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                scheduler = (
                    ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=scheduler_factor,
                        patience=scheduler_patience,
                    )
                    if scheduler_factor < 1
                    else None
                )

                model = train_model(
                    epoch_count,
                    model,
                    optimizer,
                    criterion,
                    train_loader,
                    val_loader,
                    device,
                    scheduler=scheduler,
                    save_model=False,
                )

                mae, mse = evaluate_model(model, val_loader, device, dataset)
                print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")


                if mse < best_lstm_mse:
                    best_lstm_mse = mse
                    best_lstm_model = model
                    best_lstm_model_name = model_name

                print_progress()


print(f"Best LSTM model: {best_lstm_model_name}, MSE: {best_lstm_mse:.4f}")
best_lstm_model_name += f"_mse{best_lstm_mse:.4f}"
save_model(best_lstm_model, best_lstm_model_name)
