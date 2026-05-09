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
sequence_lengths = [40, 45, 50]
print(f"{sequence_lengths=}")

fc_dropouts = [0, 0.1]
print(f"{fc_dropouts=}")

# ============================ CNN hyperparameters =============================
kernel_sizes = [3, 5]
print(f"{kernel_sizes=}")

num_filters = [32, 64]
print(f"{num_filters=}")

cnn_dropouts = [0, 0.1]
print(f"{cnn_dropouts=}")

# ========================= training hyperparameters ===========================
epochs = [60]
print(f"{epochs=}")

optimizers = [AdamW, RMSprop]
print(f"{optimizers=}")

learning_rates = [1e-3]
print(f"{learning_rates=}")

weight_decays = [1e-4, 1e-5]
print(f"{weight_decays=}")

scheduler_factors = [0.5]  # 1 is equivalent to no scheduler
print(f"{scheduler_factors=}")

scheduler_patiences = [5]
print(f"{scheduler_patiences=}")

scalers =[StandardScaler]
print(f"{scalers=}")
################################ grid search ##################################

set_seed(42)

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

cnn_count = 0
cnn_total = common_total * len(cnn_dropouts) * len(kernel_sizes) * len(num_filters)

total_runs = cnn_total
print(f"Total model trainings: {total_runs}")


def total_count():
    return cnn_count


best_cnn_mse = float("inf")
best_cnn_model = None
best_cnn_model_name = ""

start_time = time.time()


def print_progress():
    elapsed_seconds = time.time() - start_time
    remaining_seconds = estimate_remaining_time(
        elapsed_seconds, total_count(), total_runs
    )

    print(
        f"Best CNN MSE: {best_cnn_mse:.4f}"
    )
    print(
        f"Progress: {total_count()}/{total_runs} | Elapsed: {format_seconds(elapsed_seconds)} | Remaining: {format_seconds(remaining_seconds)}"
    )


for seq_len in sequence_lengths:
    for scaler in scalers:

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

            # CNN hyperparameters
            for cnn_dropout, kernel_size, num_filter in product(
                cnn_dropouts, kernel_sizes, num_filters
            ):
                set_seed(42)
                dataset = LaserData("data/Xtrain.mat", sequence_length=seq_len, scaler=scaler())
                train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)

                set_seed(42)
                model = LaserCNN(
                    seq_length=seq_len,
                    num_filters=num_filter,
                    kernel_size=kernel_size,
                    dropout=cnn_dropout,
                    fc_dropout=fc_dropout,
                ).to(device)
                cnn_count += 1
                model_name = f"CNN_seq{seq_len}_cnndrop{cnn_dropout}_fcdrop{fc_dropout}_ep{epoch_count}_opt{optimizer_name}_scale{scaler.__name__}_lr{learning_rate}_wd{weight_decay}_kern{kernel_size}_filt{num_filter}_schedf{scheduler_factor}_schedp{scheduler_patience}"
                print(
                    f"{'='*100}\nTraining CNN model ({cnn_count}/{cnn_total}) {model_name}"
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

                if mse < best_cnn_mse:
                    best_cnn_mse = mse
                    best_cnn_model = model
                    best_cnn_model_name = model_name
                print_progress()

print(f"Best CNN model: {best_cnn_model_name}, MSE: {best_cnn_mse:.4f}")
best_cnn_model_name += f"_mse{best_cnn_mse:.4f}"
save_model(best_cnn_model, best_cnn_model_name)
