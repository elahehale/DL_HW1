import sys
from pathlib import Path

_train = Path(__file__).resolve().parent
_root = _train.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_train))

import numpy as np
import torch
from torch import nn

from bo_gp import gp_ei_search, to_int
from dataset import LaserData
from models.tcn_model import LaserTCN
from train import train_model

# ---- BO / training budget (crank up for real runs) ----
BO_CALLS = 28
BO_SEED = 0
EPOCHS_PER_TRIAL = 22  # short runs for search; bump for final retrain
DATA_PATH = "data/Xtrain.mat"
SPLIT = 0.8
BATCH = 32
CLIP = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


def val_mse(model, val_loader):
    model.eval()
    s = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            s += criterion(model(xb), yb).item()
    return s / max(len(val_loader), 1)


def vec_to_cfg(u):
    u = np.clip(u, 0.0, 1.0)
    seq = to_int(u[0], 18, 56)
    lr = float(10 ** (-4.0 + 2.2 * u[1]))
    wd = float(10 ** (-6.0 + 3.5 * u[2]))
    hidden = to_int(u[3], 32, 96)
    hidden = 16 * max(2, round(hidden / 16))
    k = 3 if u[4] < 0.65 else 5
    levels = to_int(u[5], 2, 6)
    drop = float(0.05 + 0.35 * u[6])
    return {
        "seq": seq,
        "lr": lr,
        "wd": wd,
        "hidden": hidden,
        "k": k,
        "levels": levels,
        "drop": drop,
    }


def run_trial(u):
    cfg = vec_to_cfg(u)
    try:
        ds = LaserData(
            DATA_PATH,
            split_ratio=SPLIT,
            sequence_length=cfg["seq"],
            key="Xtrain",
        )
        tr, va = ds.get_loaders(batch_size=BATCH)
    except Exception:
        return 1e6

    try:
        model = LaserTCN(
            input_channels=1,
            hidden_channels=cfg["hidden"],
            kernel_size=cfg["k"],
            num_levels=cfg["levels"],
            dropout=cfg["drop"],
            output_size=1,
        ).to(device)
    except Exception:
        return 1e6

    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]
    )
    train_model(
        EPOCHS_PER_TRIAL,
        model,
        opt,
        criterion,
        tr,
        va,
        device,
        clip_grad_norm=CLIP,
        save_model=False,
    )
    loss = val_mse(model, va)
    print(
        f" trial val_mse={loss:.5f} | seq={cfg['seq']} hidden={cfg['hidden']} "
        f"k={cfg['k']} L={cfg['levels']} drop={cfg['drop']:.2f} lr={cfg['lr']:.2e}"
    )
    return loss


if __name__ == "__main__":
    dim = 7
    print("bayes opt TCN (GP + EI), dim=", dim)
    best_u, best_score, _, _ = gp_ei_search(
        run_trial, dim, n_calls=BO_CALLS, seed=BO_SEED
    )
    best = vec_to_cfg(best_u)
    print("best val mse (cheap epochs):", best_score)
    print("best cfg:", best)

    print("--- retrain best a bit longer (edit EPOCHS_FINAL) ---")
    EPOCHS_FINAL = 60
    ds = LaserData(
        DATA_PATH,
        split_ratio=SPLIT,
        sequence_length=best["seq"],
        key="Xtrain",
    )
    tr, va = ds.get_loaders(batch_size=BATCH)
    model = LaserTCN(
        input_channels=1,
        hidden_channels=best["hidden"],
        kernel_size=best["k"],
        num_levels=best["levels"],
        dropout=best["drop"],
        output_size=1,
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=best["lr"], weight_decay=best["wd"]
    )
    train_model(
        EPOCHS_FINAL,
        model,
        opt,
        criterion,
        tr,
        va,
        device,
        clip_grad_norm=CLIP,
        version="_bo_best",
        save_model=True,
    )
    print("final val mse:", val_mse(model, va))
