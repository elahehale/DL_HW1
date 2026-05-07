import sys
from pathlib import Path

_train = Path(__file__).resolve().parent
_root = _train.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_train))

import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error

from bo_gp import gp_ei_search, to_int
from dataset import LaserData
from models.transformer_model import LaserTimeSeriesTransformer
from train import train_model

BO_CALLS = 28
BO_SEED = 1
EPOCHS_PER_TRIAL = 22
DATA_PATH = "data/Xtrain.mat"
SPLIT = 0.8
BATCH = 32
CLIP = None

# (d_model, nhead) only pairs that divide cleanly
PAIR_IDX = [(32, 4), (48, 4), (64, 4), (64, 8), (96, 4), (128, 8)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


def val_mse(model, val_loader, dataset):
    model.eval()
    preds_scaled = []
    trues_scaled = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred_scaled = model(xb).cpu().numpy()
            true_scaled = yb.numpy()
            preds_scaled.append(pred_scaled)
            trues_scaled.append(true_scaled)

    if not preds_scaled:
        return float("inf")

    preds_scaled = np.vstack(preds_scaled)
    trues_scaled = np.vstack(trues_scaled)
    scaler = dataset.get_scaler()
    preds_original = scaler.inverse_transform(preds_scaled)
    trues_original = scaler.inverse_transform(trues_scaled)
    return mean_squared_error(trues_original, preds_original)


def vec_to_cfg(u):
    u = np.clip(u, 0.0, 1.0)
    seq = to_int(u[0], 18, 56)
    lr = float(10 ** (-4.0 + 2.2 * u[1]))
    wd = float(10 ** (-6.0 + 3.5 * u[2]))
    j = min(len(PAIR_IDX) - 1, int(np.floor(u[3] * len(PAIR_IDX))))
    d_model, nhead = PAIR_IDX[j]
    layers = to_int(u[4], 1, 4)
    ff_mult = 2.0 + 2.0 * u[5]
    ff = int(max(4 * d_model, round(d_model * ff_mult)))
    drop = float(0.05 + 0.25 * u[6])
    return {
        "seq": seq,
        "lr": lr,
        "wd": wd,
        "d_model": d_model,
        "nhead": nhead,
        "layers": layers,
        "ff": ff,
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
        model = LaserTimeSeriesTransformer(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_encoder_layers=cfg["layers"],
            dim_feedforward=cfg["ff"],
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
    loss = val_mse(model, va, ds)
    print(
        f" trial val_mse={loss:.5f} | seq={cfg['seq']} d={cfg['d_model']} "
        f"h={cfg['nhead']} L={cfg['layers']} ff={cfg['ff']} drop={cfg['drop']:.2f}"
    )
    return loss


if __name__ == "__main__":
    dim = 7
    print("bayes opt transformer (GP + EI), dim=", dim)
    best_u, best_score, _, _ = gp_ei_search(
        run_trial, dim, n_calls=BO_CALLS, seed=BO_SEED
    )
    best = vec_to_cfg(best_u)
    print("best val mse (cheap epochs):", best_score)
    print("best cfg:", best)

    EPOCHS_FINAL = 60
    ds = LaserData(
        DATA_PATH,
        split_ratio=SPLIT,
        sequence_length=best["seq"],
        key="Xtrain",
    )
    tr, va = ds.get_loaders(batch_size=BATCH)
    model = LaserTimeSeriesTransformer(
        d_model=best["d_model"],
        nhead=best["nhead"],
        num_encoder_layers=best["layers"],
        dim_feedforward=best["ff"],
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
    print("final val mse:", val_mse(model, va, ds))
