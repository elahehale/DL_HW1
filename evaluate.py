import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(
    model,
    test_loader,
    device,
    dataset,
    print_result=False,
    draw_predicted_vs_true=False,
    overlap_tolerance_factor=0.04,
):
    model.eval()

    preds_scaled = []
    trues_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            pred_scaled = model(X_batch).cpu().numpy()
            true_scaled = y_batch.numpy()

            preds_scaled.append(pred_scaled)
            trues_scaled.append(true_scaled)

    preds_scaled = np.vstack(preds_scaled)
    trues_scaled = np.vstack(trues_scaled)

    scaler = dataset.get_scaler()

    preds_original = scaler.inverse_transform(preds_scaled)
    trues_original = scaler.inverse_transform(trues_scaled)

    # computing MAE and MSE

    mae = mean_absolute_error(trues_original, preds_original)
    mse = mean_squared_error(trues_original, preds_original)

    if print_result:
        print(f"\nModel: {model.model_name()}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")

        # just to see some examples
        print(preds_original[:10])
        print(trues_original[:10])

    if draw_predicted_vs_true:
        value_range = np.ptp(trues_original)
        overlap_tolerance = max(value_range * overlap_tolerance_factor, 1e-8)

        overlap_mask = np.abs(preds_original - trues_original) <= overlap_tolerance
        overlap_values = np.where(
            overlap_mask, (preds_original + trues_original) / 2, np.nan
        )

        plt.figure(figsize=(12, 5))
        plt.plot(trues_original, label="True", color="tab:blue", linewidth=1)
        plt.plot(preds_original, label="Predicted", color="tab:orange", linewidth=1)
        plt.plot(
            overlap_values,
            label=f"Overlap (<= {overlap_tolerance:.4g})",
            color="tab:green",
            linewidth=2,
        )
        plt.legend()
        plt.title(f"{model.model_name()} Prediction vs True Values")
        plt.xlabel("Time step")
        plt.ylabel("Laser value")
        plt.show()

    return mae, mse


def _seq_len_from_checkpoint_name(path):
    import re
    from pathlib import Path

    name = Path(path).name
    m = re.search(r"_(\d+)seq_", name)
    return int(m.group(1)) if m else None


def _build_model_from_state_dict(sd, device, *, nhead=None):
    import re

    from models.tcn_model import LaserTCN
    from models.transformer_model import LaserTimeSeriesTransformer

    if any(k.startswith("tcn.") for k in sd):
        hidden = sd["fc.weight"].shape[1]
        out_dim = sd["fc.weight"].shape[0]
        in_ch = sd["tcn.0.conv1.weight"].shape[1]
        kernel = sd["tcn.0.conv1.weight"].shape[2]
        idxs = [
            int(re.match(r"^tcn\.(\d+)\.", k).group(1))
            for k in sd
            if re.match(r"^tcn\.(\d+)\.", k)
        ]
        num_levels = max(idxs) + 1 if idxs else 1
        model = LaserTCN(
            input_channels=in_ch,
            hidden_channels=hidden,
            kernel_size=kernel,
            num_levels=num_levels,
            dropout=0.0,
            output_size=out_dim,
        )
        return model.to(device)

    if "in_proj.weight" in sd and any(k.startswith("encoder.layers.") for k in sd):
        d_model = sd["in_proj.weight"].shape[0]
        if nhead is None:
            raise ValueError(
                "Transformer ckpt: pass nhead (same as training, e.g. --nhead 4)."
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} not divisible by nhead {nhead}")
        ff = sd["encoder.layers.0.linear1.weight"].shape[0]
        layer_idxs = [
            int(re.match(r"^encoder\.layers\.(\d+)\.", k).group(1))
            for k in sd
            if re.match(r"^encoder\.layers\.(\d+)\.", k)
        ]
        num_layers = max(layer_idxs) + 1 if layer_idxs else 1
        out_dim = sd["fc.weight"].shape[0]
        model = LaserTimeSeriesTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=ff,
            dropout=0.0,
            output_size=out_dim,
        )
        return model.to(device)

    raise ValueError("unknown checkpoint (expected TCN or LaserTimeSeriesTransformer)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from dataset import LaserData

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--seq",
        type=int,
        default=None,
        help="window length; default: parse from filename (*_NNseq_*)",
    )
    p.add_argument("--data", default="data/Xtrain.mat")
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--nhead", type=int, default=None)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path, map_location=device)

    model = _build_model_from_state_dict(state_dict, device, nhead=args.nhead)
    model.load_state_dict(state_dict)

    seq_len = args.seq if args.seq is not None else _seq_len_from_checkpoint_name(ckpt_path)
    if seq_len is None:
        raise SystemExit("need --seq or *_NNseq_* in the checkpoint filename")

    dataset = LaserData(
        args.data,
        split_ratio=args.split,
        sequence_length=seq_len,
        key="Xtrain",
    )
    _, val_loader = dataset.get_loaders(batch_size=args.batch)

    evaluate_model(
        model,
        val_loader,
        device,
        dataset,
        print_result=True,
        draw_predicted_vs_true=not args.no_plot,
    )
