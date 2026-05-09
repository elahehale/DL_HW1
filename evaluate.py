import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(
    model,
    test_loader,
    device,
    dataset,
    print_result=False,
    draw_predicted_vs_true=False,
    overlap_tolerance_factor=0.2,
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
        # value range = max - min
        # value_range = np.ptp(trues_original)
        value_range = np.std(trues_original)
        # calculate tolerance based on the range
        overlap_tolerance = value_range * overlap_tolerance_factor
        # calculate a mask based on whether diff <= tolerance
        overlap_mask = np.abs(preds_original - trues_original) <= overlap_tolerance
        # assign overlap value
        # if within tolerance: midpoint
        # else: nan (wont draw)
        overlap_values = np.where(
            overlap_mask, (preds_original + trues_original) / 2, np.nan
        )

        plt.figure(figsize=(12, 5))
        plt.plot(
            trues_original,
            label="True",
            color="tab:blue",
            marker="o",
            markersize=3,
            linewidth=1,
        )
        plt.plot(
            preds_original,
            label="Predicted",
            color="tab:orange",
            marker="o",
            markersize=3,
            linewidth=1,
        )
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
        # plt.show()
        plt.savefig(os.path.join("out", f"{model.model_name()}_pvst.png"))

    return mae, mse
