import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, test_loader, device, dataset, draw_predicted_vs_true=True):
    model.eval()

    with torch.no_grad():
        X_batch, y_batch = next(iter(test_loader))
        X_batch = X_batch.to(device)

        pred_scaled = model(X_batch).cpu().numpy()
        true_scaled = y_batch.numpy()

    scaler = dataset.get_scaler()

    pred_original = scaler.inverse_transform(pred_scaled)
    true_original = scaler.inverse_transform(true_scaled)

    # computing MAE and MSE

    mae = mean_absolute_error(true_original,pred_original)
    mse = mean_squared_error(true_original,pred_original)
    print(f"\nModel: {model.model_name()}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    # just to see some examples
    print(pred_original[:10])
    print(true_original[:10])

    if draw_predicted_vs_true:
        draw_predicted_vs_true_plot(model, test_loader, dataset, device)
    
    return mae, mse


def draw_predicted_vs_true_plot(model, loader, dataset, device, overlap_tolerance=None):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            y_pred = model(X_batch).cpu().numpy()

            preds.append(y_pred)
            trues.append(y_batch.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    # optional: back to original scale
    preds_original = dataset.get_scaler().inverse_transform(preds)
    trues_original = dataset.get_scaler().inverse_transform(trues)

    if overlap_tolerance is None:
        value_range = np.ptp(trues_original)
        overlap_tolerance = max(value_range * 0.04, 1e-8)

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
