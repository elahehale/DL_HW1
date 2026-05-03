import torch
import matplotlib.pyplot as plt
import numpy as np

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

    # just to see some examples
    print(pred_original[:10])
    print(true_original[:10])


    if draw_predicted_vs_true:
        draw_predicted_vs_true_plot(model, test_loader, dataset, device)


def draw_predicted_vs_true_plot(model, loader, dataset, device):
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

    plt.figure(figsize=(12, 5))
    plt.plot(trues_original, label="True")
    plt.plot(preds_original, label="Predicted")
    plt.legend()
    plt.title("LSTM Prediction vs True Values")
    plt.xlabel("Time step")
    plt.ylabel("Laser value")
    plt.show()