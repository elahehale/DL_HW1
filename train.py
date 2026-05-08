import torch
import os
from models.base_model import LaserBaseModule
from util import clear_current_line
import matplotlib.pyplot as plt


def plot_loss(losses, type, model, save_name):
    print("save plots")
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label=f"{type} Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model.model_name()} Loss Graph")

    plt.legend()
    plt.grid(True)

    os.makedirs("out/plots", exist_ok=True)

    plot_path = f"out/plots/{save_name}_{type}_Loss.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def train_model(
    epochs,
    model: LaserBaseModule,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    scheduler=None,
    version="",
    save_model=False,
    plot_losses = False
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()

            model.clip_gradients()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        if val_loader is not None:
            model.eval()

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(
                f"\rEpoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f}",
                end="",
                flush=True,
            )
        else:
            print(
                f"\rEpoch [{epoch + 1}/{epochs}] " f"Train Loss: {train_loss:.6f}",
                end="",
                flush=True,
            )

        if scheduler is not None:
            if val_loader is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step(train_loss)

    print()

    X_train, _ = train_loader.dataset.tensors
    seq_len = X_train.shape[1]
    model_name = model.model_name()
    opt_label = (
        f"{optimizer.__class__.__name__}_lr{optimizer.param_groups[0]['lr']:.5f}_"
    )
    
    config = f"{epochs}epochs_{seq_len}seq_{opt_label}_"
    save_name = f"{model_name}{config}{version}"
    save_path = f"out/models/{save_name}.pth"
    
    if plot_losses:
        plot_loss(train_losses, "Train", model, save_name)
        plot_loss(val_losses, "Val", model, save_name)

    if save_model:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"{version} model saved.")

    return model
