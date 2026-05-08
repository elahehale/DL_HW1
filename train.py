import torch
import os
from torch.nn.utils import clip_grad_norm_
from models.base_model import LaserBaseModule
from util import clear_current_line


def train_model(
    epochs,
    model: LaserBaseModule,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    scheduler=None,
    clip_grad_norm=None,
    version="",
    save_model=False,
):

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()

            if clip_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)

                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"\rEpoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f}",
                end="",
                flush=True,
            )
        else:
            print(
                f"\rEpoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.6f}",
                end="",
                flush=True,
            )


        if scheduler is not None:
            if val_loader is not None:
                scheduler.step(val_loss)
            else:
                scheduler.step(train_loss)


    print()

    if save_model:
        X_train, _ = train_loader.dataset.tensors
        seq_len = X_train.shape[1]
        model_name = model.model_name()
        opt_label = (
            f"{optimizer.__class__.__name__}_lr{optimizer.param_groups[0]['lr']}_"
        )
        config = f"{epochs}epochs_{seq_len}seq_{opt_label}_"
        save_path = f"out/models/{model_name}{config}{version}.pth"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"{version} model saved.")

    return model
