import torch
import os

def train_model(epochs, model, optimizer, criterion, train_loader, val_loader, device, version = "", save_model = True):

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
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

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
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

    if save_model:
        X_train, _ = train_loader.dataset.tensors
        seq_len = X_train.shape[1]
        model_name = f"{model.__class__.__name__}_h{model.lstm.hidden_size}_l{model.lstm.num_layers}_"
        opt_label = f"{optimizer.__class__.__name__}_lr{optimizer.param_groups[0]['lr']}"
        config = f"{epochs}epochs_{seq_len}seq_{opt_label}_"
        save_path = f"out/models/{model_name}{config}{version}.pth"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"{version} model saved.")