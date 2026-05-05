import os
import torch

def save_model(model, model_name):
    save_path = os.path.join("out", "models", f"{model_name}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def clear_current_line(length=80):
    print("\r" + " " * length + "\r", end="", flush=True)