import os
import random
import numpy as np
import torch

def save_model(model, model_name):
    save_path = os.path.join("out", "models", f"{model_name}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def clear_current_line(length=80):
    print("\r" + " " * length + "\r", end="", flush=True)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = seconds % 3600 // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_remaining_time(
    elapsed_seconds: float,
    finished_count: int,
    total_count: int,
) -> float:
    if finished_count <= 0:
        return float("inf")

    return elapsed_seconds * (total_count - finished_count) / finished_count


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False