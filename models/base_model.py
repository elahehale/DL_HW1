from abc import ABC, abstractmethod

import torch.nn as nn


class LaserBaseModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.hidden_size = None
        self.num_layers = None

    @abstractmethod
    def forward(self, x):
        pass

    def model_name(self):
        parts = [self.__class__.__name__]

        if self.hidden_size is not None:
            parts.append(f"h{self.hidden_size}")

        if self.num_layers is not None:
            parts.append(f"l{self.num_layers}")

        return "_".join(parts)
