"""Neural equalizer (GNN / deep unfolding)."""

import torch
import torch.nn as nn
import numpy as np
from .kalman_equalizer import Equalizer


class NeuralEqualizer(Equalizer):
    """Learned neural equalizer using a lightweight MLP.

    Future: extend to GNN for graph-based fiber topology or
    deep unfolding of iterative algorithms.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_layers: int = 3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def equalize(self, states: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.real(states), dtype=torch.float32)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        with torch.no_grad():
            y = self.net(x)
        return y.numpy().reshape(states.shape)
