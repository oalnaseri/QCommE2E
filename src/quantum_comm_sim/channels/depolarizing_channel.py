"""Depolarizing qubit channel."""

import numpy as np
from .base_channel import BaseChannel


class DepolarizingChannel(BaseChannel):
    """Standard depolarizing channel for qubits.

    E(rho) = (1 - p) rho + p * I / 2
    """

    def __init__(self, p: float = 0.1):
        super().__init__(dim=2)
        self.p = p

    def apply(self, rho: np.ndarray) -> np.ndarray:
        return (1 - self.p) * rho + self.p * np.eye(2) / 2.0
