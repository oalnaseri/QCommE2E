"""Dephasing (phase damping) channel."""

import numpy as np
from .base_channel import BaseChannel


class DephasingChannel(BaseChannel):
    """Phase damping channel.

    E(rho) = (1 - p) rho + p Z rho Z
    """

    def __init__(self, p: float = 0.1):
        super().__init__(dim=2)
        self.p = p
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        return (1 - self.p) * rho + self.p * self.Z @ rho @ self.Z
