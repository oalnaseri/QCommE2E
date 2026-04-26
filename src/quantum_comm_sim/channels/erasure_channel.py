"""Quantum erasure channel."""

import numpy as np
from .base_channel import BaseChannel


class ErasureChannel(BaseChannel):
    """Erasure channel with erasure probability p.

    With probability 1-p the state is transmitted; with probability p
    it is replaced by a fixed erasure state.
    """

    def __init__(self, p: float = 0.1, dim: int = 2):
        super().__init__(dim)
        self.p = p

    def apply(self, rho: np.ndarray) -> np.ndarray:
        # Placeholder: probabilistic replacement
        erasure_state = np.eye(self.dim) / self.dim
        return (1 - self.p) * rho + self.p * erasure_state
