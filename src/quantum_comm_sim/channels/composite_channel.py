"""
Academic References:
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010. Ch. 8 (Serial concatenation of channels).
  - Wilde, M.M., Quantum Information Theory, Cambridge University Press, 2013.
"""

"""Cascade multiple channels sequentially."""

import numpy as np
from typing import List
from .base_channel import BaseChannel


class CompositeChannel(BaseChannel):
    """Compose a list of channels as E_total = E_n ... E_2 E_1."""

    def __init__(self, channels: List[BaseChannel]):
        # Inherit dim from first channel
        super().__init__(dim=channels[0].dim)
        self.channels = channels

    def apply(self, rho: np.ndarray) -> np.ndarray:
        for ch in self.channels:
            rho = ch.apply(rho)
        return rho
