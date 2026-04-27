"""
Academic References:
  - Gottesman, D., Stabilizer Codes and Quantum Error Correction,
    arXiv:quant-ph/9705052, 1997 (Caltech PhD thesis).
  - Lidar, D.A. & Brun, T.A. (Eds.), Quantum Error Correction, Cambridge, 2013.
"""

"""Quantum error encoders."""

import numpy as np
from abc import ABC, abstractmethod


class Encoder(ABC):
    """Abstract encoder."""

    @abstractmethod
    def encode(self, states: np.ndarray) -> np.ndarray:
        """Apply encoding to input states."""


class IdentityEncoder(Encoder):
    """No-op encoder for baseline."""

    def encode(self, states: np.ndarray) -> np.ndarray:
        return states
