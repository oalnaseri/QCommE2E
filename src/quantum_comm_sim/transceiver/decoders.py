"""
Academic References:
  - Lidar, D.A. & Brun, T.A. (Eds.), Quantum Error Correction, Cambridge, 2013.
  - Joschiura, A. et al., Quantum Autoencoders for Denoising,
    arXiv:2304.00079, 2023.
"""

"""Quantum decoders / demapping."""

import numpy as np
from abc import ABC, abstractmethod


class Decoder(ABC):
    """Abstract decoder."""

    @abstractmethod
    def decode(self, states: np.ndarray) -> np.ndarray:
        """Apply decoding to received states."""


class IdentityDecoder(Decoder):
    """No-op decoder for baseline."""

    def decode(self, states: np.ndarray) -> np.ndarray:
        return states
