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
