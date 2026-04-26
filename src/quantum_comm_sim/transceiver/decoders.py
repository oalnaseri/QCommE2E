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
