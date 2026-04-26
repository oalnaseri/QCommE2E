"""Quantum transmitter with state preparation and modulation."""

import numpy as np
from .modulators import Modulator


class Transmitter:
    """Prepare quantum states and apply encoding."""

    def __init__(self, modulator: Modulator, encoder=None):
        self.modulator = modulator
        self.encoder = encoder

    def transmit(self, symbols: np.ndarray) -> np.ndarray:
        states = self.modulator.modulate(symbols)
        if self.encoder is not None:
            states = self.encoder.encode(states)
        return states
