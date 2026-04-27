"""
Academic References:
  - OShea, T.J. & Hoydis, J., An Introduction to Deep Learning for the Physical Layer,
    IEEE Trans. Cogn. Commun. Netw., 3(4), 563-575, 2017.
  - Felix, A. et al., OFDM Autoencoder for End-to-End Learning of Communications Systems,
    IEEE SPAWC, 2018.
  - Dorner, S. et al., Deep learning based communication over the air,
    IEEE J. Sel. Topics Signal Process., 12(1), 132-143, 2018.
"""

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
