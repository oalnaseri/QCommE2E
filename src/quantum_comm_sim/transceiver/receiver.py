"""
Academic References:
  - OShea, T.J. & Hoydis, J., An Introduction to Deep Learning for the Physical Layer,
    IEEE Trans. Cogn. Commun. Netw., 3(4), 563-575, 2017.
  - Felix, A. et al., OFDM Autoencoder for End-to-End Learning of Communications Systems,
    IEEE SPAWC, 2018.
"""

"""Quantum receiver with detection and decoding."""

import numpy as np
from .detectors import Detector


class Receiver:
    """Detect incoming quantum states and demodulate to classical symbols."""

    def __init__(self, detector: Detector, decoder=None, equalizer=None):
        self.detector = detector
        self.decoder = decoder
        self.equalizer = equalizer

    def receive(self, states: np.ndarray) -> np.ndarray:
        if self.equalizer is not None:
            states = self.equalizer.equalize(states)
        if self.decoder is not None:
            states = self.decoder.decode(states)
        return self.detector.detect(states)
