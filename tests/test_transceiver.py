"""Unit tests for transceiver blocks."""

import numpy as np
from quantum_comm_sim.transceiver.modulators import QPSKModulator


def test_qpsk_modulator_shape():
    mod = QPSKModulator(dim=2)
    symbols = np.array([0, 1, 2, 3])
    states = mod.modulate(symbols)
    assert states.shape == (4, 2, 2)
