"""
Academic References:
  - Proakis, J.G. & Salehi, M., Digital Communications, 5th Ed., McGraw-Hill, 2008.
  - pytest documentation: https://docs.pytest.org/
"""

"""Unit tests for transceiver blocks."""

import numpy as np
from quantum_comm_sim.transceiver.detectors import PrettyGoodMeasurementDetector
from quantum_comm_sim.transceiver.modulators import QPSKModulator


def test_qpsk_modulator_shape():
    mod = QPSKModulator(dim=2)
    symbols = np.array([0, 1, 2, 3])
    states = mod.modulate(symbols)
    assert states.shape == (4, 2, 2)


def test_qpsk_symbols_to_bits_are_two_bit_labels():
  mod = QPSKModulator(dim=2)
  bits = mod.symbols_to_bits(np.array([0, 1, 2, 3]))
  np.testing.assert_array_equal(
    bits,
    np.array(
      [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]
    ),
  )


def test_pretty_good_measurement_detector_recovers_reference_states():
  mod = QPSKModulator(dim=2)
  detector = PrettyGoodMeasurementDetector(
    mod.reference_states(),
    labels=mod.symbol_alphabet(),
  )
  detected = detector.detect(mod.reference_states())
  np.testing.assert_array_equal(detected, mod.symbol_alphabet())
