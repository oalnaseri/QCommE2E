"""
Academic References:
  - Proakis, J.G. & Salehi, M., Digital Communications, 5th Ed., McGraw-Hill, 2008.
  - pytest documentation: https://docs.pytest.org/
"""

"""Unit tests for metrics."""

import numpy as np
from quantum_comm_sim.metrics.ber_ser import compute_ber, compute_ser


def test_ber_perfect():
    tx = np.array([0, 1, 0, 1])
    rx = np.array([0, 1, 0, 1])
    assert compute_ber(tx, rx) == 0.0


def test_ber_half():
    tx = np.array([0, 1])
    rx = np.array([1, 0])
    assert compute_ber(tx, rx) == 1.0
