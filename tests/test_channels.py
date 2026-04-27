"""
Academic References:
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010.
  - pytest documentation: https://docs.pytest.org/
"""

"""Unit tests for quantum channels."""

import numpy as np
from quantum_comm_sim.channels.bosonic_channel import BosonicChannel
from quantum_comm_sim.channels.depolarizing_channel import DepolarizingChannel
from quantum_comm_sim.utils.state_utils import ket_to_density, is_density_matrix


def test_bosonic_preserves_density():
    psi = np.array([1, 0])
    rho = ket_to_density(psi)
    ch = BosonicChannel(dim=2, loss_db=1.0, noise_factor=0.0)
    rho_out = ch.apply(rho)
    assert is_density_matrix(rho_out)


def test_depolarizing_reduces_fidelity():
    psi = np.array([1, 0])
    rho = ket_to_density(psi)
    ch = DepolarizingChannel(p=0.5)
    rho_out = ch.apply(rho)
    fidelity = np.abs(np.vdot(psi, rho_out @ psi))
    assert fidelity < 1.0
