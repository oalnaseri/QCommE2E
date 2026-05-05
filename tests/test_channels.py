"""
Academic References:
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010.
  - pytest documentation: https://docs.pytest.org/
"""

"""Unit tests for quantum channels."""

import numpy as np
from quantum_comm_sim.channels.bosonic_channel import BosonicChannel
from quantum_comm_sim.channels.dephasing_channel import DephasingChannel
from quantum_comm_sim.channels.depolarizing_channel import DepolarizingChannel
from quantum_comm_sim.channels.erasure_channel import ErasureChannel
from quantum_comm_sim.channels.pmd_channel import PMDChannel
from quantum_comm_sim.channels.turbulence_channel import TurbulenceChannel
from quantum_comm_sim.utils.state_utils import ket_to_density, is_density_matrix


def test_bosonic_preserves_density():
    psi = np.array([1, 0])
    rho = ket_to_density(psi)
    ch = BosonicChannel(dim=2, loss_db=1.0, noise_factor=0.0)
    rho_out = ch.apply(rho)
    assert is_density_matrix(rho_out)


def test_bosonic_pure_loss_matches_single_photon_decay():
  rho = np.array([[0, 0], [0, 1]], dtype=complex)
  ch = BosonicChannel(dim=2, loss_db=3.0, noise_factor=0.0)
  rho_out = ch.apply(rho)
  expected = np.array(
    [[1.0 - ch.transmission, 0.0], [0.0, ch.transmission]],
    dtype=complex,
  )
  assert np.allclose(rho_out, expected, atol=1.0e-6)


def test_depolarizing_reduces_fidelity():
    psi = np.array([1, 0])
    rho = ket_to_density(psi)
    ch = DepolarizingChannel(p=0.5)
    rho_out = ch.apply(rho)
    fidelity = np.abs(np.vdot(psi, rho_out @ psi))
    assert fidelity < 1.0


def test_dephasing_preserves_populations_and_damps_coherence():
  rho = np.array([[0.4, 0.3], [0.3, 0.6]], dtype=complex)
  ch = DephasingChannel(p=0.25)
  rho_out = ch.apply(rho)
  assert np.allclose(np.diag(rho_out), np.diag(rho))
  assert np.isclose(rho_out[0, 1], 0.75 * rho[0, 1])
  assert is_density_matrix(rho_out)


def test_erasure_channel_adds_orthogonal_flag_state():
  psi = np.array([1, 0])
  rho = ket_to_density(psi)
  ch = ErasureChannel(p=0.25, dim=2)
  rho_out = ch.apply(rho)
  assert rho_out.shape == (3, 3)
  assert np.allclose(rho_out[:2, :2], 0.75 * rho)
  assert np.isclose(rho_out[2, 2], 0.25)
  assert is_density_matrix(rho_out)


def test_turbulence_preserves_density():
  psi = np.array([0, 1])
  rho = ket_to_density(psi)
  ch = TurbulenceChannel(turbulence_model="malaga", alpha=2.5, beta=2.0, dim=2)
  np.random.seed(0)
  rho_out = ch.apply(rho)
  assert is_density_matrix(rho_out)


def test_pmd_increases_decoherence_with_dgd():
  psi = np.array([1, 1]) / np.sqrt(2)
  rho = ket_to_density(psi)

  np.random.seed(0)
  low_dgd = PMDChannel(dgd_mean_ps=0.0, num_sections=4, signal_bandwidth_ghz=20.0).apply(rho)
  np.random.seed(0)
  high_dgd = PMDChannel(dgd_mean_ps=50.0, num_sections=4, signal_bandwidth_ghz=20.0).apply(rho)

  purity_low = np.real(np.trace(low_dgd @ low_dgd))
  purity_high = np.real(np.trace(high_dgd @ high_dgd))
  assert is_density_matrix(low_dgd)
  assert is_density_matrix(high_dgd)
  assert purity_high < purity_low
