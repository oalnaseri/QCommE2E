"""
Academic References:
  - Giovannetti, V. et al., Bosonic Gaussian channels,
    arXiv:quant-ph/0404050, 2004.
  - Holevo, A.S. & Werner, R.F., Evaluating capacities of bosonic Gaussian channels,
    Physical Review A, 63(3), 032312, 2001.
  - Serafini, A., Quantum Continuous Variables, CRC Press, 2017.
  - Weedbrook, C. et al., Gaussian quantum information,
    Reviews of Modern Physics, 84(2), 621, 2012.
"""

"""Bosonic Gaussian channel (loss + noise)."""

from math import comb

import numpy as np
from scipy.linalg import expm

from .base_channel import BaseChannel


def _annihilation_operator(dim: int) -> np.ndarray:
    operator = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        operator[n - 1, n] = np.sqrt(n)
    return operator


def _thermal_state(dim: int, n_th: float) -> np.ndarray:
    if n_th <= 0:
        state = np.zeros((dim, dim), dtype=complex)
        state[0, 0] = 1.0
        return state

    probs = np.array(
        [n_th**n / ((1.0 + n_th) ** (n + 1)) for n in range(dim)],
        dtype=float,
    )
    probs = probs / probs.sum()
    return np.diag(probs.astype(complex))


def _beamsplitter_unitary(dim: int, transmission: float) -> np.ndarray:
    transmission = float(np.clip(transmission, 0.0, 1.0))
    theta = np.arccos(np.sqrt(transmission))
    system_a = _annihilation_operator(dim)
    env_b = _annihilation_operator(dim)
    generator = theta * (
        np.kron(system_a, env_b.conj().T) - np.kron(system_a.conj().T, env_b)
    )
    return expm(generator)


def _partial_trace_environment(joint_state: np.ndarray, dim: int) -> np.ndarray:
    tensor = joint_state.reshape(dim, dim, dim, dim)
    return np.einsum("ijkj->ik", tensor)


def _pure_loss_kraus(dim: int, transmission: float) -> list[np.ndarray]:
    transmission = float(np.clip(transmission, 0.0, 1.0))
    kraus = []
    for lost_photons in range(dim):
        operator = np.zeros((dim, dim), dtype=complex)
        for n in range(lost_photons, dim):
            operator[n - lost_photons, n] = (
                np.sqrt(comb(n, lost_photons))
                * ((1.0 - transmission) ** (lost_photons / 2.0))
                * (transmission ** ((n - lost_photons) / 2.0))
            )
        kraus.append(operator)
    return kraus


def _apply_kraus_channel(rho: np.ndarray, kraus_ops: list[np.ndarray]) -> np.ndarray:
    rho_out = np.zeros_like(rho, dtype=complex)
    for operator in kraus_ops:
        rho_out += operator @ rho @ operator.conj().T
    return 0.5 * (rho_out + rho_out.conj().T)


def apply_pure_loss_channel(rho: np.ndarray, transmission: float) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    return _apply_kraus_channel(rho, _pure_loss_kraus(rho.shape[0], transmission))


class BosonicChannel(BaseChannel):
    """Single-mode bosonic thermal-loss channel in a truncated Fock basis.

    The channel mixes the input mode with a thermal environment state on
    a beamsplitter of transmissivity eta.
    """

    def __init__(self, dim: int = 4, loss_db: float = 3.0, noise_factor: float = 0.01):
        super().__init__(dim)
        self.loss_db = loss_db
        self.noise_factor = max(0.0, float(noise_factor))
        self.transmission = 10 ** (-loss_db / 10.0)
        self.environment_state = _thermal_state(self.dim, self.noise_factor)
        self.beamsplitter_unitary = _beamsplitter_unitary(self.dim, self.transmission)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        rho = self._coerce_density_matrix(rho)
        joint_state = np.kron(rho, self.environment_state)
        joint_out = self.beamsplitter_unitary @ joint_state @ self.beamsplitter_unitary.conj().T
        return _partial_trace_environment(joint_out, self.dim)
