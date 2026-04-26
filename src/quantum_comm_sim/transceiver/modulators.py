"""Modulators for quantum state preparation."""

import numpy as np
from abc import ABC, abstractmethod


class Modulator(ABC):
    """Abstract modulator."""

    @abstractmethod
    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """Map symbols to quantum states (density matrices or amplitudes)."""


class QPSKModulator(Modulator):
    """QPSK modulation for coherent states or qubits."""

    def __init__(self, dim: int = 2):
        self.dim = dim
        self.constellation = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 1]) / np.sqrt(2),
            3: np.array([1, -1]) / np.sqrt(2),
        }

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        states = []
        for s in symbols:
            psi = self.constellation[int(s) % 4]
            rho = np.outer(psi, psi.conj())
            states.append(rho)
        return np.stack(states)
