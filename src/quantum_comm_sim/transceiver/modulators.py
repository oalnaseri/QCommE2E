"""
Academic References:
  - Proakis, J.G. & Salehi, M., Digital Communications, 5th Ed., McGraw-Hill, 2008.
  - Agrell, E. et al., Capacity of a modulo-sum optical channel,
    J. Lightwave Technol., 37(7), 1629-1637, 2019.
"""

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

    BIT_LABELS = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=int,
    )

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

    def symbol_alphabet(self) -> np.ndarray:
        """Return the supported symbol labels in detector order."""
        return np.array(sorted(self.constellation), dtype=int)

    def reference_states(self) -> np.ndarray:
        """Return the density-matrix codebook used by this modulator."""
        return self.modulate(self.symbol_alphabet())

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Map symbol labels to a fixed two-bit representation."""
        symbol_array = np.asarray(symbols, dtype=int)
        bits = np.full(symbol_array.shape + (self.BIT_LABELS.shape[1],), -1, dtype=int)
        valid = (symbol_array >= 0) & (symbol_array < len(self.BIT_LABELS))
        bits[valid] = self.BIT_LABELS[symbol_array[valid]]
        return bits
