"""
Academic References:
  - Proakis, J.G. & Salehi, M., Digital Communications, 5th Ed., McGraw-Hill, 2008.
  - Agrell, E. et al., Capacity of a modulo-sum optical channel,
    J. Lightwave Technol., 37(7), 1629-1637, 2019.
"""

"""Modulators for quantum state preparation."""

import numpy as np
from abc import ABC, abstractmethod


def _binary_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=int)


def _gray_bits(value: int, width: int) -> np.ndarray:
    gray = value ^ (value >> 1)
    return _binary_bits(gray, width)


class Modulator(ABC):
    """Abstract modulator."""

    @abstractmethod
    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """Map symbols to quantum states (density matrices or amplitudes)."""


class QAMModulator(Modulator):
    """Square M-QAM modulation mapped to qubit density matrices.

    This class supports M = 4, 16, 64, ... (square powers of two).
    Each constellation point is embedded in a qubit pure state using
    psi = [1, alpha] / ||[1, alpha]||, where alpha is the normalized
    complex QAM point.
    """

    def __init__(self, m_order: int = 16, dim: int = 2, amplitude_scale: float = 1.0):
        if dim != 2:
            raise ValueError("QAMModulator currently supports dim=2 only")
        if m_order < 4:
            raise ValueError("m_order must be >= 4 for QAM")

        root_m = int(np.sqrt(m_order))
        if root_m * root_m != m_order:
            raise ValueError("m_order must be a perfect square for square QAM")
        if (root_m & (root_m - 1)) != 0:
            raise ValueError("sqrt(m_order) must be a power of two")

        self.dim = dim
        self.m_order = int(m_order)
        self.bits_per_symbol = int(np.log2(self.m_order))
        self.axis_bits = self.bits_per_symbol // 2
        self.amplitude_scale = float(amplitude_scale)

        levels = np.arange(-(root_m - 1), root_m, 2, dtype=float)
        q_levels = levels[::-1]
        norm = np.sqrt((2.0 / 3.0) * (self.m_order - 1))
        self.points = []
        self.bit_labels = []
        self.constellation = {}

        symbol = 0
        for q_index, q_level in enumerate(q_levels):
            q_bits = _gray_bits(q_index, self.axis_bits)
            for i_index, i_level in enumerate(levels):
                i_bits = _gray_bits(i_index, self.axis_bits)
                point = self.amplitude_scale * (i_level + 1j * q_level) / norm
                self.points.append(point)
                self.bit_labels.append(np.concatenate([i_bits, q_bits]))
                self.constellation[symbol] = point
                symbol += 1

        self.points = np.asarray(self.points, dtype=complex)
        self.bit_labels = np.asarray(self.bit_labels, dtype=int)

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        states = []
        for symbol in np.asarray(symbols, dtype=int):
            point = self.constellation[int(symbol) % self.m_order]
            psi = np.array([1.0, point], dtype=complex)
            psi = psi / np.linalg.norm(psi)
            states.append(np.outer(psi, psi.conj()))
        return np.stack(states)

    def symbol_alphabet(self) -> np.ndarray:
        return np.arange(self.m_order, dtype=int)

    def reference_states(self) -> np.ndarray:
        return self.modulate(self.symbol_alphabet())

    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        symbol_array = np.asarray(symbols, dtype=int)
        bits = np.full(symbol_array.shape + (self.bits_per_symbol,), -1, dtype=int)
        valid = (symbol_array >= 0) & (symbol_array < self.m_order)
        bits[valid] = self.bit_labels[symbol_array[valid]]
        return bits


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
