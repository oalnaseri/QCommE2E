"""Bit and symbol error rate metrics."""

import numpy as np


def compute_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Compute bit error rate."""
    errors = np.sum(tx_bits != rx_bits)
    return errors / max(1, tx_bits.size)


def compute_ser(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    """Compute symbol error rate."""
    errors = np.sum(tx_symbols != rx_symbols)
    return errors / max(1, tx_symbols.size)
