"""Mutual information and generalized mutual information."""

import numpy as np


def estimate_mi(tx_symbols: np.ndarray, rx_llrs: np.ndarray) -> float:
    """Placeholder for mutual information estimation.

    In practice, use histogram-based or KDE-based MI estimators.
    """
    return 1.0 - np.mean(np.abs(tx_symbols - rx_llrs))


def estimate_gmi(tx_symbols: np.ndarray, rx_llrs: np.ndarray) -> float:
    """Placeholder for generalized mutual information.

    GMI approximates achievable information rate with mismatched decoding.
    """
    return estimate_mi(tx_symbols, rx_llrs)
