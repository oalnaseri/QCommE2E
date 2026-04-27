"""
Academic References:
  - Alvarado, A. et al., Replacing the Soft-Decision FEC Limit Paradigm in the
    Design of Optical Communication Systems, J. Lightwave Technol., 33(20), 4338-4352, 2015.
  - Gumus, K. et al., End-to-End Learning of Geometrical Shaping Maximizing
    Generalized Mutual Information, arXiv:1912.05638, 2019.
  - Bocherer, G. et al., Bandwidth efficient and rate-matched low-density parity-check
    coded modulation, IEEE Trans. Commun., 63(12), 4651-4665, 2015.
"""

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
