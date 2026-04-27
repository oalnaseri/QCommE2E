"""
Academic References:
  - Hunter, J.D., Matplotlib: A 2D graphics environment, Computing in Science & Engineering,
    9(3), 90-95, 2007.
  - Hecht, N. et al., Constellation diagram analysis for optical communications,
    J. Lightwave Technol., 34(2), 533-541, 2016.
"""

"""Plotting utilities for constellations and metrics."""

import matplotlib.pyplot as plt
import numpy as np


def plot_constellation(symbols: np.ndarray, title: str = "Constellation"):
    """Scatter plot of complex symbols."""
    plt.figure()
    if symbols.ndim == 1:
        x = np.real(symbols)
        y = np.imag(symbols)
    else:
        x = np.real(symbols[:, 0])
        y = np.imag(symbols[:, 0])
    plt.scatter(x, y, alpha=0.6)
    plt.title(title)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()


def plot_ber_curve(snr_db: np.ndarray, ber: np.ndarray, label: str = "BER"):
    """Plot BER vs SNR curve."""
    plt.figure()
    plt.semilogy(snr_db, ber, marker='o', label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
