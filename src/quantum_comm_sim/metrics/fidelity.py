"""Quantum fidelity metrics."""

import numpy as np


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute quantum fidelity F(rho, sigma).

    For pure states: |<psi|phi>|^2.
    For mixed states: (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2.
    """
    if rho.ndim == 2 and sigma.ndim == 2:
        sqrt_rho = np.real(np.linalg.eigvalsh(rho))
        sqrt_rho = np.diag(np.sqrt(np.maximum(sqrt_rho, 0)))
        mat = sqrt_rho @ sigma @ sqrt_rho
        eig = np.linalg.eigvalsh(mat)
        return float(np.sum(np.sqrt(np.maximum(eig, 0))) ** 2)
    return float(np.abs(np.vdot(rho, sigma)) ** 2)
