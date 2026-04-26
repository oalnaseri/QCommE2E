"""Density matrix and state vector helpers."""

import numpy as np


def ket_to_density(psi: np.ndarray) -> np.ndarray:
    """Convert pure state vector to density matrix."""
    return np.outer(psi, psi.conj())


def is_density_matrix(rho: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if matrix is a valid density matrix."""
    if rho.shape[0] != rho.shape[1]:
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        return False
    eig = np.linalg.eigvalsh(rho)
    return np.all(eig >= -tol)
