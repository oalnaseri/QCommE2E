"""
Academic References:
  - Jozsa, R., Fidelity for mixed quantum states, J. Mod. Opt., 41(12), 2315-2323, 1994.
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010. Sec. 9.2.3 (Fidelity).
  - Uhlmann, A., The transition probability in the state space of a *-algebra,
    Reports on Mathematical Physics, 9(2), 273-279, 1976.
"""

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
