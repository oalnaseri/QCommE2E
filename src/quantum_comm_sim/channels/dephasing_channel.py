"""
Academic References:
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010. Sec. 8.3 (Phase damping / dephasing).
  - Preskill, J., Quantum Computation, Lecture Notes, Caltech.
"""

"""Dephasing (phase damping) channel."""

import numpy as np

from .base_channel import BaseChannel


class DephasingChannel(BaseChannel):
    """Phase-damping channel.

    The qubit populations are preserved and the off-diagonal elements
    are attenuated by a factor (1 - p).
    """

    def __init__(self, p: float = 0.1):
        super().__init__(dim=2)
        self._validate_probability("p", p)
        self.p = p

    def apply(self, rho: np.ndarray) -> np.ndarray:
        rho = self._coerce_density_matrix(rho, dim=2)
        rho_out = rho.copy()
        rho_out[0, 1] *= (1 - self.p)
        rho_out[1, 0] *= (1 - self.p)
        return rho_out
