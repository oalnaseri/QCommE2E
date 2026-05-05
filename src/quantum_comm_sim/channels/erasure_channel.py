"""
Academic References:
  - Bennett, C.H. et al., Capacities of quantum erasure channels,
    Physical Review Letters, 78(16), 3217, 1997.
  - Grassl, M. et al., Quantum erasure-correcting codes,
    arXiv:quant-ph/9610042, 1996.
"""

"""Quantum erasure channel."""

import numpy as np

from .base_channel import BaseChannel


class ErasureChannel(BaseChannel):
    """Erasure channel with erasure probability p.

    With probability 1-p the state is transmitted in the original
    code space; with probability p it is mapped to an orthogonal
    erasure flag on a (d+1)-dimensional output space.
    """

    def __init__(self, p: float = 0.1, dim: int = 2):
        super().__init__(dim + 1)
        self._validate_probability("p", p)
        self.p = p
        self.input_dim = dim
        self.erasure_label = -1

    def apply(self, rho: np.ndarray) -> np.ndarray:
        rho = self._coerce_density_matrix(rho, dim=self.input_dim)
        rho_out = np.zeros((self.input_dim + 1, self.input_dim + 1), dtype=complex)
        rho_out[:self.input_dim, :self.input_dim] = (1 - self.p) * rho
        rho_out[self.input_dim, self.input_dim] = self.p
        return rho_out
