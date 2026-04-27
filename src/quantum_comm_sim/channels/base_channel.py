"""
Academic References:
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010. Ch. 8 (Quantum Channels, Kraus operators).
  - Choi, M.-D., Completely positive linear maps on complex matrices,
    Linear Algebra and its Applications, 1975.
  - Kraus, K., States, Effects, and Operations, Springer, 1983.
  - Wilde, M.M., Quantum Information Theory, Cambridge University Press, 2013.
"""

"""Abstract base for quantum channels using Kraus formalism."""

from abc import ABC, abstractmethod
import numpy as np


class BaseChannel(ABC):
    """Abstract quantum channel.

    All channels must implement the `apply` method following the Kraus
    operator formalism:
        E(rho) = sum_k E_k rho E_k^dagger
    """

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply channel to density matrix rho.

        Args:
            rho: Density matrix of shape (dim, dim).

        Returns:
            Output density matrix of shape (dim, dim).
        """

    def __call__(self, rho: np.ndarray) -> np.ndarray:
        return self.apply(rho)
