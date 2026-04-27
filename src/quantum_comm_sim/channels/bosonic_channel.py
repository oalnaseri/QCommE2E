"""
Academic References:
  - Giovannetti, V. et al., Bosonic Gaussian channels,
    arXiv:quant-ph/0404050, 2004.
  - Holevo, A.S. & Werner, R.F., Evaluating capacities of bosonic Gaussian channels,
    Physical Review A, 63(3), 032312, 2001.
  - Serafini, A., Quantum Continuous Variables, CRC Press, 2017.
  - Weedbrook, C. et al., Gaussian quantum information,
    Reviews of Modern Physics, 84(2), 621, 2012.
"""

"""Bosonic Gaussian channel (loss + noise)."""

import numpy as np
from .base_channel import BaseChannel


class BosonicChannel(BaseChannel):
    """Model attenuation and thermal noise via beamsplitter-like interaction.

    For a single mode, this is equivalent to a Gaussian loss channel
    with excess noise. In the discrete truncation used here, we model
    loss as a probabilistic erasure and add Gaussian noise in the
    quadratures.
    """

    def __init__(self, dim: int = 4, loss_db: float = 3.0, noise_factor: float = 0.01):
        super().__init__(dim)
        self.loss_db = loss_db
        self.noise_factor = noise_factor
        self.transmission = 10 ** (-loss_db / 10.0)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        # Placeholder: apply transmission scaling and add small thermal noise
        rho_out = self.transmission * rho
        noise = self.noise_factor * np.eye(self.dim) / self.dim
        rho_out = rho_out + noise
        # Renormalize trace
        rho_out = rho_out / np.trace(rho_out)
        return rho_out
