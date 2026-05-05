"""
Academic References:
  - Jurado-Navas, A. et al., A unified statistical model for Malaga distributed optical
    scattering communications, Optics Communications, 468, 126204, 2020.
  - Jurado-Navas, A. et al., A unifying statistical model for atmospheric optical
    scintillation, arXiv:1202.1798, 2012.
  - Al-Habash, M.A. et al., Mathematical model for the irradiance PDF of a laser beam
    propagating through turbulent media, Optical Engineering, 40(8), 1554-1562, 2001.
  - Yang, F. et al., Performance Analysis of Free-Space Optical Links Over Malaga (M)
    Turbulence Channels with Pointing Errors, arXiv:1805.05572, 2018.
"""

"""FSO turbulence channel (Gamma-Gamma / Malaga-inspired)."""

import numpy as np

from .base_channel import BaseChannel
from .bosonic_channel import apply_pure_loss_channel


class TurbulenceChannel(BaseChannel):
    """Free-space optical turbulence channel.

    Models turbulence as a random transmittance process using a
    Gamma-Gamma scintillation surrogate for Malaga-like fading,
    combined with a passive attenuation channel.
    """

    def __init__(
        self,
        turbulence_model: str = "malaga",
        alpha: float = 2.5,
        beta: float = 2.0,
        pointing_error: float = 1.0e-6,
        beam_waist: float = 0.02,
        link_distance_m: float = 1000.0,
        dim: int = 2,
    ):
        super().__init__(dim)
        self.turbulence_model = turbulence_model
        self.alpha = alpha
        self.beta = beta
        self.pointing_error = pointing_error
        self.beam_waist = beam_waist
        self.link_distance_m = link_distance_m

    def _sample_transmissivity(self) -> float:
        waist = max(self.beam_waist, 1.0e-12)
        pointing_coupling = np.exp(-2.0 * (self.pointing_error / waist) ** 2)

        if self.turbulence_model in {"malaga", "gamma-gamma", "gamma_gamma"}:
            large_scale = np.random.gamma(shape=self.alpha, scale=1.0 / self.alpha)
            small_scale = np.random.gamma(shape=self.beta, scale=1.0 / self.beta)
            scintillation = large_scale * small_scale
        else:
            sigma = 0.5 / np.sqrt(max(self.alpha, 1.0e-12))
            scintillation = np.random.lognormal(mean=-0.5 * sigma**2, sigma=sigma)

        return float(np.clip(pointing_coupling * scintillation, 0.0, 1.0))

    def apply(self, rho: np.ndarray) -> np.ndarray:
        rho = self._coerce_density_matrix(rho)
        transmission = self._sample_transmissivity()
        return apply_pure_loss_channel(rho, transmission)
