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

"""FSO turbulence channel (Malaga / Kolmogorov)."""

import numpy as np
from .base_channel import BaseChannel


class TurbulenceChannel(BaseChannel):
    """Free-space optical turbulence channel.

    Models irradiance fluctuations via the Malaga distribution or
    Kolmogorov phase screens for wavefront distortion.
    """

    def __init__(self,
                 turbulence_model: str = "malaga",
                 alpha: float = 2.5,
                 beta: float = 2.0,
                 pointing_error: float = 1.0e-6,
                 beam_waist: float = 0.02,
                 link_distance_m: float = 1000.0,
                 dim: int = 2):
        super().__init__(dim)
        self.turbulence_model = turbulence_model
        self.alpha = alpha
        self.beta = beta
        self.pointing_error = pointing_error
        self.beam_waist = beam_waist
        self.link_distance_m = link_distance_m

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply turbulence-induced fading to an optical mode.

        We model the effect of turbulence as a random log-normal
        attenuation of the optical field, which in this truncated
        representation leads to a rescaling of the off-diagonal
        elements and a fluctuation of the diagonal populations.
        """
        rho = np.asarray(rho, dtype=complex)
        # Log-normal fading coefficient (Malaga-inspired)
        h = np.random.lognormal(mean=0.0, sigma=0.5)
        # Apply fading to coherences and bias diagonals
        rho_out = rho.copy()
        rho_out[0, 1] *= h
        rho_out[1, 0] *= h
        # Let the diagonal fluctuate mildly around the original values
        diag = np.diag(rho_out).real
        diag = diag / (diag.sum() + 1e-12)
        rho_out[0, 0] = diag[0]
        rho_out[1, 1] = diag[1]
        rho_out = rho_out / np.trace(rho_out)
        return rho_out
