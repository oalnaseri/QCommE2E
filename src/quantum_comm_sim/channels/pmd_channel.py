"""
Academic References:
  - Gordon, J.P. & Kogelnik, H., PMD fundamentals: polarization mode dispersion
  in optical fibers, Proceedings of the National Academy of Sciences, 97(9), 4541-4550, 2000.
  - Poole, C.D. & Wagner, R.E., Phenomenological approach to polarization dispersion
  in long single-mode fibers, Electronics Letters, 22(19), 1029-1030, 1986.
  - Wai, P.K.A. & Menyuk, C.R., Polarization mode dispersion, decorrelations,
  and fiber modeling, J. Lightwave Technol., 14(2), 148-157, 1996.
"""

"""Polarization Mode Dispersion (PMD) channel."""

import numpy as np

from .base_channel import BaseChannel


def _haar_unitary_2() -> np.ndarray:
  matrix = np.random.normal(size=(2, 2)) + 1j * np.random.normal(size=(2, 2))
  q_mat, r_mat = np.linalg.qr(matrix)
  phases = np.diag(r_mat)
  phases = np.where(np.abs(phases) > 1.0e-12, phases / np.abs(phases), 1.0)
  return q_mat @ np.diag(np.conj(phases))


class PMDChannel(BaseChannel):
  """Spectrum-averaged PMD surrogate for a polarization qubit.

  Each section applies a random principal-state basis rotation and a
  dephasing factor derived from the section DGD and signal bandwidth.
  """

  def __init__(
    self,
    dgd_mean_ps: float = 1.0,
    num_sections: int = 10,
    signal_bandwidth_ghz: float = 10.0,
  ):
    super().__init__(dim=2)
    self.dgd_mean_ps = dgd_mean_ps
    self.num_sections = max(1, int(num_sections))
    self.signal_bandwidth_ghz = signal_bandwidth_ghz

  def _section_visibility(self) -> float:
    section_dgd_s = (self.dgd_mean_ps * 1.0e-12) / np.sqrt(self.num_sections)
    sigma_omega = 2.0 * np.pi * self.signal_bandwidth_ghz * 1.0e9
    return float(np.exp(-0.5 * (sigma_omega * section_dgd_s) ** 2))

  def apply(self, rho: np.ndarray) -> np.ndarray:
    rho = self._coerce_density_matrix(rho, dim=2)
    visibility = self._section_visibility()
    rho_out = rho.copy()
    for _ in range(self.num_sections):
      jones_rotation = _haar_unitary_2()
      rho_psp = jones_rotation.conj().T @ rho_out @ jones_rotation
      rho_psp[0, 1] *= visibility
      rho_psp[1, 0] *= visibility
      rho_out = jones_rotation @ rho_psp @ jones_rotation.conj().T
    return 0.5 * (rho_out + rho_out.conj().T)
