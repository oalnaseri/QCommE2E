"""
Academic References:
  - Gerry, C.C. & Knight, P.L., Introductory Quantum Optics, Cambridge, 2005.
  - Leonhardt, U., Measuring the Quantum State of Light, Cambridge, 1997.
  - Wiseman, H.M. & Milburn, G.J., Quantum Measurement and Control, Cambridge, 2009.
"""

"""Detection schemes."""

import numpy as np
from abc import ABC, abstractmethod


class Detector(ABC):
    """Abstract detector."""

    @abstractmethod
    def detect(self, states: np.ndarray) -> np.ndarray:
        """Map quantum states to classical symbols or soft decisions."""


class HomodyneDetector(Detector):
    """Homodyne detection measuring a quadrature."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def detect(self, states: np.ndarray) -> np.ndarray:
        # Placeholder: measure real part of off-diagonal
        if states.ndim == 3:
            vals = np.real(states[:, 0, 1])
        else:
            vals = np.real(states)
        return (vals > self.threshold).astype(int)
