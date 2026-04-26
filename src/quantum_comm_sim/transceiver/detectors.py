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
