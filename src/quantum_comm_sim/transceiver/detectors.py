"""
Academic References:
  - Gerry, C.C. & Knight, P.L., Introductory Quantum Optics, Cambridge, 2005.
  - Leonhardt, U., Measuring the Quantum State of Light, Cambridge, 1997.
  - Wiseman, H.M. & Milburn, G.J., Quantum Measurement and Control, Cambridge, 2009.
"""

"""Detection schemes."""

import numpy as np
from abc import ABC, abstractmethod


def _inverse_sqrt_psd(matrix: np.ndarray, atol: float = 1.0e-12) -> np.ndarray:
    """Return the inverse square root of a positive semidefinite matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    safe_eigenvalues = np.where(eigenvalues > atol, 1.0 / np.sqrt(eigenvalues), 0.0)
    return eigenvectors @ np.diag(safe_eigenvalues) @ eigenvectors.conj().T


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


class PrettyGoodMeasurementDetector(Detector):
    """Hard-decision detector based on the pretty-good measurement.

    For an ensemble of reference states with priors p_i, the POVM is
        E_i = p_i rho_bar^(-1/2) rho_i rho_bar^(-1/2)
    where rho_bar = sum_i p_i rho_i.
    """

    def __init__(
        self,
        reference_states: np.ndarray,
        labels: np.ndarray | None = None,
        priors: np.ndarray | None = None,
    ):
        states = np.asarray(reference_states, dtype=complex)
        if states.ndim != 3 or states.shape[1] != states.shape[2]:
            raise ValueError("reference_states must have shape (M, dim, dim)")

        num_states = states.shape[0]
        if labels is None:
            labels = np.arange(num_states)
        labels = np.asarray(labels)
        if labels.shape != (num_states,):
            raise ValueError("labels must have shape (M,)")

        if priors is None:
            priors = np.full(num_states, 1.0 / num_states)
        priors = np.asarray(priors, dtype=float)
        if priors.shape != (num_states,):
            raise ValueError("priors must have shape (M,)")
        if np.any(priors < 0):
            raise ValueError("priors must be non-negative")
        prior_sum = priors.sum()
        if prior_sum <= 0:
            raise ValueError("priors must sum to a positive value")
        priors = priors / prior_sum

        average_state = np.tensordot(priors, states, axes=(0, 0))
        inv_sqrt_average = _inverse_sqrt_psd(average_state)
        povm = []
        for prior, state in zip(priors, states):
            element = prior * inv_sqrt_average @ state @ inv_sqrt_average
            povm.append(0.5 * (element + element.conj().T))

        self.reference_states = states
        self.labels = labels
        self.priors = priors
        self.povm = np.stack(povm)

    def detect(self, states: np.ndarray) -> np.ndarray:
        rho = np.asarray(states, dtype=complex)
        single_state = False
        if rho.ndim == 2:
            rho = rho[None, ...]
            single_state = True
        elif rho.ndim != 3:
            raise ValueError("states must have shape (N, dim, dim) or (dim, dim)")

        scores = np.real(np.einsum("aij,nji->na", self.povm, rho))
        detected = self.labels[np.argmax(scores, axis=1)]
        return detected[0] if single_state else detected
