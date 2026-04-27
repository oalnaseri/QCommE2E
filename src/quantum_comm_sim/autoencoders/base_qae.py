"""
Academic References:
  - Romero, J. et al., Quantum autoencoders for efficient compression of quantum data,
    Quantum Sci. Technol., 2(4), 045001, 2017.
  - Bravo-Prieto, C. et al., Quantum autoencoders with enhanced data encoding,
    arXiv:2010.06599, 2020.
  - Bergholm, V. et al., PennyLane: Automatic differentiation of hybrid
    quantum-classical computations, arXiv:1811.04968, 2018.
"""

"""Base class for quantum autoencoders."""

import pennylane as qml
from abc import ABC, abstractmethod


class BaseQAE(ABC):
    """Abstract quantum autoencoder compensator.
    Designed to be used inside the receiver pipeline.
    """

    def __init__(self, num_qubits: int = 2, num_layers: int = 4):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    @abstractmethod
    def encode(self, state):
        """Compress quantum state into latent subspace."""

    @abstractmethod
    def decode(self, latent):
        """Reconstruct state from latent subspace."""
