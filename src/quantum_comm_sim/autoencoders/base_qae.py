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
