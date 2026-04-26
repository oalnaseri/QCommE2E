"""Variational QAE with fully parameterized ansatz."""

import pennylane as qml
import numpy as np
from .base_qae import BaseQAE


class VariationalQAE(BaseQAE):
    """Fully variational QAE using a hardware-efficient ansatz."""

    def __init__(self, num_qubits: int = 4, num_layers: int = 4, latent_dim: int = 2):
        super().__init__(num_qubits, num_layers)
        self.latent_dim = latent_dim
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.randn(num_layers, num_qubits, 3) * 0.1

    def _circuit(self, params, state):
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                for q in range(0, self.num_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(1, self.num_qubits - 1, 2):
                    qml.CNOT(wires=[q, q + 1])
            return qml.density_matrix(wires=range(self.latent_dim))
        return circuit(state.flatten(), params)

    def encode(self, state):
        return self._circuit(self.params, state)

    def decode(self, latent):
        return latent
