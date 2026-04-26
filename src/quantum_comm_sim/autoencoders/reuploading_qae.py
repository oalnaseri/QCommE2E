"""Data re-uploading QAE."""

import pennylane as qml
import numpy as np
from .base_qae import BaseQAE


class ReuploadingQAE(BaseQAE):
    """Data re-uploading QAE for enhanced expressibility.
    Reference: https://arxiv.org/abs/1904.12260
    """

    def __init__(self, num_qubits: int = 2, num_layers: int = 6):
        super().__init__(num_qubits, num_layers)
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.randn(num_layers, num_qubits, 2) * 0.1

    def _circuit(self, params, state):
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RY(inputs[q] * weights[l, q, 0], wires=q)
                    qml.RZ(weights[l, q, 1], wires=q)
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.density_matrix(wires=0)
        return circuit(state.flatten(), params)

    def encode(self, state):
        return self._circuit(self.params, state)

    def decode(self, latent):
        return latent
