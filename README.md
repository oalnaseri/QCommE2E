# Quantum Communication Simulator

End-to-End Quantum Communication System Simulator focusing on quantum channel modeling, transceiver components, and learned compensation via quantum autoencoders (QAE).

## Features

- **Quantum Channels**: Bosonic Gaussian, depolarizing, dephasing, erasure, PMD, FSO turbulence (Malaga/Kolmogorov), and composite cascades.
- **Transceiver Blocks**: State preparation (coherent, squeezed, OAM), modulation, POVM measurement, homodyne detection, photon counting.
- **Quantum Autoencoders**: Spatial-mode QAE, data re-uploading QAE, and variational QAE architectures for channel compensation.
- **Equalization**: Quaternion-based Kalman adaptive equalizer and neural (GNN/deep unfolding) equalizers.
- **Metrics**: BER/SER, mutual information (MI), generalized mutual information (GMI), quantum fidelity, entanglement measures.
- **E2E Training**: Joint optimization of transmitter and receiver parameters via gradient descent.

## Quick Start

```bash
conda env create -f environment.yml
conda activate quantum-comm-sim
pip install -e ".[dev]"

python experiments/fso_turbulence_e2e/run.py
pytest tests/ -v
```

## Repository Structure

```
quantum-comm-sim/
├── src/quantum_comm_sim/       # Core library
├── config/                       # YAML experiment configs
├── experiments/                  # Reproducible experiment scripts
├── notebooks/                    # Tutorials
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## License
MIT
