# Quantum Communication Simulator

End-to-End Quantum Communication System (QCommE2E) focusing on quantum channel modeling, transceiver components, and learned compensation via quantum autoencoders (QAE).

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
conda activate QCommE2E
pip install -e ".[dev]"

python experiments/fso_turbulence_e2e/run.py
pytest tests/ -v
```

## Repository Structure

```
QCommE2E/
├── src/quantum_comm_sim/       # Core library
├── config/                       # YAML experiment configs
├── experiments/                  # Reproducible experiment scripts
├── notebooks/                    # Tutorials
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

## If you use this code, please cite the following paper

Fading-Memory Quaternion-Based Kalman Filter for Quantum Channel PMD Compensation <https://doi.org/10.1109/ACCESS.2026.3675938>

```
@ARTICLE{alnaseri2026fading,
  author={Alnaseri, Omar and Himeur, Yassine and Titouni, Salem and Timmermann, Jens and Atalla, Shadi},
  journal={IEEE Access}, 
  title={Fading-Memory Quaternion-Based Kalman Filter for Quantum Channel PMD Compensation}, 
  year={2026},
  volume={14},
  number={},
  pages={46943-46952},
  doi={10.1109/ACCESS.2026.3675938}}
