"""Experiment: FSO turbulence with QAE E2E training."""

import numpy as np
from quantum_comm_sim.channels.turbulence_channel import TurbulenceChannel
from quantum_comm_sim.transceiver.modulators import QPSKModulator
from quantum_comm_sim.transceiver.transmitter import Transmitter
from quantum_comm_sim.transceiver.receiver import Receiver
from quantum_comm_sim.transceiver.detectors import HomodyneDetector
from quantum_comm_sim.autoencoders.spatial_mode_qae import SpatialModeQAE
from quantum_comm_sim.pipeline.simulator import Simulator


def main():
    np.random.seed(42)
    mod = QPSKModulator(dim=2)
    tx = Transmitter(modulator=mod)
    ch = TurbulenceChannel(
        turbulence_model="malaga",
        alpha=2.5,
        beta=2.0,
        link_distance_m=1000.0,
        dim=2
    )
    qae = SpatialModeQAE(num_qubits=4, num_layers=4, latent_dim=2)
    det = HomodyneDetector(threshold=0.0)
    rx = Receiver(detector=det, decoder=qae)
    sim = Simulator(tx, ch, rx, num_symbols=2000, seed=42)
    results = sim.run()
    print(f"BER={results['ber']:.4e}, SER={results['ser']:.4e}, GMI={results['gmi']:.4f}")


if __name__ == "__main__":
    main()
