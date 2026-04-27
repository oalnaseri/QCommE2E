"""
Academic References:
  - Nielsen, M.A. & Chuang, I.L., Quantum Computation and Quantum Information,
    Cambridge University Press, 2010. Ch. 8 (Quantum channels).
  - Bennett, C.H. et al., Capacities of quantum erasure channels,
    Phys. Rev. Lett., 78(16), 3217, 1997.
"""

"""Experiment: Baseline comparison across quantum channels."""

import numpy as np
from quantum_comm_sim.channels.depolarizing_channel import DepolarizingChannel
from quantum_comm_sim.channels.dephasing_channel import DephasingChannel
from quantum_comm_sim.channels.bosonic_channel import BosonicChannel
from quantum_comm_sim.transceiver.modulators import QPSKModulator
from quantum_comm_sim.transceiver.transmitter import Transmitter
from quantum_comm_sim.transceiver.receiver import Receiver
from quantum_comm_sim.transceiver.detectors import HomodyneDetector
from quantum_comm_sim.pipeline.simulator import Simulator


def main():
    np.random.seed(42)
    mod = QPSKModulator(dim=2)
    tx = Transmitter(modulator=mod)
    det = HomodyneDetector(threshold=0.0)
    rx = Receiver(detector=det)
    channels = {
        "depolarizing": DepolarizingChannel(p=0.1),
        "dephasing": DephasingChannel(p=0.1),
        "bosonic": BosonicChannel(dim=2, loss_db=3.0),
    }
    for name, ch in channels.items():
        sim = Simulator(tx, ch, rx, num_symbols=5000, seed=42)
        results = sim.run()
        print(f"[{name}] BER={results['ber']:.4e}, SER={results['ser']:.4e}")


if __name__ == "__main__":
    main()
