"""
Academic References:
  - Poole, C.D. & Wagner, R.E., Phenomenological approach to polarization dispersion
    in long single-mode fibers, Electronics Letters, 22(19), 1029-1030, 1986.
  - IntechOpen, Applications of Kalman Filters for Coherent Optical Communication Systems,
    2017.
  - Gordon, J.P. & Kogelnik, H., PMD fundamentals, PNAS, 97(9), 4541-4550, 2000.
"""

"""Experiment: Fiber PMD with QAE compensation."""

import numpy as np
from quantum_comm_sim.channels.pmd_channel import PMDChannel
from quantum_comm_sim.channels.bosonic_channel import BosonicChannel
from quantum_comm_sim.channels.composite_channel import CompositeChannel
from quantum_comm_sim.transceiver.modulators import QPSKModulator
from quantum_comm_sim.transceiver.transmitter import Transmitter
from quantum_comm_sim.transceiver.receiver import Receiver
from quantum_comm_sim.transceiver.detectors import HomodyneDetector
from quantum_comm_sim.equalization.kalman_equalizer import KalmanEqualizer
from quantum_comm_sim.pipeline.simulator import Simulator


def main():
    np.random.seed(42)
    mod = QPSKModulator(dim=2)
    tx = Transmitter(modulator=mod)
    pmd = PMDChannel(dgd_mean_ps=1.0, num_sections=10)
    bos = BosonicChannel(dim=2, loss_db=0.2, noise_factor=0.005)
    ch = CompositeChannel([pmd, bos])
    det = HomodyneDetector(threshold=0.0)
    eq = KalmanEqualizer(quaternion_enabled=True, adaptation_rate=0.01)
    rx = Receiver(detector=det, equalizer=eq)
    sim = Simulator(tx, ch, rx, num_symbols=5000, seed=42)
    results = sim.run()
    print(f"BER={results['ber']:.4e}, SER={results['ser']:.4e}, GMI={results['gmi']:.4f}")


if __name__ == "__main__":
    main()
