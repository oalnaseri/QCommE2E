"""End-to-end simulation runner."""

import numpy as np
from typing import Optional, Dict, Any
from ..channels.base_channel import BaseChannel
from ..transceiver.transmitter import Transmitter
from ..transceiver.receiver import Receiver
from ..metrics.ber_ser import compute_ber, compute_ser
from ..metrics.mi_gmi import estimate_gmi
from ..utils.logger import setup_logger

logger = setup_logger()


class Simulator:
    """Run a single E2E simulation sweep."""

    def __init__(self,
                 transmitter: Transmitter,
                 channel: BaseChannel,
                 receiver: Receiver,
                 num_symbols: int = 10000,
                 seed: Optional[int] = None):
        self.transmitter = transmitter
        self.channel = channel
        self.receiver = receiver
        self.num_symbols = num_symbols
        self.seed = seed

    def run(self) -> Dict[str, Any]:
        if self.seed is not None:
            np.random.seed(self.seed)
        symbols = np.random.randint(0, 4, size=self.num_symbols)
        tx_states = self.transmitter.transmit(symbols)
        rx_states = []
        for state in tx_states:
            rx_states.append(self.channel.apply(state))
        rx_states = np.array(rx_states)
        rx_symbols = self.receiver.receive(rx_states)
        ber = compute_ber(symbols % 2, rx_symbols % 2)
        ser = compute_ser(symbols, rx_symbols)
        gmi = estimate_gmi(symbols, rx_symbols)
        logger.info(f"BER={ber:.4e}, SER={ser:.4e}, GMI={gmi:.4f}")
        return {
            "tx_symbols": symbols,
            "rx_symbols": rx_symbols,
            "ber": ber,
            "ser": ser,
            "gmi": gmi,
        }
