"""Gradient-based end-to-end training of TX/RX parameters."""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Dict, Any
from ..utils.logger import setup_logger

logger = setup_logger()


class E2ETrainer(nn.Module):
    """Joint TX/RX module for differentiable E2E optimization.

    Wraps transmitter and receiver blocks into a PyTorch nn.Module
    so that gradients flow through both modulation/encoding and
    decoding/demodulation jointly.
    """

    def __init__(self,
                 tx_module: nn.Module,
                 channel_fn: Callable,
                 rx_module: nn.Module,
                 objective: str = "neg_gmi"):
        super().__init__()
        self.tx_module = tx_module
        self.channel_fn = channel_fn
        self.rx_module = rx_module
        self.objective = objective

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        tx_signal = self.tx_module(symbols)
        rx_signal = self.channel_fn(tx_signal)
        out = self.rx_module(rx_signal)
        return out

    def fit(self,
            num_symbols: int,
            epochs: int = 100,
            lr: float = 1e-3,
            seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            symbols = torch.randint(0, 4, (num_symbols,))
            out = self.forward(symbols)
            loss = self._compute_loss(symbols, out)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")
        return {"history": history}

    def _compute_loss(self, tx: torch.Tensor, rx: torch.Tensor) -> torch.Tensor:
        if self.objective == "neg_ser":
            return torch.mean((tx != rx).float())
        # Default: MSE surrogate for neg GMI
        return torch.nn.functional.mse_loss(rx.float(), tx.float())
