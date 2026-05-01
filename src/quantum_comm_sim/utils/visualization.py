"""
Academic References:
  - Hunter, J.D., Matplotlib: A 2D graphics environment, Computing in Science & Engineering,
    9(3), 90-95, 2007.
  - Hecht, N. et al., Constellation diagram analysis for optical communications,
    J. Lightwave Technol., 34(2), 533-541, 2016.
"""

"""Plotting utilities for constellations and metrics."""

import importlib

import matplotlib.pyplot as plt
import numpy as np

from quantum_comm_sim.utils.bloch import BlochVisualizer


SYMBOL_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
}


def states_to_constellation(states: np.ndarray) -> np.ndarray:
    """Project simulator states into 2D constellation coordinates."""
    states = np.asarray(states)

    if states.ndim == 3 and states.shape[1:] == (2, 2):
        in_phase = 2.0 * np.real(states[:, 0, 1])
        quadrature = np.real(states[:, 0, 0] - states[:, 1, 1])
        return np.column_stack([in_phase, quadrature])

    if states.ndim == 2 and states.shape == (2, 2):
        in_phase = 2.0 * np.real(states[0, 1])
        quadrature = np.real(states[0, 0] - states[1, 1])
        return np.array([[in_phase, quadrature]])

    if states.ndim == 2 and states.shape[0] == 2 and not np.iscomplexobj(states):
        return states.T

    if states.ndim == 2 and states.shape[1] == 2 and not np.iscomplexobj(states):
        return states

    if states.ndim == 1:
        symbol_map = np.array(
            [
                [0.0, 1.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        )
        return symbol_map[states.astype(int) % len(symbol_map)]

    if np.iscomplexobj(states):
        flat = states.reshape(-1)
        return np.column_stack([flat.real, flat.imag])

    raise ValueError(f"Unsupported shape for constellation plotting: {states.shape}")


def density_to_bloch(rho_batch: np.ndarray) -> np.ndarray:
    """Map 2x2 density matrices to Bloch vectors."""
    return BlochVisualizer.density_to_bloch(rho_batch)


def _resolve_bloch_class():
    try:
        module = importlib.import_module("qutip")
    except ImportError:
        return None
    return module.Bloch


def plot_constellation(*args, title: str = "Constellation", labels=None, colors=None):
    """Scatter plot of complex symbols or projected quantum states.

    Supported call patterns:
    - plot_constellation(symbols, title="...")
    - plot_constellation(ax, symbols, title, labels=...)
    """
    if len(args) == 0:
        raise TypeError("plot_constellation requires at least one positional argument")

    ax = None
    symbols = None
    plot_title = title

    if hasattr(args[0], "scatter"):
        if len(args) < 2:
            raise TypeError("plot_constellation(ax, symbols, ...) requires a symbols argument")
        ax = args[0]
        symbols = args[1]
        if len(args) >= 3:
            plot_title = args[2]
    else:
        symbols = args[0]
        if len(args) >= 2:
            plot_title = args[1]
        _, ax = plt.subplots()

    if symbols is None:
        ax.text(0.5, 0.5, "No constellation data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(plot_title)
        return ax.figure

    points = states_to_constellation(symbols)
    x = points[:, 0]
    y = points[:, 1]

    if labels is None:
        ax.scatter(x, y, s=5, alpha=0.6)
    else:
        labels = np.asarray(labels)
        if len(labels) != len(points):
            raise ValueError(
                f"Label count {len(labels)} does not match point count {len(points)}"
            )
        palette = colors or SYMBOL_COLORS
        for symbol in sorted(np.unique(labels)):
            mask = labels == symbol
            ax.scatter(
                x[mask],
                y[mask],
                s=8,
                alpha=0.45,
                color=palette.get(int(symbol), "tab:gray"),
            )

    ax.set_title(plot_title)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure


def plot_bloch_spheres(
    tx_states: np.ndarray,
    rx_states: np.ndarray,
    title_prefix: str,
    max_points: int = 200,
):
    """Plot TX and RX Bloch spheres side by side using QuTiP when available."""
    bloch_class = _resolve_bloch_class()
    if bloch_class is None:
        print(
            f"[Bloch] Skipping Bloch-sphere plot for {title_prefix} "
            "(QuTiP not available in this environment)."
        )
        return None

    tx = np.asarray(tx_states)
    rx = np.asarray(rx_states)
    if tx.ndim == 2:
        tx = tx[None, ...]
        rx = rx[None, ...]

    sample_count = len(tx)
    if sample_count == 0:
        return None

    keep = min(max_points, sample_count)
    indices = np.random.choice(sample_count, size=keep, replace=False)
    tx_vecs = density_to_bloch(tx[indices])
    rx_vecs = density_to_bloch(rx[indices])

    fig = plt.figure(figsize=(10, 4))
    ax_tx = fig.add_subplot(1, 2, 1, projection="3d")
    ax_rx = fig.add_subplot(1, 2, 2, projection="3d")

    tx_bloch = bloch_class(fig=fig, axes=ax_tx)
    rx_bloch = bloch_class(fig=fig, axes=ax_rx)
    tx_bloch.add_points(tx_vecs.T)
    rx_bloch.add_points(rx_vecs.T)
    tx_bloch.title = f"{title_prefix} - TX Bloch sphere"
    rx_bloch.title = f"{title_prefix} - RX Bloch sphere"
    tx_bloch.render()
    rx_bloch.render()
    plt.tight_layout()
    plt.show()
    return fig


def plot_ber_curve(snr_db: np.ndarray, ber: np.ndarray, label: str = "BER"):
    """Plot BER vs SNR curve."""
    plt.figure()
    plt.semilogy(snr_db, ber, marker='o', label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
