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
from matplotlib.colors import to_hex

from quantum_comm_sim.utils.bloch import BlochVisualizer


SYMBOL_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    -1: "tab:gray",
}


def _resolve_symbol_palette(labels=None, colors=None):
    """Build a stable symbol-to-color mapping for the provided labels."""
    palette = dict(SYMBOL_COLORS)
    if colors is not None:
        palette.update(colors)

    if labels is None:
        return palette

    label_array = np.asarray(labels)
    unique_labels = sorted({int(label) for label in label_array})
    valid_labels = [label for label in unique_labels if label >= 0]

    if valid_labels:
        cmap_name = "tab20" if len(valid_labels) <= 20 else "gist_ncar"
        cmap = plt.get_cmap(cmap_name, len(valid_labels))
        for index, symbol in enumerate(valid_labels):
            palette.setdefault(symbol, cmap(index))

    if -1 in unique_labels:
        palette.setdefault(-1, SYMBOL_COLORS[-1])

    return {symbol: to_hex(color, keep_alpha=True) for symbol, color in palette.items()}


def _sample_plot_indices(sample_count: int, max_points: int, labels=None) -> np.ndarray:
    """Sample plot indices while preserving symbol coverage when labels exist."""
    if sample_count == 0:
        return np.array([], dtype=int)

    keep = min(max_points, sample_count)
    if keep == sample_count or labels is None:
        return np.random.choice(sample_count, size=keep, replace=False)

    label_array = np.asarray(labels)
    if len(label_array) != sample_count:
        raise ValueError(
            f"Label count {len(label_array)} does not match point count {sample_count}"
        )

    selected = []
    unique_labels = sorted(np.unique(label_array))
    per_label = max(1, keep // max(len(unique_labels), 1))

    for symbol in unique_labels:
        symbol_indices = np.flatnonzero(label_array == symbol)
        if len(symbol_indices) == 0:
            continue
        take = min(per_label, len(symbol_indices), keep - len(selected))
        if take > 0:
            selected.extend(np.random.choice(symbol_indices, size=take, replace=False).tolist())
        if len(selected) == keep:
            break

    if len(selected) < keep:
        remaining = np.setdiff1d(np.arange(sample_count), np.array(selected, dtype=int))
        take = min(keep - len(selected), len(remaining))
        if take > 0:
            selected.extend(np.random.choice(remaining, size=take, replace=False).tolist())

    return np.sort(np.array(selected[:keep], dtype=int))


def _extract_qubit_block(states: np.ndarray) -> np.ndarray:
    """Extract the leading 2x2 qubit block from a density matrix batch."""
    states = np.asarray(states)

    if states.ndim == 3 and states.shape[1] == states.shape[2] and states.shape[1] >= 2:
        return states[:, :2, :2]

    if states.ndim == 2 and states.shape[0] == states.shape[1] and states.shape[0] >= 2:
        return states[:2, :2]

    return states


def _qam_points_from_qubit_density(states: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Recover approximate complex QAM points from qubit density matrices.

    For QAM states embedded as psi ~ [1, alpha], an ideal pure state satisfies
    alpha = rho[1, 0] / rho[0, 0]. We use the same ratio as a robust plotting
    proxy for mildly mixed outputs and clip extremes for stability.
    """
    rho = np.asarray(_extract_qubit_block(states), dtype=complex)
    if rho.ndim == 2:
        rho = rho[None, ...]
    if rho.ndim != 3 or rho.shape[1:] != (2, 2):
        raise ValueError("QAM projection expects shape (N, 2, 2) or (2, 2)")

    denom = np.clip(np.real(rho[:, 0, 0]), 1.0e-9, None)
    alpha = rho[:, 1, 0] / denom
    alpha = np.clip(alpha.real, -clip, clip) + 1j * np.clip(alpha.imag, -clip, clip)
    return np.column_stack([alpha.real, alpha.imag])


def states_to_constellation(states: np.ndarray) -> np.ndarray:
    """Project simulator states into 2D constellation coordinates."""
    states = np.asarray(_extract_qubit_block(states))

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
    """Map density matrices to Bloch vectors using the leading qubit block."""
    return BlochVisualizer.density_to_bloch(_extract_qubit_block(rho_batch))


def _resolve_bloch_class():
    try:
        module = importlib.import_module("qutip")
    except ImportError:
        return None
    return module.Bloch


def _render_bloch_axis(ax, vectors: np.ndarray, title: str, point_colors=None):
    """Render a Bloch sphere with optional per-point colors using Matplotlib."""
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color="0.7", linewidth=0.6, alpha=0.3)
    ax.plot([-1, 1], [0, 0], [0, 0], color="0.35", linewidth=0.8, alpha=0.6)
    ax.plot([0, 0], [-1, 1], [0, 0], color="0.35", linewidth=0.8, alpha=0.6)
    ax.plot([0, 0], [0, 0], [-1, 1], color="0.35", linewidth=0.8, alpha=0.6)

    ax.scatter(
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        s=44,
        c=point_colors,
        alpha=0.9,
        edgecolors="none",
        depthshade=False,
    )

    ax.text(1.1, 0.0, 0.0, "x", fontsize=10)
    ax.text(0.0, 1.1, 0.0, "y", fontsize=10)
    ax.text(0.0, 0.0, 1.12, r"$|0\rangle$", fontsize=11, ha="center")
    ax.text(0.0, 0.0, -1.18, r"$|1\rangle$", fontsize=11, ha="center")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=35)
    ax.set_title(title)


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

    label_array = None if labels is None else np.asarray(labels)
    use_qam_projection = False
    if label_array is not None:
        valid_labels = label_array[label_array >= 0]
        use_qam_projection = len(np.unique(valid_labels)) > 4

    if use_qam_projection:
        try:
            points = _qam_points_from_qubit_density(symbols)
        except ValueError:
            points = states_to_constellation(symbols)
    else:
        points = states_to_constellation(symbols)

    x = points[:, 0]
    y = points[:, 1]

    if labels is None:
        ax.scatter(x, y, s=5, alpha=0.6)
    else:
        labels = label_array
        if len(labels) != len(points):
            raise ValueError(
                f"Label count {len(labels)} does not match point count {len(points)}"
            )
        palette = _resolve_symbol_palette(labels, colors)
        for symbol in sorted(np.unique(labels)):
            mask = labels == symbol
            ax.scatter(
                x[mask],
                y[mask],
                s=10,
                alpha=0.55,
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
    labels=None,
    colors=None,
):
    """Plot TX and RX Bloch spheres side by side with shared symbol colors."""

    tx = np.asarray(tx_states)
    rx = np.asarray(rx_states)
    if tx.ndim == 2:
        tx = tx[None, ...]
        rx = rx[None, ...]

    sample_count = len(tx)
    if sample_count == 0:
        return None

    label_array = None if labels is None else np.asarray(labels)
    indices = _sample_plot_indices(sample_count, max_points, label_array)
    tx_vecs = density_to_bloch(tx[indices])
    rx_vecs = density_to_bloch(rx[indices])

    point_colors = None
    if label_array is not None:
        palette = _resolve_symbol_palette(label_array, colors)
        sampled_labels = label_array[indices]
        point_colors = [palette.get(int(symbol), "tab:gray") for symbol in sampled_labels]

    fig = plt.figure(figsize=(10, 4), dpi=180)
    ax_tx = fig.add_subplot(1, 2, 1, projection="3d")
    ax_rx = fig.add_subplot(1, 2, 2, projection="3d")

    _render_bloch_axis(ax_tx, tx_vecs, f"{title_prefix} - TX Bloch sphere", point_colors)
    _render_bloch_axis(ax_rx, rx_vecs, f"{title_prefix} - RX Bloch sphere", point_colors)
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
