"""
Academic References:
  - Python Logging HOWTO, https://docs.python.org/3/howto/logging.html
"""

"""Logging utilities."""

import logging
import sys


def setup_logger(name: str = "quantum_comm_sim", level: int = logging.INFO):
    """Configure standard logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
