"""Structured logging for hopfield_llm."""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the package. Call once at entry point."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger("hopfield_llm")
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the hopfield_llm namespace."""
    return logging.getLogger(f"hopfield_llm.{name}")
