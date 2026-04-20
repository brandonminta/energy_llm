"""Query extraction strategies for hook-captured activations.

A query extractor maps a full activation tensor [T, d_m] to a single query
vector [d_m].  The choice of extractor encodes the hypothesis about *which*
part of the activation sequence should act as the retrieval query against the
memory bank.

Changing the extractor changes the research hypothesis without touching model
loading, hook registration, or metric computation.

Usage
-----
Pass any extractor (or a custom callable) to ``capture_prefill`` via
``query_fn``.  The signature must be:

    def my_extractor(h: torch.Tensor) -> torch.Tensor:
        # h: [T, d_m] — full per-layer activation at a given position
        # returns: [d_m]
        ...
"""

from __future__ import annotations

from typing import Callable, Protocol

import torch


class QueryExtractor(Protocol):
    """Protocol for query extraction callables."""

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """Extract a 1-D query from a 2-D activation tensor.

        Args:
            h: Activation tensor of shape [T, d_m].

        Returns:
            Query vector of shape [d_m].
        """
        ...


# ------------------------------------------------------------------
# Built-in extractors
# ------------------------------------------------------------------


def last_token(h: torch.Tensor) -> torch.Tensor:
    """Select the last token's activation.  [T, d_m] → [d_m]

    Default strategy during prefill: the last question token encodes the
    full context and acts as the retrieval cue.
    """
    return h[-1]


def first_token(h: torch.Tensor) -> torch.Tensor:
    """Select the first token's activation.  [T, d_m] → [d_m]"""
    return h[0]


def mean_tokens(h: torch.Tensor) -> torch.Tensor:
    """Mean-pool all tokens.  [T, d_m] → [d_m]

    Alternative: treats the question as a bag of positions.
    """
    return h.mean(dim=0)


def make_positional(index: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return an extractor that selects a fixed token position.

    Args:
        index: Token index (supports negative indexing).

    Returns:
        Callable [T, d_m] → [d_m].
    """
    def _extractor(h: torch.Tensor) -> torch.Tensor:
        return h[index]
    _extractor.__name__ = f"positional[{index}]"
    return _extractor
