"""Utility helpers for extraction."""

from __future__ import annotations

import torch


def align_banks_to_layers(
    banks: dict[int, torch.Tensor],
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Build a position-indexed bank dict aligned to 1-indexed layer_indices.

    Banks are keyed by 0-indexed layer number.  Hidden states use 1-indexed
    layer_indices (1 = first transformer layer).  This function maps position
    *i* in the layer_indices list to ``banks[layer_indices[i] - 1]``.

    Returns:
        Dict mapping position index to bank tensor, only for layers present
        in both *banks* and *layer_indices*.
    """
    return {
        i: banks[li - 1]
        for i, li in enumerate(layer_indices)
        if (li - 1) in banks
    }
