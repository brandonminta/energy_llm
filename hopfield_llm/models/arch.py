"""Architecture resolution utilities for HF causal LMs.

Resolves the backbone sub-module and the layer list for any model family
supported by this project (Qwen, Phi, Mistral, Llama, Gemma, GPT-style,
GPT-NeoX/Pythia).

These functions are separated from the loader so they can be tested
independently with lightweight nn.Module stubs and extended when new
architecture families are added.
"""

from __future__ import annotations

import torch.nn as nn

# ---------------------------------------------------------------------------
# Architecture lookup tables
# ---------------------------------------------------------------------------

# Each entry is an ordered attribute path walked from the top-level model.
# The first path that fully resolves to an nn.Module wins.
BACKBONE_PATHS: list[list[str]] = [
    ["model"],        # Qwen2.5, Llama-3, Phi-3, Gemma-2, Mistral
    ["transformer"],  # GPT-2 / GPT-J style
    ["gpt_neox"],     # GPT-NeoX / Pythia
]

# Attribute names tried on the resolved backbone to find the layer list.
# The first name that resolves to a sized sequence wins.
LAYER_ATTRS: list[str] = [
    "layers",   # Qwen2.5, Llama, Phi-3, Gemma, Mistral
    "h",        # GPT-2 / BLOOM
    "block",    # T5 encoder/decoder
    "blocks",   # generic fallback
]


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def resolve_backbone(model: nn.Module) -> nn.Module:
    """Return the backbone sub-module of *model*.

    Tries each path in :data:`BACKBONE_PATHS` in order and returns the first
    one that resolves to an ``nn.Module``.

    Raises
    ------
    AttributeError
        If no path resolves successfully.
    """
    for path in BACKBONE_PATHS:
        obj: object = model
        found = True
        for attr in path:
            if not hasattr(obj, attr):
                found = False
                break
            obj = getattr(obj, attr)
        if found and isinstance(obj, nn.Module):
            return obj  # type: ignore[return-value]

    available = [a for a in dir(model) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot resolve backbone for {model.__class__.__name__}. "
        f"Top-level attrs: {available}"
    )


def resolve_layers(backbone: nn.Module) -> nn.ModuleList:
    """Return the transformer layer list from *backbone*.

    Tries each attribute name in :data:`LAYER_ATTRS` in order and returns
    the first one that is a sized sequence.

    Raises
    ------
    AttributeError
        If no attribute resolves to a sized sequence.
    """
    for attr in LAYER_ATTRS:
        if hasattr(backbone, attr):
            candidate = getattr(backbone, attr)
            try:
                _ = len(candidate)
                return candidate  # type: ignore[return-value]
            except TypeError:
                pass

    available = [a for a in dir(backbone) if not a.startswith("_")]
    raise AttributeError(
        f"Cannot resolve layer list in backbone {backbone.__class__.__name__}. "
        f"Available attrs: {available}"
    )
