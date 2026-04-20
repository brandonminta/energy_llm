"""MLP memory-bank extraction from transformer layer weights.

A memory bank is a per-layer matrix of shape [K, D] whose rows are candidate
patterns for Hopfield retrieval.  The default hypothesis uses W_down (the
down-projection of the SwiGLU MLP), but other projections or combinations can
be selected via the ``bank`` argument.

Swapping the ``bank`` argument changes the research hypothesis about *what
constitutes memory* without touching hooks, metrics, or the pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from hopfield_llm.utils.logging import get_logger

log = get_logger("memory.banks")

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


# ------------------------------------------------------------------
# Weight extraction helpers
# ------------------------------------------------------------------


def _get_weight_float32(module: Any) -> torch.Tensor:
    """Extract a linear module's weight as float32 on CPU.

    Supports: nn.Linear, bitsandbytes Linear4bit, Linear8bitLt.

    Returns:
        Tensor [out_features, in_features] in float32 on CPU.
    """
    if bnb is not None and isinstance(module, bnb.nn.Linear4bit):
        return (
            bnb.functional.dequantize_4bit(
                module.weight.data, module.weight.quant_state
            )
            .to(torch.float32)
            .cpu()
        )
    if bnb is not None and isinstance(module, bnb.nn.Linear8bitLt):
        return module.weight.data.to(torch.float32).cpu()
    if hasattr(module, "weight") and module.weight is not None:
        return module.weight.detach().to(torch.float32).cpu()
    raise TypeError(f"Cannot extract weight from {type(module).__name__}")


def _normalize_rows(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return w / w.norm(dim=1, keepdim=True).clamp_min(eps)


def _first_existing_attr(
    obj: Any, candidates: Iterable[str], role: str
) -> tuple[str, Any]:
    for name in candidates:
        if hasattr(obj, name):
            return name, getattr(obj, name)
    attrs = [a for a in dir(obj) if "proj" in a or "fc" in a or a.startswith("w")]
    raise AttributeError(
        f"No attribute for role='{role}'. Tried={tuple(candidates)}. Found={attrs}"
    )


def _resolve_proj(mlp: Any, bank: str) -> tuple[str, Any]:
    """Resolve an MLP sub-module by semantic role name."""
    if bank == "gate":
        return _first_existing_attr(mlp, ("gate_proj", "fc1", "w1"), role="gate")
    if bank == "up":
        return _first_existing_attr(mlp, ("up_proj", "w3", "fc1"), role="up")
    if bank == "down":
        return _first_existing_attr(mlp, ("down_proj", "w2", "fc2"), role="down")
    explicit = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3"}
    if bank in explicit:
        return _first_existing_attr(mlp, (bank,), role=bank)
    raise ValueError(f"Unknown bank='{bank}'")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def extract_banks(
    llm,
    bank: str = "down",
    normalize: bool = False,
    strict: bool = True,
    return_metadata: bool = False,
) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    """Extract per-layer memory banks from the model MLP projections.

    Supported ``bank`` values
    -------------------------
    Single projections:
        gate, up, down, gate_proj, up_proj, down_proj, fc1, fc2, w1, w2, w3

    Composite banks:
        gate_plus_up   — normalised average of W_gate and W_up rows
        gate_up_concat — row concatenation [W_gate; W_up]

    Args:
        llm:             HFLLM instance with resolved ``layers`` attribute.
        bank:            Which projection(s) to use as the memory bank.
        normalize:       Apply row-wise L2 normalisation to bank rows.
        strict:          Raise on per-layer errors (True) or log them (False).
        return_metadata: Also return a per-layer metadata dict.

    Returns:
        banks: dict[layer_idx → Tensor[K, D]]
        metadata (optional): dict[layer_idx → info dict]
    """
    banks: dict[int, torch.Tensor] = {}
    metadata: dict[int, dict[str, Any]] = {}

    config_hidden = getattr(llm.model.config, "hidden_size", None)
    config_intermediate = getattr(llm.model.config, "intermediate_size", None)

    for i, layer in enumerate(llm.layers):
        try:
            if not hasattr(layer, "mlp"):
                raise AttributeError(f"Layer {i} has no 'mlp' attribute")
            mlp = layer.mlp

            if bank in {
                "gate", "up", "down",
                "gate_proj", "up_proj", "down_proj",
                "fc1", "fc2", "w1", "w2", "w3",
            }:
                source_name, module = _resolve_proj(mlp, bank)
                w = _get_weight_float32(module)
                source = source_name

            elif bank == "gate_plus_up":
                gname, gmod = _resolve_proj(mlp, "gate")
                uname, umod = _resolve_proj(mlp, "up")
                wg = _get_weight_float32(gmod)
                wu = _get_weight_float32(umod)
                if wg.shape != wu.shape:
                    raise ValueError(f"Layer {i}: gate/up shape mismatch {wg.shape} vs {wu.shape}")
                wg_n = _normalize_rows(wg)
                del wg
                wu_n = _normalize_rows(wu)
                del wu
                wg_n.add_(wu_n).mul_(0.5)
                del wu_n
                w = _normalize_rows(wg_n)
                del wg_n
                source = f"norm(0.5*(norm({gname})+norm({uname})))"

            elif bank == "gate_up_concat":
                gname, gmod = _resolve_proj(mlp, "gate")
                uname, umod = _resolve_proj(mlp, "up")
                wg = _get_weight_float32(gmod)
                wu = _get_weight_float32(umod)
                if wg.shape[1] != wu.shape[1]:
                    raise ValueError(f"Layer {i}: hidden dim mismatch {wg.shape} vs {wu.shape}")
                w = torch.cat([wg, wu], dim=0)
                source = f"concat({gname}, {uname})"

            else:
                raise ValueError(
                    f"Unknown bank='{bank}'. Supported: gate, up, down, "
                    "gate_plus_up, gate_up_concat, gate_proj, up_proj, down_proj, "
                    "fc1, fc2, w1, w2, w3"
                )

            if w.ndim != 2:
                raise ValueError(f"Layer {i}: expected 2D weight, got shape={tuple(w.shape)}")

            out_features, in_features = w.shape
            if bank in {"gate", "up", "gate_proj", "up_proj", "fc1", "w1", "w3",
                        "gate_plus_up", "gate_up_concat"}:
                if config_hidden is not None and in_features != config_hidden:
                    raise ValueError(
                        f"Layer {i}: in_features={in_features} != hidden_size={config_hidden}"
                    )
            if bank in {"gate", "up", "gate_proj", "up_proj", "fc1", "w1", "w3"}:
                if config_intermediate is not None and out_features != config_intermediate:
                    raise ValueError(
                        f"Layer {i}: out_features={out_features} != intermediate_size={config_intermediate}"
                    )

            if normalize and bank != "gate_plus_up":
                w = _normalize_rows(w)

            banks[i] = w
            metadata[i] = {
                "layer_idx": i,
                "bank": bank,
                "source": source,
                "shape": tuple(w.shape),
                "dtype": str(w.dtype),
                "K": int(w.shape[0]),
                "D": int(w.shape[1]),
            }

        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"Failed extracting bank='{bank}' at layer {i}: {exc}"
                ) from exc
            metadata[i] = {"layer_idx": i, "bank": bank, "error": str(exc)}
            log.warning("Layer %d: %s", i, exc)

    log.info(
        "Extracted bank='%s' from %d/%d layers (normalize=%s)",
        bank, len(banks), len(llm.layers), normalize,
    )
    if return_metadata:
        return banks, metadata
    return banks
