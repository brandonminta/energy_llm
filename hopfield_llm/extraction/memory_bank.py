"""MLP memory-bank extraction from transformer layers."""

from __future__ import annotations

from typing import Any, Iterable

import torch

from hopfield_llm.utils.logging import get_logger

log = get_logger("extraction.memory_bank")

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_linear_weight_as_float32(module: Any) -> torch.Tensor:
    """Extract weight from a linear module as float32 tensor on CPU.

    Supports torch.nn.Linear, bitsandbytes Linear4bit, and Linear8bitLt.

    Returns:
        Tensor of shape [out_features, in_features] in float32 on CPU.
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

    raise TypeError(f"Cannot extract weight from module type {type(module).__name__}")


def _normalize_rows(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise L2 normalization."""
    return w / w.norm(dim=1, keepdim=True).clamp_min(eps)


def _get_first_existing_attr(
    obj: Any, candidates: Iterable[str], role: str
) -> tuple[str, Any]:
    """Return (name, value) of the first existing attribute from *candidates*."""
    for name in candidates:
        if hasattr(obj, name):
            return name, getattr(obj, name)

    attrs = [a for a in dir(obj) if ("proj" in a or "fc" in a or a.startswith("w"))]
    raise AttributeError(
        f"No compatible attribute for role='{role}'. "
        f"Candidates={tuple(candidates)}. Found={attrs}"
    )


def _resolve_mlp_proj_by_bank(mlp: Any, bank: str) -> tuple[str, Any]:
    """Resolve the base projection by semantic bank role.

    Semantic banks:
        gate -> gate_proj / fc1 / w1
        up   -> up_proj / w3 / fc1
        down -> down_proj / w2 / fc2

    Explicit names:
        fc1, fc2, w1, w2, w3, gate_proj, up_proj, down_proj
    """
    if bank == "gate":
        return _get_first_existing_attr(mlp, ("gate_proj", "fc1", "w1"), role="gate")
    if bank == "up":
        return _get_first_existing_attr(mlp, ("up_proj", "w3", "fc1"), role="up")
    if bank == "down":
        return _get_first_existing_attr(mlp, ("down_proj", "w2", "fc2"), role="down")

    explicit = {
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2", "w1", "w2", "w3",
    }
    if bank in explicit:
        return _get_first_existing_attr(mlp, (bank,), role=bank)

    raise ValueError(f"Unsupported bank type: '{bank}'")


# ------------------------------------------------------------------
# Main extraction
# ------------------------------------------------------------------


def extract_mlp_memory_bank(
    llm,
    bank: str = "gate",
    normalize: bool = False,
    strict: bool = True,
    return_metadata: bool = False,
) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    """Extract a per-layer memory bank from the model MLP projections.

    Supported banks:
        gate, up, down              — single projection
        gate_plus_up                — norm(0.5*(norm(W_gate) + norm(W_up)))
        gate_up_concat              — row concatenation [W_gate; W_up]
        gate_proj, up_proj, etc.    — explicit projection names

    Args:
        llm: HFLLM instance.
        bank: Bank type to extract.
        normalize: Apply row-wise L2 normalization.
        strict: Raise on per-layer errors (True) or store error in metadata (False).
        return_metadata: Also return a per-layer metadata dict.

    Returns:
        banks: dict mapping layer index (0-indexed) to Tensor [K, D].
        metadata (optional): dict mapping layer index to info dict.
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
                source_name, module = _resolve_mlp_proj_by_bank(mlp, bank)
                w = _get_linear_weight_as_float32(module)
                source = source_name

            elif bank == "gate_plus_up":
                gate_name, gate_mod = _resolve_mlp_proj_by_bank(mlp, "gate")
                up_name, up_mod = _resolve_mlp_proj_by_bank(mlp, "up")
                w_gate = _get_linear_weight_as_float32(gate_mod)
                w_up = _get_linear_weight_as_float32(up_mod)
                if w_gate.shape != w_up.shape:
                    raise ValueError(
                        f"Layer {i}: gate/up shape mismatch: {w_gate.shape} vs {w_up.shape}"
                    )
                w = _normalize_rows(
                    0.5 * (_normalize_rows(w_gate) + _normalize_rows(w_up))
                )
                source = f"norm(0.5*(norm({gate_name})+norm({up_name})))"

            elif bank == "gate_up_concat":
                gate_name, gate_mod = _resolve_mlp_proj_by_bank(mlp, "gate")
                up_name, up_mod = _resolve_mlp_proj_by_bank(mlp, "up")
                w_gate = _get_linear_weight_as_float32(gate_mod)
                w_up = _get_linear_weight_as_float32(up_mod)
                if w_gate.shape[1] != w_up.shape[1]:
                    raise ValueError(
                        f"Layer {i}: gate/up hidden dim mismatch: {w_gate.shape} vs {w_up.shape}"
                    )
                w = torch.cat([w_gate, w_up], dim=0)
                source = f"concat({gate_name}, {up_name})"

            else:
                raise ValueError(
                    f"Unknown bank: '{bank}'. Use one of: gate, up, down, "
                    "gate_plus_up, gate_up_concat, gate_proj, up_proj, down_proj, "
                    "fc1, fc2, w1, w2, w3"
                )

            if w.ndim != 2:
                raise ValueError(
                    f"Layer {i}: expected 2D tensor, got shape={tuple(w.shape)}"
                )

            out_features, in_features = w.shape

            if bank in {
                "gate", "up", "gate_proj", "up_proj",
                "fc1", "w1", "w3", "gate_plus_up", "gate_up_concat",
            }:
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
                "device": str(w.device),
                "K": int(w.shape[0]),
                "D": int(w.shape[1]),
            }

        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"Failed extracting bank='{bank}' at layer {i}: {e}"
                ) from e
            metadata[i] = {"layer_idx": i, "bank": bank, "error": str(e)}

    log.info(
        "Extracted bank='%s' from %d/%d layers (normalize=%s)",
        bank, len(banks), len(llm.layers), normalize,
    )

    if return_metadata:
        return banks, metadata
    return banks
