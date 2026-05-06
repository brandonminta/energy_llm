"""MLP memory-bank extraction from transformer layer weights.

A memory bank is a per-layer matrix of shape [K, D] whose rows are candidate
patterns for Hopfield retrieval.

KEY-SPACE PROBE (default, bank="up"):
  Query  = FFN input x^l  [d]
  Bank   = rows of W_up   [d_m, d]
  Scores = W_up @ x       [d_m]

VALUE-SPACE PROBE (bank="down_values"):
  Query  = FFN output y^l          [d]
  Bank   = rows of W_down.T        [d_m, d]   (columns of W_down)
  Scores = W_down.T @ y            [d_m]

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
    if bank == "down_values":
        return _first_existing_attr(mlp, ("down_proj", "w2", "fc2"), role="down")
    explicit = {"gate_proj", "up_proj", "fc1", "w1", "w3"}
    if bank in explicit:
        return _first_existing_attr(mlp, (bank,), role=bank)
    raise ValueError(f"Unknown bank='{bank}'")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

# Supported single-projection banks.  bank='down' is rejected with a hard
# error in extract_banks(); 'down_values' is the principled value probe.
_SINGLE_BANKS = {
    "gate", "up", "down_values",
    "gate_proj", "up_proj",
    "fc1", "w1", "w3",
}

_KEY_SPACE   = {"gate", "up", "gate_proj", "up_proj", "fc1", "w1", "w3"}
_VALUE_TRANS = {"down_values"}


def extract_banks(
    llm,
    bank: str = "up",
    normalize: bool = True,
    strict: bool = True,
    return_metadata: bool = False,
) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
    """Extract per-layer memory banks from the model MLP projections.

    Supported ``bank`` values
    -------------------------
    Key-space (query = FFN input, default probe):
        up, gate, up_proj, gate_proj, fc1, w1, w3

    Value-space (query = FFN output):
        down_values  — rows of W_down.T; theoretically correct value probe

    Args:
        llm:             HFLLM instance with resolved ``layers`` attribute.
        bank:            Which projection to use as the memory bank.
        normalize:       Apply row-wise L2 normalisation to bank rows.
        strict:          Raise on per-layer errors (True) or log them (False).
        return_metadata: Also return a per-layer metadata dict (includes 'M').

    Returns:
        banks: dict[layer_idx → Tensor[K, D]]
        metadata (optional): dict[layer_idx → info dict]
    """
    if bank in ("down", "down_proj", "fc2", "w2"):
        raise ValueError(
            f"bank='{bank}' is not supported. Raw W_down rows are output projection "
            "weights, not value vectors. Use bank='down_values' (rows of W_down.T) "
            "for the value-space probe, or bank='up' for the key-space probe."
        )

    if bank not in _SINGLE_BANKS:
        raise ValueError(
            f"Unknown bank='{bank}'. Supported: {sorted(_SINGLE_BANKS)}"
        )

    banks: dict[int, torch.Tensor] = {}
    metadata: dict[int, dict[str, Any]] = {}

    config_hidden = getattr(llm.model.config, "hidden_size", None)
    config_intermediate = getattr(llm.model.config, "intermediate_size", None)

    for i, layer in enumerate(llm.layers):
        try:
            if not hasattr(layer, "mlp"):
                raise AttributeError(f"Layer {i} has no 'mlp' attribute")
            mlp = layer.mlp

            source_name, module = _resolve_proj(mlp, bank)
            w_raw = _get_weight_float32(module)  # [out_features, in_features]

            if w_raw.ndim != 2:
                raise ValueError(f"Layer {i}: expected 2D weight, got shape={tuple(w_raw.shape)}")

            out_f, in_f = w_raw.shape

            # Shape validation per bank family
            if bank in _KEY_SPACE:
                # W_up: maps hidden → intermediate; weight [d_m, d]
                if config_hidden is not None and in_f != config_hidden:
                    raise ValueError(
                        f"Layer {i}: in_features={in_f} != hidden_size={config_hidden} "
                        f"for key-space bank='{bank}'"
                    )
                if config_intermediate is not None and out_f != config_intermediate:
                    raise ValueError(
                        f"Layer {i}: out_features={out_f} != intermediate_size={config_intermediate} "
                        f"for key-space bank='{bank}'"
                    )
                w = w_raw  # [d_m, d] — rows are key vectors

            elif bank in _VALUE_TRANS:
                # down_values: W_down.T; original weight [d, d_m], transposed → [d_m, d]
                if config_intermediate is not None and in_f != config_intermediate:
                    raise ValueError(
                        f"Layer {i}: original down_proj in_features={in_f} != "
                        f"intermediate_size={config_intermediate} for bank='down_values'"
                    )
                if config_hidden is not None and out_f != config_hidden:
                    raise ValueError(
                        f"Layer {i}: original down_proj out_features={out_f} != "
                        f"hidden_size={config_hidden} for bank='down_values'"
                    )
                w = w_raw.T.contiguous()  # [d_m, d] — rows are value vectors

            else:
                w = w_raw

            # M_raw: max row norm of the unnormalised weight (model property,
            # useful as a diagnostic across layers).
            M_raw = float(w.norm(dim=1).max())

            source = source_name if bank not in _VALUE_TRANS else f"{source_name}.T"

            if normalize:
                w = _normalize_rows(w)

            # M_used: max row norm of the *actual* matrix used for retrieval.
            # When normalize=True this is 1.0 by construction.  This is the
            # value compute_energy needs for the 0.5*M^2 term in Eq. 1 of
            # Ramsauer et al. 2020 — "M of the matrix actually being used".
            M_used = float(w.norm(dim=1).max())
            log.debug(
                "Layer %d: bank='%s', M_raw=%.4f, M_used=%.4f, shape=%s",
                i, bank, M_raw, M_used, tuple(w.shape),
            )

            banks[i] = w
            metadata[i] = {
                "layer_idx": i,
                "bank": bank,
                "source": source,
                "shape": tuple(w.shape),
                "dtype": str(w.dtype),
                "K": int(w.shape[0]),
                "D": int(w.shape[1]),
                "M":     M_used,   # used by compute_energy
                "M_raw": M_raw,    # diagnostic only
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
