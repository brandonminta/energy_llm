"""Stage 1 — memory bank extraction (subsec:stage1).

The memory bank of layer l is M^(l) = A^(l) = W_1^(l)T, the gate projection
with stored patterns indexed by rows (eq:memory_bank_definition). In the
PyTorch ``nn.Linear`` convention, ``gate_proj.weight`` already has shape
[K_l, D] = [out_features, in_features], which *is* W_1^T row-indexed, so the
bank is the weight tensor itself with no transpose.

Per the implementation spec: dequantise if quantised, cast float32 on CPU,
row-wise L2 normalise, and store the normalised rows ``m_hat`` AND the
per-row norms so the raw gate pre-activations g remain recoverable from the
retrieval scores s by the fixed rescaling of eq:retrieval_scores.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from energy_llm.io_artifacts import save_torch_atomic
from energy_llm.models import resolve_decoder_layers, resolve_mlp, resolve_projection

NORM_EPS = 1e-12  # guards a pathological all-zero row; never triggered on real banks


def _dequantized_weight(linear: nn.Module) -> torch.Tensor:
    """Return the projection weight as a dense tensor, dequantising
    bitsandbytes 4-bit parameters when present (subsec:stage1)."""
    weight = linear.weight
    if weight.__class__.__name__ == "Params4bit":
        import bitsandbytes.functional as bnb_fn
        return bnb_fn.dequantize_4bit(weight.data, weight.quant_state)
    return weight.data


def extract_memory_bank(model: nn.Module, bank_id: str = "gate") -> dict[str, Any]:
    """Walk every decoder layer and build {m_hat, row_norms, dims} per layer.

    All arithmetic runs in float32 on CPU: the row norms and the log-sum-exp
    are sensitive to the reduced mantissa of half precision (subsec:stage1).
    """
    layers = resolve_decoder_layers(model)
    m_hat: dict[int, torch.Tensor] = {}
    row_norms: dict[int, torch.Tensor] = {}
    dims: dict[int, list[int]] = {}

    for l, layer in enumerate(layers):
        proj = resolve_projection(resolve_mlp(layer), bank_id)
        weight = _dequantized_weight(proj).float().cpu()  # [out, in]
        if bank_id == "w2":
            # down_proj.weight is [D, K_l]; rows of W_2^T live in the
            # transpose. Capability only — never evaluated (subsec:stage1).
            bank = weight.T.contiguous()
        else:
            bank = weight.contiguous()  # gate/value: [K_l, D] already row-indexed
        norms = torch.linalg.vector_norm(bank, dim=1)  # [K_l]
        m_hat[l] = bank / norms.clamp_min(NORM_EPS).unsqueeze(1)
        row_norms[l] = norms
        dims[l] = list(bank.shape)

    return {"m_hat": m_hat, "row_norms": row_norms, "dims": dims}


def build_banks(
    model: nn.Module,
    model_id: str,
    bank_id: str = "gate",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the banks artifact and optionally serialise it to banks.pt."""
    bank = extract_memory_bank(model, bank_id=bank_id)
    artifact = {
        "bank_id": bank_id,
        "model_id": model_id,
        "n_layers": len(bank["m_hat"]),
        "dims": bank["dims"],
        "m_hat": bank["m_hat"],
        "row_norms": bank["row_norms"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if output_path is not None:
        save_torch_atomic(output_path, artifact)
    return artifact


def load_banks(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    required = {"bank_id", "model_id", "n_layers", "m_hat", "row_norms"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"banks artifact {path} is missing keys: {sorted(missing)}")
    return artifact
