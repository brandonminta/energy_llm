"""Model loading and SwiGLU sub-module resolution.

Models are loaded through ``AutoModelForCausalLM.from_pretrained`` so the
checkpoint's exact parameterisation is reproduced (subsec:stage1). The alias
registry resolves the SwiGLU projections (``gate_proj`` / ``up_proj`` /
``down_proj``) for the Qwen and Llama families.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    n_layers: int       # L
    hidden_dim: int     # D
    intermediate_dim: int  # D_ff = K_l


# Registered experiment models (subsec:models). Other HF ids pass through
# with dims resolved from the checkpoint config.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen25_3b": ModelSpec("Qwen/Qwen2.5-3B-Instruct", 36, 2048, 11008),
    "llama32_3b": ModelSpec("meta-llama/Llama-3.2-3B-Instruct", 28, 3072, 8192),
}

# SwiGLU projection attribute names on the decoder layer's mlp module,
# shared by the Qwen2.5 and Llama-3 families.
SWIGLU_PROJECTIONS = {
    "gate": "gate_proj",   # W_1^T rows — the memory bank (eq:memory_bank_definition)
    "value": "up_proj",    # B = V^T rows — capability only, never evaluated
    "w2": "down_proj",     # rows of W_2^T — capability only, never evaluated
}


def resolve_model_id(alias_or_id: str) -> str:
    spec = MODEL_REGISTRY.get(alias_or_id.lower())
    return spec.model_id if spec is not None else alias_or_id


def load_model_and_tokenizer(
    alias_or_id: str,
    dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    device: str = "auto",
):
    """Load a frozen causal LM + tokenizer. 4-bit NF4 is a local-dev path only
    and must never produce a reported number (subsec:models)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve_model_id(alias_or_id)
    torch_dtype = getattr(torch, dtype)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs: dict = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Decoder layer list for Qwen/Llama-style models (model.model.layers)."""
    for path in (("model", "layers"), ("transformer", "h")):
        obj: nn.Module = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj  # type: ignore[return-value]
    raise AttributeError(
        f"Cannot resolve decoder layers for {model.__class__.__name__}"
    )


def resolve_mlp(layer: nn.Module) -> nn.Module:
    if not hasattr(layer, "mlp"):
        raise AttributeError(f"Decoder layer {layer.__class__.__name__} has no 'mlp'")
    return layer.mlp


def resolve_projection(mlp: nn.Module, bank_id: str) -> nn.Module:
    """Resolve the SwiGLU projection backing ``bank_id``.

    The raw down-projection orientation is rejected explicitly: its rows live
    in the output dimension and would silently confuse keys with values
    (subsec:stage1). Use bank_id='w2' for rows of W_2^T.
    """
    if bank_id in ("down", "down_proj"):
        raise ValueError(
            "bank_id 'down_proj' is rejected: raw down_proj rows live in the "
            "output dimension and confuse keys with values. Use bank_id='w2' "
            "for rows of W_2^T (capability only; never evaluated)."
        )
    if bank_id not in SWIGLU_PROJECTIONS:
        raise ValueError(
            f"Unknown bank_id '{bank_id}'. Choices: {sorted(SWIGLU_PROJECTIONS)}"
        )
    attr = SWIGLU_PROJECTIONS[bank_id]
    if not hasattr(mlp, attr):
        raise AttributeError(
            f"MLP module {mlp.__class__.__name__} has no '{attr}'; "
            "is this a SwiGLU (Qwen/Llama-family) model?"
        )
    return getattr(mlp, attr)
