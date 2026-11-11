"""Model loading, architecture resolution and the clean greedy generation
config.

Dimensions (L, D, D_ff, RMSNorm eps) are read from the checkpoint config and
checked against the actual weights; nothing is hardcoded per model.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

log = logging.getLogger("energy_llm.core.model")


@dataclass(frozen=True)
class ModelDims:
    n_layers: int        # L
    d_model: int         # D
    d_ff: int            # D_ff
    rms_norm_eps: float


@dataclass
class ModelBundle:
    """Everything extraction needs about a loaded model."""
    model: nn.Module
    tokenizer: object
    dims: ModelDims
    stop_token_ids: list[int]
    generation_config: object
    device: torch.device
    dtype: str           # actual parameter dtype, recorded in the manifest


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    try:
        return model.model.layers
    except AttributeError:
        raise AttributeError(
            f"{model.__class__.__name__} has no model.layers; only Qwen2/Llama-style "
            "decoders are supported"
        ) from None


# Architecture adapter: the RMSNorm in front of the gated FFN, per
# ``config.model_type``. Its INPUT is the read point (the residual x) and its
# effective gain is the vector x_hat is multiplied by: RMSNorm(x) = x_hat * gain.
# Scope: pre-RMSNorm decoders with a gated, dense FFN (overview §3).
_FFN_NORM = {"qwen2": "post_attention_layernorm", "llama": "post_attention_layernorm"}


def _arch(model: nn.Module) -> str:
    mt = str(getattr(model.config, "model_type", ""))
    if mt in _FFN_NORM:
        return mt
    if mt.startswith("gemma"):
        raise NotImplementedError(
            f"{mt}: not implemented. GemmaRMSNorm scales by (1 + weight), so the effective "
            "gain would be 1 + gamma, and Gemma 2/3 put the FFN read point at "
            "pre_feedforward_layernorm (post_attention_layernorm normalises the attention "
            "output); neither is implemented nor tested")
    raise ValueError(f"unsupported architecture {mt!r}: supported {sorted(_FFN_NORM)}")


def ffn_norm(model: nn.Module, layer: nn.Module) -> nn.Module:
    """The norm whose input is the FFN read point of ``layer``."""
    return getattr(layer, _FFN_NORM[_arch(model)])


def effective_gain(model: nn.Module, layer: nn.Module) -> torch.Tensor:
    """float32 [D] gain of the FFN norm: gamma = weight for Qwen2 / Llama."""
    return ffn_norm(model, layer).weight.detach().float()


def model_dims(model: nn.Module) -> ModelDims:
    cfg = model.config
    dims = ModelDims(
        n_layers=int(cfg.num_hidden_layers),
        d_model=int(cfg.hidden_size),
        d_ff=int(cfg.intermediate_size),
        rms_norm_eps=float(cfg.rms_norm_eps),
    )
    layers = decoder_layers(model)
    if len(layers) != dims.n_layers:
        raise ValueError(f"config says L={dims.n_layers}, model has {len(layers)} layers")
    for l, layer in enumerate(layers):
        w = layer.mlp.gate_proj.weight
        if tuple(w.shape) != (dims.d_ff, dims.d_model):
            raise ValueError(f"layer {l}: gate_proj.weight {tuple(w.shape)} != (D_ff, D)")
        norm = ffn_norm(model, layer)
        if tuple(norm.weight.shape) != (dims.d_model,):
            raise ValueError(f"layer {l}: FFN norm weight is not [D]")
        eps = getattr(norm, "variance_epsilon", None)
        if eps is not None and float(eps) != dims.rms_norm_eps:
            raise ValueError(f"layer {l}: norm eps {eps} != config rms_norm_eps")
    return dims


def stop_token_ids(model: nn.Module, tokenizer) -> list[int]:
    """Every token that ends a turn: generation_config.eos_token_id (int or
    list, e.g. <|im_end|> for Qwen, <|eot_id|> for Llama) plus the tokenizer
    eos. Read BEFORE the generation config is replaced."""
    ids: set[int] = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(int(e) for e in eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    if not ids:
        raise ValueError("model defines no stop tokens")
    return sorted(ids)


def clean_generation_config(max_new_tokens: int, stop_ids: list[int], pad_id: int):
    """Pure greedy decoding. Replaces the hub generation_config entirely:
    Qwen2.5 ships repetition_penalty=1.05, which transformers would otherwise
    merge in and which makes decoding differ from argmax of the logits."""
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        eos_token_id=list(stop_ids), pad_token_id=pad_id,
        repetition_penalty=1.0, temperature=1.0, top_k=None, top_p=1.0,
        no_repeat_ngram_size=0,
        return_dict_in_generate=True, output_logits=True,   # raw logits for log-prob/entropy
    )


def prepare_bundle(model: nn.Module, tokenizer, max_new_tokens: int,
                   require_bf16: bool = True) -> ModelBundle:
    """Validate a loaded model and install the clean greedy config."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    if require_bf16 and dtype != torch.bfloat16:
        raise RuntimeError(f"model loaded as {dtype}, expected torch.bfloat16")
    dims = model_dims(model)
    stops = stop_token_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stops[0]
    gen_cfg = clean_generation_config(max_new_tokens, stops, pad)
    model.generation_config = gen_cfg
    return ModelBundle(
        model=model, tokenizer=tokenizer, dims=dims, stop_token_ids=stops,
        generation_config=gen_cfg, device=next(model.parameters()).device,
        dtype=str(dtype).replace("torch.", ""),
    )


def release_cuda() -> None:
    """Free cached GPU memory between questions (after a failure or OOM)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_id: str, revision: str, device: str = "auto", dtype=torch.bfloat16):
    """Load model + tokenizer at the exact hub revision in bfloat16 (``dtype``
    is overridden only by verify's float32 reference). With HF_HUB_OFFLINE=1
    (compute nodes) this reads only the local cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required (offset mapping for the role mask)")
    return model, tokenizer
