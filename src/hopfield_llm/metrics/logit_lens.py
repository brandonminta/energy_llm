"""Logit-lens entropy as a parallel hallucination signal.

The logit lens projects the residual stream at layer l through the model's
final layer norm and unembedding (LM head), giving a vocabulary distribution
"as if the model stopped reasoning at layer l".

Per-layer per-token entropy of that distribution is a known, bank-free
hallucination signal — high mid-layer entropy with low final-layer entropy
is the classic "model commits to an answer it isn't confident about"
pattern (Belrose et al. 2023, "Eliciting Latent Predictions ...").

This module is independent of the Hopfield energy framework; comparing the
two metrics on the same trajectory is the cleanest test of whether the
energy framing carries signal beyond an established baseline.

The capture path is one teacher-forced forward pass over
``chat_template(question) + generated_text`` with hooks that store the
post-block residual at each layer.  Per-layer per-token entropy and
top-token probability are computed after the pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class LogitLensResult:
    """Per-layer logit-lens metrics for one (prompt, generated_text) pair.

    All fields are NaN-safe float32 numpy arrays.
    """
    prefill_entropy:  np.ndarray   # [L]            entropy at last prompt token
    prefill_top_prob: np.ndarray   # [L]            top-1 probability at last prompt token
    gen_entropy:      np.ndarray   # [L, T_gen]     per-layer per-token entropy
    gen_top_prob:     np.ndarray   # [L, T_gen]     per-layer per-token top-1 probability
    n_tokens:         int          # T_gen actually captured


# ------------------------------------------------------------------
# Architecture helpers
# ------------------------------------------------------------------


def _resolve_final_norm(model) -> torch.nn.Module:
    """Locate the final layer norm preceding the LM head.

    Supports Qwen2/Llama (``model.model.norm``), Gemma2 (``model.model.norm``),
    Phi-3 (``model.model.norm``), GPT-2 (``model.transformer.ln_f``), and
    GPT-NeoX (``model.gpt_neox.final_layer_norm``).
    """
    backbone = model.model if hasattr(model, "model") else model
    for name in ("norm", "final_layernorm", "ln_f", "final_layer_norm"):
        if hasattr(backbone, name):
            return getattr(backbone, name)
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for name in ("ln_f", "final_layernorm"):
            if hasattr(transformer, name):
                return getattr(transformer, name)
    gpt_neox = getattr(model, "gpt_neox", None)
    if gpt_neox is not None and hasattr(gpt_neox, "final_layer_norm"):
        return gpt_neox.final_layer_norm
    raise AttributeError(
        f"Could not locate final layer norm on {type(model).__name__}. "
        "Add the attribute name to _resolve_final_norm."
    )


def _resolve_lm_head(model) -> torch.nn.Module:
    """Locate the unembedding linear layer."""
    for name in ("lm_head", "embed_out", "output_projection"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(
        f"Could not locate LM head on {type(model).__name__}. "
        "Add the attribute name to _resolve_lm_head."
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


@torch.no_grad()
def compute_logit_lens(
    llm,
    question: str,
    generated_text: str,
    apply_chat_template: bool = True,
) -> LogitLensResult:
    """One teacher-forced forward pass; return per-layer logit-lens metrics.

    Computes the entropy and top-1 probability of
    ``LMHead(FinalNorm(residual_l))`` at:

      * ``prefill``:  position prompt_len - 1 (the last prompt token —
                      whose distribution predicts the first generated token).
      * ``gen[t]``:   position prompt_len + t - 1 for t in [0, T_gen)
                      (each predicts the next generated token under
                      teacher forcing).

    Args:
        llm:                  HFLLM instance.
        question:             Question text.
        generated_text:       Model-generated continuation.
        apply_chat_template:  Wrap the question in the tokenizer's chat
                              template before tokenizing — must match what
                              run_trajectory used at capture time.

    Returns:
        LogitLensResult.  Empty `gen_entropy`/`gen_top_prob` (shape `[L, 0]`)
        when ``generated_text`` is empty.
    """
    L = llm.n_layers
    final_norm = _resolve_final_norm(llm.model)
    lm_head    = _resolve_lm_head(llm.model)

    # Build the same prompt that run_trajectory observed at generation time.
    if apply_chat_template and hasattr(llm.tokenizer, "apply_chat_template"):
        prompt_text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt_text = question

    full_text = prompt_text + (generated_text or "")
    enc = llm.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(llm.device)
    T_full = input_ids.shape[1]

    prompt_enc = llm.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(prompt_enc["input_ids"].shape[1])
    if prompt_len < 1 or prompt_len > T_full:
        # Defensive: tokenization re-encoded the boundary; clamp.
        prompt_len = max(1, min(prompt_len, T_full))
    T_gen = T_full - prompt_len

    # ── Capture post-block residual at each layer ──────────────────────────
    # For HF causal LMs each transformer layer returns a tuple whose [0] is
    # the post-block residual `[B, T, d]`.  A forward hook on the layer
    # captures it cheaply; we only keep the positions we will project.
    captured: dict[int, torch.Tensor] = {}
    keep_positions = [prompt_len - 1] + list(range(prompt_len, T_full))
    keep_idx = torch.tensor(keep_positions, dtype=torch.long, device=llm.device)

    def _make_hook(idx: int):
        def _hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            # Slice in the hook so we don't keep the full [T, d] tensor.
            captured[idx] = h[0, keep_idx, :].detach()
        return _hook

    handles = [
        llm.layers[l].register_forward_hook(_make_hook(l))
        for l in range(L)
    ]
    try:
        llm.model(input_ids=input_ids, output_hidden_states=False)
    finally:
        for h in handles:
            h.remove()

    # ── Project residuals → vocab distribution → entropy/top_prob ──────────
    prefill_entropy  = np.full(L, np.nan, dtype=np.float32)
    prefill_top_prob = np.full(L, np.nan, dtype=np.float32)
    gen_entropy      = np.full((L, T_gen), np.nan, dtype=np.float32)
    gen_top_prob     = np.full((L, T_gen), np.nan, dtype=np.float32)

    # Determine the LM head's compute dtype to keep the matmul precision
    # consistent (4-bit models keep lm_head in fp16).
    head_weight_dtype = getattr(lm_head, "weight", None)
    target_dtype = head_weight_dtype.dtype if head_weight_dtype is not None else torch.float32

    for l, h_kept in captured.items():
        # h_kept: [1 + T_gen, d]
        h = h_kept.to(target_dtype)
        h = final_norm(h)
        logits = lm_head(h)                            # [1 + T_gen, V]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        # Entropy = -sum_v p*log_p; top_prob = max_v p
        ent = -(probs * log_probs).sum(dim=-1)         # [1 + T_gen]
        top = probs.max(dim=-1).values                  # [1 + T_gen]

        prefill_entropy[l]  = float(ent[0])
        prefill_top_prob[l] = float(top[0])
        if T_gen > 0:
            gen_entropy[l]  = ent[1:].cpu().numpy()
            gen_top_prob[l] = top[1:].cpu().numpy()

    return LogitLensResult(
        prefill_entropy=prefill_entropy,
        prefill_top_prob=prefill_top_prob,
        gen_entropy=gen_entropy,
        gen_top_prob=gen_top_prob,
        n_tokens=T_gen,
    )
