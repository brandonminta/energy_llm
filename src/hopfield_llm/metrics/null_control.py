"""Shuffled-bank null control.

The Hopfield retrieval energy of an activation against an over-complete bank
(11k+ MLP rows) tends to look similar regardless of which rows are present —
because the activation accidentally aligns with *some* row by sheer dimension.
A shuffled-bank control answers the question:

    "Does the bank's row identity matter, or would any random K-row matrix
    of the same shape produce indistinguishable energies?"

If real and shuffled energies move together across samples, the metric is
not picking up bank-specific structure and any apparent signal is an
artifact of the matrix shape and norm distribution, not its content.

Implementation: per layer, permute the row order of the bank with a fixed
sample-specific seed, then run capture_prefill + capture_generation against
the permuted bank.  Saves the same shape arrays as the real run so analyze
can subtract them.
"""

from __future__ import annotations

import numpy as np
import torch

from hopfield_llm.hooks.capture import capture_generation, capture_prefill


def _shuffle_banks(
    banks: dict[int, torch.Tensor], seed: int
) -> dict[int, torch.Tensor]:
    """Return a per-layer randomly permuted copy of the banks dict.

    Each layer is permuted independently with a deterministic generator so
    the same sample yields the same shuffle on re-runs.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    out: dict[int, torch.Tensor] = {}
    for layer_idx, w in banks.items():
        K = w.shape[0]
        perm = torch.randperm(K, generator=g)
        out[layer_idx] = w[perm].to(w.device)
    return out


@torch.no_grad()
def compute_shuffled_bank_energy(
    llm,
    question: str,
    generated_text: str,
    banks: dict[int, torch.Tensor],
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    m_per_layer: dict[int, float] = {},
    hook_target: str = "mlp_input",
    normalize_query: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run prefill + a teacher-forced generation pass with a shuffled bank.

    The teacher-forcing avoids re-sampling — we want to compare energies on
    the *same* generated tokens, not on a fresh sample from the model.

    Returns:
        (prefill_energy [L], gen_energy [L, T_gen]) under the shuffled bank.
    """
    shuffled = _shuffle_banks(banks, seed)

    # Prefill on the chat-templated question (matches run_trajectory).
    pref = capture_prefill(
        llm, question, shuffled,
        beta=beta, threshold=threshold, energy_mode=energy_mode,
        hook_target=hook_target, normalize_query=normalize_query,
        m_per_layer=m_per_layer, apply_chat_template=True,
    )
    prefill_energy = pref.energy_last  # [L]

    if not generated_text:
        L = prefill_energy.shape[0]
        return prefill_energy, np.zeros((L, 0), dtype=np.float32)

    # Teacher-forced "generation" — re-run the chat-templated prompt + the
    # actual generated tokens through capture_prefill (no model.generate).
    if hasattr(llm.tokenizer, "apply_chat_template"):
        prompt_text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt_text = question
    full = capture_prefill(
        llm, prompt_text + generated_text, shuffled,
        beta=beta, threshold=threshold, energy_mode=energy_mode,
        hook_target=hook_target, normalize_query=normalize_query,
        m_per_layer=m_per_layer, apply_chat_template=False,
    )
    T_prompt = int(
        llm.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        ["input_ids"].shape[1]
    )
    T_full = full.energy.shape[1]
    T_prompt = max(1, min(T_prompt, T_full))
    gen_energy = full.energy[:, T_prompt:].astype(np.float32)
    return prefill_energy.astype(np.float32), gen_energy
