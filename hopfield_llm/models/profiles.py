"""Predefined model profiles for supported LLM architectures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    L: int  # number of transformer layers
    D: int  # hidden dimension
    safe_4bit: bool


MODEL_PROFILES: dict[str, ModelProfile] = {
    alias: ModelProfile(model_id=mid, L=L, D=D, safe_4bit=s4)
    for aliases, mid, L, D, s4 in [
        # ── 1–3 B  (local + HPC) ─────────────────────────────────────────────
        (("qwen25_1_5b", "qwen15b"),
         "Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, True),
        (("qwen25_3b", "qwen3b", "qwen2.5-3b", "qwen2.5-3b-instruct"),
         "Qwen/Qwen2.5-3B-Instruct", 36, 2048, True),
        (("phi3_mini", "phi3mini"),
         "microsoft/Phi-3-mini-4k-instruct", 32, 3072, True),
        (("llama32_3b", "llama3b"),
         "meta-llama/Llama-3.2-3B-Instruct", 28, 3072, True),
        (("gemma2_2b", "gemma2b"),
         "google/gemma-2-2b-it", 26, 2304, True),
        # ── 7–9 B  (HPC recommended, 4-bit fits A100 20 GB) ──────────────────
        (("qwen25_7b", "qwen7b"),
         "Qwen/Qwen2.5-7B-Instruct", 28, 3584, True),
        (("mistral7b",),
         "mistralai/Mistral-7B-Instruct-v0.3", 32, 4096, False),  # bitsandbytes compat issue
        (("llama31_8b", "llama8b"),
         "meta-llama/Llama-3.1-8B-Instruct", 32, 4096, True),
        (("gemma2_9b", "gemma9b"),
         "google/gemma-2-9b-it", 42, 3584, True),
        # ── 12–14 B  (HPC A100 20 GB, 4-bit ≈ 7–8 GB) ───────────────────────
        (("phi3_medium", "phi3medium"),
         "microsoft/Phi-3-medium-4k-instruct", 40, 5120, True),
        (("qwen25_14b", "qwen14b"),
         "Qwen/Qwen2.5-14B-Instruct", 48, 5120, True),
        # ── sub-2 B  (fast local smoke-tests) ────────────────────────────────
        (("llama32_1b", "llama1b"),
         "meta-llama/Llama-3.2-1B-Instruct", 16, 2048, True),
    ]
    for alias in aliases
}

DEFAULT_MODEL_ALIAS = "qwen25_3b"


def resolve_model_profile(model_id_or_alias: str) -> ModelProfile | None:
    return MODEL_PROFILES.get(model_id_or_alias.lower())
