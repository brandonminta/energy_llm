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
        (("qwen25_3b", "qwen3b", "qwen2.5-3b", "qwen2.5-3b-instruct"),
         "Qwen/Qwen2.5-3B-Instruct", 36, 2048, True),
        (("qwen25_1_5b", "qwen15b"),
         "Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, True),
        (("phi3_mini", "phi3mini"),
         "microsoft/Phi-3-mini-4k-instruct", 32, 3072, True),
        (("mistral7b",),
         "mistralai/Mistral-7B-Instruct-v0.3", 32, 4096, False),
        (("llama32_3b", "llama3b"),
         "meta-llama/Llama-3.2-3B-Instruct", 28, 3072, True),
    ]
    for alias in aliases
}

DEFAULT_MODEL_ALIAS = "qwen25_3b"


def resolve_model_profile(model_id_or_alias: str) -> ModelProfile | None:
    return MODEL_PROFILES.get(model_id_or_alias.lower())
