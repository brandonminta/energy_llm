from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    L: int
    D: int
    safe_4bit: bool


MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen25_3b": ModelProfile("Qwen/Qwen2.5-3B-Instruct", L=36, D=2048, safe_4bit=True),
    "qwen3b": ModelProfile("Qwen/Qwen2.5-3B-Instruct", L=36, D=2048, safe_4bit=True),
    "qwen25_1_5b": ModelProfile("Qwen/Qwen2.5-1.5B-Instruct", L=28, D=1536, safe_4bit=True),
    "qwen15b": ModelProfile("Qwen/Qwen2.5-1.5B-Instruct", L=28, D=1536, safe_4bit=True),
    "phi3_mini": ModelProfile("microsoft/Phi-3-mini-4k-instruct", L=32, D=3072, safe_4bit=True),
    "phi3mini": ModelProfile("microsoft/Phi-3-mini-4k-instruct", L=32, D=3072, safe_4bit=True),
    "mistral7b": ModelProfile("mistralai/Mistral-7B-Instruct-v0.3", L=32, D=4096, safe_4bit=False),
    "llama32_3b": ModelProfile("meta-llama/Llama-3.2-3B-Instruct", L=28, D=3072, safe_4bit=True),
    "llama3b": ModelProfile("meta-llama/Llama-3.2-3B-Instruct", L=28, D=3072, safe_4bit=True),
}

DEFAULT_MODEL_ALIAS = "qwen25_3b"
