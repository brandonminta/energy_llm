"""MLP memory-bank and trajectory extraction."""

from hopfield_llm.extraction.generation import GenerationResult, run_generation_pass
from hopfield_llm.extraction.memory_bank import extract_mlp_memory_bank
from hopfield_llm.extraction.prefill import PrefillResult, run_prefill_pass

__all__ = [
    "GenerationResult",
    "PrefillResult",
    "extract_mlp_memory_bank",
    "run_generation_pass",
    "run_prefill_pass",
]
