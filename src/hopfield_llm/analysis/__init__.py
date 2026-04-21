"""Analysis utilities: prompt templates and token-level answer alignment."""

from hopfield_llm.analysis.alignment import (
    content_token_mask,
    extract_answer_features,
    find_answer_span,
)
from hopfield_llm.analysis.prompts import (
    CONCISE_V1,
    PROMPT_TEMPLATE_VERSION,
    build_chat_prompt,
)

__all__ = [
    "CONCISE_V1",
    "PROMPT_TEMPLATE_VERSION",
    "build_chat_prompt",
    "find_answer_span",
    "content_token_mask",
    "extract_answer_features",
]
