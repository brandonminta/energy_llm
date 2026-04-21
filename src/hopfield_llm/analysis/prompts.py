"""Prompt templates for structured LLM generation.

CONCISE_V1 is the default system prompt for factual QA generation.
build_chat_prompt() wraps a question in the chat template the tokenizer expects.
"""

from __future__ import annotations

CONCISE_V1 = (
    "Answer the question directly in one short sentence. "
    "Do not explain your reasoning. Do not repeat the question."
)

PROMPT_TEMPLATE_VERSION = "concise_v1"


def build_chat_prompt(
    tokenizer,
    question: str,
    system_prompt: str = CONCISE_V1,
) -> str:
    """Format a question as a chat prompt using the tokenizer's template.

    Args:
        tokenizer:     HuggingFace tokenizer with apply_chat_template support.
        question:      Question text.
        system_prompt: System-level instruction prepended before the user turn.

    Returns:
        Formatted string ready for tokenization and generation.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
