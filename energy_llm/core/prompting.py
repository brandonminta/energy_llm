"""Prompt construction and the per-position role mask.

The official chat template of the model is applied to exactly two messages:
the single system instruction (always explicit, so Qwen does not inject its
default system prompt) and the question. No few-shot. ``date_string`` is
passed when configured, and is mandatory when the template reads it (Llama
3.x would otherwise insert today's date and break reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass

ROLE_TEMPLATE = 0
ROLE_INSTRUCTION = 1
ROLE_QUESTION = 2
ROLE_RESPONSE = 3
ROLE_NAMES = {ROLE_TEMPLATE: "template", ROLE_INSTRUCTION: "instruction",
              ROLE_QUESTION: "question", ROLE_RESPONSE: "response"}


class PromptError(ValueError):
    pass


@dataclass
class Prompt:
    text: str
    input_ids: list[int]
    roles: list[int]     # one code per prompt position


def check_template(tokenizer, date_string: str | None) -> None:
    template = tokenizer.chat_template or ""
    if not template:
        raise PromptError("tokenizer has no chat template")
    if "date_string" in template and not date_string:
        raise PromptError("the chat template reads date_string; set prompt.date_string")


def _overlaps(a: int, b: int, lo: int, hi: int) -> bool:
    return a < hi and b > lo


def build_prompt(tokenizer, system: str, question: str, date_string: str | None) -> Prompt:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    extra = {"date_string": date_string} if date_string else {}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **extra,
    )
    s0 = text.find(system)
    if s0 < 0:
        raise PromptError("system instruction not found verbatim in the rendered prompt")
    s1 = s0 + len(system)
    q0 = text.find(question, s1)
    if q0 < 0:
        raise PromptError("question not found verbatim in the rendered prompt")
    q1 = q0 + len(question)

    # The template already contains BOS where the model expects it.
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    roles = []
    for a, b in enc["offset_mapping"]:
        if _overlaps(a, b, q0, q1):
            roles.append(ROLE_QUESTION)
        elif _overlaps(a, b, s0, s1):
            roles.append(ROLE_INSTRUCTION)
        else:
            roles.append(ROLE_TEMPLATE)
    if ROLE_QUESTION not in roles:
        raise PromptError("no prompt token maps to the question")
    return Prompt(text=text, input_ids=list(enc["input_ids"]), roles=roles)
