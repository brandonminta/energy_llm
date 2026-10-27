"""Prompt + role mask on a real tokenizer. Opt-in: set ENERGY_LLM_TOKENIZER_DIRS
to a colon-separated list of local tokenizer directories (e.g. snapshots of
Qwen/Qwen2.5-3B-Instruct and meta-llama/Llama-3.2-3B-Instruct)."""

import os

import pytest

from energy_llm.prompting import (
    ROLE_INSTRUCTION, ROLE_QUESTION, build_prompt, check_template,
)

DIRS = [d for d in os.environ.get("ENERGY_LLM_TOKENIZER_DIRS", "").split(":") if d]
SYSTEM = "Answer the question briefly and directly."
QUESTION = "What happens if you crack your knuckles a lot?"


@pytest.mark.skipif(not DIRS, reason="ENERGY_LLM_TOKENIZER_DIRS not set")
@pytest.mark.parametrize("path", DIRS)
def test_prompt_matches_official_template(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    date = "26 Jul 2024" if "date_string" in tok.chat_template else None
    check_template(tok, date)
    p = build_prompt(tok, SYSTEM, QUESTION, date)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    extra = {"date_string": date} if date else {}
    ref = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=False, **extra)
    assert p.input_ids == list(ref)
    assert "You are Qwen" not in p.text
    sys_txt = tok.decode([t for t, r in zip(p.input_ids, p.roles) if r == ROLE_INSTRUCTION])
    q_txt = tok.decode([t for t, r in zip(p.input_ids, p.roles) if r == ROLE_QUESTION])
    assert sys_txt.strip() == SYSTEM and q_txt.strip() == QUESTION
