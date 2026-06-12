"""Shared fixtures: a tiny random-weight Qwen2-style SwiGLU model and a
matching whitespace tokenizer with a chat template. Everything runs on CPU
with no downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORDS = [
    "what", "is", "the", "color", "of", "sky", "capital", "france", "who",
    "wrote", "hamlet", "where", "are", "alps", "tallest", "mountain", "river",
    "longest", "ocean", "deepest", "planet", "largest", "smallest", "country",
    "city", "animal", "fastest", "bird", "fish", "tree", "oldest", "answer",
] + [f"w{i}" for i in range(64)]


def make_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<eos>": 1, "<pad>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %} answer{% endif %}"
    )
    return fast


def make_tiny_model(vocab_size: int, n_layers: int = 2, hidden: int = 32,
                    intermediate: int = 48, seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return make_tiny_tokenizer()


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer):
    return make_tiny_model(vocab_size=tiny_tokenizer.vocab_size + 4)


@pytest.fixture(scope="session")
def tiny_banks(tiny_model):
    from energy_llm.banks import build_banks
    return build_banks(tiny_model, model_id="tiny-qwen2", bank_id="gate")


def make_questions(n: int, seed: int = 7) -> list[str]:
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        words = rng.sample(WORDS, rng.randint(4, 7))
        out.append(" ".join(words))
    return out


def make_tiny_samples(n: int = 8):
    from energy_llm.data import Sample
    questions = make_questions(n)
    return [
        Sample(sample_id=f"tiny_{i:03d}", question=q, gold_answers=["sky", "blue"],
               category="cat_a" if i % 2 == 0 else "cat_b", source="tiny")
        for i, q in enumerate(questions)
    ]


@pytest.fixture()
def tiny_samples():
    return make_tiny_samples(8)
