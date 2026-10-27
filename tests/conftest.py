"""Shared fixtures: tiny random-weight Qwen2 and Llama models with the same
architecture as the targets, word-level tokenizers with chat templates that
mimic the real ones (Qwen injects a default system prompt when none is given;
Llama reads date_string), and a generation_config carrying hub-style
sampling settings that extraction must override. CPU only, no downloads."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYSTEM = "answer briefly and directly"
DATE = "26 jul 2024"
QUESTION_WORDS = [
    "what", "is", "the", "color", "of", "sky", "capital", "france", "who",
    "wrote", "hamlet", "where", "are", "alps", "tallest", "mountain", "river",
    "longest", "ocean", "deepest", "planet", "largest", "smallest", "country",
]
WARMUP_WORDS = [f"w{i}" for i in range(40)]
TEMPLATE_WORDS = ["system", "user", "assistant", "eot", "bos", "you", "default",
                  "today", "answer", "briefly", "and", "directly", "26", "jul", "2024"]

QWEN_TEMPLATE = (
    "{%- if messages[0]['role'] != 'system' %}system you are default eot {% endif -%}"
    "{%- for m in messages %}{{ m['role'] }} {{ m['content'] }} eot {% endfor -%}"
    "{%- if add_generation_prompt %}assistant{% endif %}"
)
LLAMA_TEMPLATE = (
    "bos {% for m in messages %}{{ m['role'] }} "
    "{% if m['role'] == 'system' %}today {{ date_string }} {% endif %}"
    "{{ m['content'] | trim }} eot {% endfor %}"
    "{% if add_generation_prompt %}assistant{% endif %}"
)


def make_tokenizer(template: str):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<eos>": 1, "<pad>": 2}
    for w in TEMPLATE_WORDS + QUESTION_WORDS + WARMUP_WORDS:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                   eos_token="<eos>", pad_token="<pad>")
    fast.chat_template = template
    return fast


def make_model(arch: str, vocab_size: int, eot_id: int, eos_id: int, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    kw = dict(vocab_size=vocab_size, hidden_size=32, intermediate_size=48,
              num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=256, tie_word_embeddings=True,
              initializer_range=0.2)  # 0.02 collapses greedy output to one token
    if arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(rms_norm_eps=1e-6, **kw))
    else:
        model = LlamaForCausalLM(LlamaConfig(rms_norm_eps=1e-5, **kw))
    with torch.no_grad():
        for layer in model.model.layers:
            # non-trivial gamma so folding it into the bank is actually tested
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
            layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gc = model.generation_config
    gc.eos_token_id = [eos_id, eot_id]
    gc.do_sample, gc.temperature, gc.top_k = True, 0.7, 20
    gc.repetition_penalty = 1.3
    return model


def make_pair(arch: str):
    tok = make_tokenizer(QWEN_TEMPLATE if arch == "qwen2" else LLAMA_TEMPLATE)
    eot = tok.convert_tokens_to_ids("eot")
    model = make_model(arch, len(tok) + 4, eot_id=eot, eos_id=tok.eos_token_id)
    return model, tok


def make_bundle(model, tok, max_new_tokens: int = 8):
    from energy_llm.model import prepare_bundle

    return prepare_bundle(model, tok, max_new_tokens, require_bf16=False)


@pytest.fixture(params=["qwen2", "llama"])
def pair(request):
    return make_pair(request.param)


@pytest.fixture()
def qwen_pair():
    return make_pair("qwen2")


def make_questions(n: int, words: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.sample(words, rng.randint(4, 7))) for _ in range(n)]


def write_jsonl(path: Path, questions: list[str], prefix: str, extra: bool = True) -> Path:
    with open(path, "w") as fh:
        for i, q in enumerate(questions):
            rec = {"id": f"{prefix}_{i:03d}", "question": q}
            if extra:
                rec |= {"answer": "gold must never propagate", "category": "c"}
            fh.write(json.dumps(rec) + "\n")
    return path


def make_cfg(out_dir: Path, arch: str = "qwen2", **extraction):
    from energy_llm.config import RunConfig

    raw = {
        "name": f"tiny-{arch}", "seed": 42, "output_dir": str(out_dir),
        "model": {"id": f"tiny/{arch}", "revision": "0" * 40},
        "prompt": {"system": SYSTEM, "date_string": DATE if arch == "llama" else None},
        "generation": {"max_new_tokens": 8},
        "calibration": {"target_norm_entropy": 0.55, "log10_beta_min": -1.0,
                        "log10_beta_max": 4.0, "num_grid": 51},
        "extraction": {"jaccard_top_k": 8, "samples_per_shard": 3, "max_attempts": 3,
                       } | extraction,
    }
    return RunConfig.from_dict(raw)


@pytest.fixture()
def calibrated(tmp_path, qwen_pair):
    """A calibrated tiny Qwen run with an evaluation JSONL of 7 items."""
    from energy_llm.calibrate import run_calibrate
    from energy_llm.storage import read_items

    model, tok = qwen_pair
    bundle = make_bundle(model, tok)
    cfg = make_cfg(tmp_path / "run")
    warm = write_jsonl(tmp_path / "warmup.jsonl", make_questions(4, WARMUP_WORDS, 1), "warm")
    rec = run_calibrate(cfg, bundle, read_items(warm), warm)
    assert rec is not None
    ev = write_jsonl(tmp_path / "eval.jsonl", make_questions(7, QUESTION_WORDS, 2), "q")
    return cfg, bundle, ev, rec
