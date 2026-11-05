"""energy-llm verify / inspect end to end on the tiny models, including a
negative control: a wrong read point must make verify fail."""

import json

import pytest

from energy_llm.pipeline.calibrate import run_calibrate
from energy_llm.verification.inspection import inspect_item
from energy_llm.storage import read_items
from energy_llm.verification.verify import format_report, run_verify
from tests.conftest import QUESTION_WORDS, WARMUP_WORDS, make_bundle, make_cfg, make_questions, write_jsonl


@pytest.fixture()
def run(tmp_path, pair):
    model, tok = pair
    arch = "llama" if "date_string" in tok.chat_template else "qwen2"
    bundle = make_bundle(model, tok)
    cfg = make_cfg(tmp_path / "run", arch)
    warm = write_jsonl(tmp_path / "warm.jsonl", make_questions(4, WARMUP_WORDS, 1), "warm")
    assert run_calibrate(cfg, bundle, read_items(warm), warm) is not None
    ev = write_jsonl(tmp_path / "eval.jsonl", make_questions(4, QUESTION_WORDS, 2), "q")
    return cfg, bundle, read_items(ev)


def test_verify_all_pass_on_tiny_models(run):
    cfg, bundle, items = run
    r = run_verify(cfg, bundle, items, n=3)
    status = {c["name"]: c["status"] for c in r["checks"]}
    assert set(status.values()) == {"PASS"}, status
    assert set(status) == {"reference", "gate", "tf_vs_cache", "alignment", "hidden",
                           "files_vs_memory", "permutation", "limits", "monotone",
                           "self_divergence", "determinism", "calibration"}
    saved = json.loads(open(f"{r['extraction_dir']}/verify_report.json").read())
    assert saved["passed"] is True
    assert "ALL PASS" in format_report(r)


def test_verify_fails_when_gamma_is_ignored(run, monkeypatch):
    from energy_llm.core import bank as bank_mod

    cfg, bundle, items = run

    def no_gamma(self, l, x_hat):
        w = self.gate_weights[l].float()
        return (x_hat @ w.T) / (w.norm(dim=1) * self.d_model ** 0.5)

    monkeypatch.setattr(bank_mod.Bank, "scores", no_gamma)
    r = run_verify(cfg, bundle, items, n=1)
    status = {c["name"]: c["status"] for c in r["checks"]}
    assert not r["passed"]
    assert status["gate"] == "FAIL" and status["reference"] == "FAIL"


def test_cli_verify_exit_code(run, tmp_path, monkeypatch):
    import yaml

    from energy_llm import cli

    cfg, bundle, items = run
    monkeypatch.setattr(cli, "_load", lambda args: (cfg, items, bundle, 0.0))
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    assert cli.main(["verify", "--config", str(p), "--input", "unused", "--n", "1"]) == 0


def test_inspect_shows_positions_and_phases(run):
    cfg, bundle, items = run
    r = run_verify(cfg, bundle, items, n=2)
    text = inspect_item(r["extraction_dir"], items[0].id, tokenizer=bundle.tokenizer)
    lines = text.splitlines()
    assert lines[0].startswith(f"id {items[0].id}")
    header = lines.index(next(l for l in lines if l.split()[:2] == ["pos", "token_id"]))
    rows = lines[header + 1: lines.index("", header)]
    where = [l.split()[3] if len(l.split()) > 3 else "" for l in rows]
    for r in ("question_last", "prompt_last", "response_first"):
        assert where.count(r) == 1, r
    for name in ("energy", "norm_entropy", "margin", "var_score", "topk_mass"):
        start = lines.index(next(l for l in lines if l.startswith(f"{name} per layer")))
        assert lines[start + 1].split() == ["layer", "question_last", "prompt_last",
                                            "response_first", "response_mean"]
        assert len([l for l in lines[start + 2: start + 2 + bundle.dims.n_layers]
                    if l.split()[0].isdigit()]) == bundle.dims.n_layers
