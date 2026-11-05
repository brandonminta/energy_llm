"""Production pipeline vs the naive float64 reference (dev/reference.py).

Tolerances (energy_llm.verification.verify.reference_tolerances / score_tolerance):
- scores: |ds| <= D u_fp32 (first-order bound for a float32 dot product of D
  terms; Cauchy-Schwarz bounds the normalised term sum by 1);
- every trace: first-order Lipschitz bound in the max norm from the OBSERVED
  score error (E: ds; margin: 2 ds; H_bar: 2 beta ds), plus 1e-5 for float32
  reductions.
"""

import numpy as np
import pytest

from energy_llm.core.bank import build_bank
from energy_llm.core.capture import ResidualHooks, compute_item, forward_item
from energy_llm.core.prompting import build_prompt
from energy_llm.verification.verify import compare_with_reference, load_reference
from tests.conftest import DATE, QUESTION_WORDS, SYSTEM, make_bundle, make_questions

ref = load_reference()


def _both(bundle, bank, question, beta):
    hooks = ResidualHooks(bundle.model)
    date = DATE if "date_string" in bundle.tokenizer.chat_template else None
    tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, question, date))
    hooks.remove()
    L = bundle.dims.n_layers
    prod = compute_item(bundle, bank, tr, "x", beta, list(range(L)))
    r = ref.reference_item(bundle.model, {l: tr.residuals[l].numpy() for l in range(L)},
                           tr.n_prompt, tr.gen_ids, bundle.stop_token_ids, list(range(L)), beta)
    return tr, prod, r


@pytest.mark.parametrize("beta_star", [0.7, 12.0, 300.0])
def test_production_matches_reference(pair, beta_star):
    bundle = make_bundle(*pair)
    bank = build_bank(bundle.model, bundle.dims)
    for q in make_questions(3, QUESTION_WORDS, 5):
        tr, prod, r = _both(bundle, bank, q, beta_star)
        assert r["response_positions"] == tr.response_positions
        cmp = compare_with_reference(bundle, bank, tr, prod, r, list(range(bundle.dims.n_layers)),
                                     beta_star)
        assert cmp["score_error"] <= cmp["score_tolerance"]
        assert cmp["nan_pattern_equal"]
        assert cmp["max_error_over_tolerance"] <= 1.0, cmp
        assert cmp["ok"]


def test_reference_handles_empty_response(qwen_pair):
    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    q = "who wrote hamlet"
    first = _both(bundle, bank, q, 10.0)[0].gen_ids[0]
    stops = sorted(set(bundle.stop_token_ids) | {first})
    bundle.stop_token_ids, bundle.generation_config.eos_token_id = stops, stops
    tr, prod, r = _both(bundle, bank, q, 10.0)
    assert tr.empty_response and r["response_positions"] == []
    cmp = compare_with_reference(bundle, bank, tr, prod, r, list(range(bundle.dims.n_layers)), 10.0)
    assert cmp["ok"] and cmp["nan_pattern_equal"]


def test_reference_detects_a_wrong_pipeline(qwen_pair, monkeypatch):
    """Negative control: dropping gamma from the production scores must fail."""
    from energy_llm.core import bank as bank_mod

    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    hooks = ResidualHooks(bundle.model)
    tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, "who wrote hamlet", None))
    hooks.remove()

    def no_gamma(self, l, x_hat):
        w = self.gate_weights[l].float()
        return (x_hat @ w.T) / (w.norm(dim=1) * np.sqrt(self.d_model))

    monkeypatch.setattr(bank_mod.Bank, "scores", no_gamma)
    L = bundle.dims.n_layers
    prod = compute_item(bundle, bank, tr, "x", 10.0, list(range(L)))
    r = ref.reference_item(bundle.model, {l: tr.residuals[l].numpy() for l in range(L)},
                           tr.n_prompt, tr.gen_ids, bundle.stop_token_ids, list(range(L)), 10.0)
    assert not compare_with_reference(bundle, bank, tr, prod, r, list(range(L)), 10.0)["ok"]
