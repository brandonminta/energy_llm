"""Production pipeline vs the naive float64 reference (dev/reference.py).

Tolerances (energy_llm.verify.reference_tolerances / score_tolerance):
- scores: |ds| <= D u_fp32 (first-order bound for a float32 dot product of D
  terms; Cauchy-Schwarz bounds the normalised term sum by 1);
- every derived quantity: first-order Lipschitz bound in the max norm from
  the OBSERVED score error (E: ds; margin: 2 ds; H_bar: 2 beta ds; JS and
  Hellinger^2: 4 beta ds; delta_*: twice), plus 1e-5 for float32 reductions;
- Jaccard top-k: 1/k (a single swap at the k-th boundary).
"""

import numpy as np
import pytest

from energy_llm.bank import build_bank
from energy_llm.capture import FEATURES, VARIANTS, ResidualHooks, compute_item, forward_item
from energy_llm.prompting import build_prompt
from energy_llm.verify import compare_with_reference, load_reference
from tests.conftest import DATE, QUESTION_WORDS, SYSTEM, make_bundle, make_questions

ref = load_reference()


def test_reference_feature_order_matches_production():
    assert ref.FEATURES == FEATURES and ref.VARIANTS == VARIANTS


def _both(bundle, bank, question, betas, k=8):
    hooks = ResidualHooks(bundle.model)
    date = DATE if "date_string" in bundle.tokenizer.chat_template else None
    tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, question, date))
    hooks.remove()
    L = bundle.dims.n_layers
    prod = compute_item(bundle, bank, tr, "x", betas[1], [b / betas[1] for b in betas], k, list(range(L)))
    r = ref.reference_item(bundle.model, {l: tr.residuals[l].numpy() for l in range(L)},
                           tr.n_prompt, tr.gen_ids, bundle.stop_token_ids, betas, k,
                           list(range(L)), betas[1])
    return tr, prod, r


@pytest.mark.parametrize("beta_star", [0.7, 12.0, 300.0])
def test_production_matches_reference(pair, beta_star):
    bundle = make_bundle(*pair)
    bank = build_bank(bundle.model, bundle.dims)
    betas = [beta_star / 2, beta_star, beta_star * 2]
    for q in make_questions(3, QUESTION_WORDS, 5):
        tr, prod, r = _both(bundle, bank, q, betas)
        assert r["response_positions"] == tr.response_positions
        cmp = compare_with_reference(bundle, bank, tr, prod, r, list(range(bundle.dims.n_layers)),
                                     betas, beta_star, 8)
        assert cmp["score_error"] <= cmp["score_tolerance"]
        assert cmp["nan_pattern_equal"]
        assert cmp["max_error_over_tolerance"] <= 1.0, cmp
        assert cmp["ok"]


def test_reference_handles_empty_response(qwen_pair):
    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    q = "who wrote hamlet"
    first = _both(bundle, bank, q, [5.0, 10.0, 20.0])[0].gen_ids[0]
    stops = sorted(set(bundle.stop_token_ids) | {first})
    bundle.stop_token_ids, bundle.generation_config.eos_token_id = stops, stops
    tr, prod, r = _both(bundle, bank, q, [5.0, 10.0, 20.0])
    assert tr.empty_response and r["response_positions"] == []
    cmp = compare_with_reference(bundle, bank, tr, prod, r, list(range(bundle.dims.n_layers)),
                                 [5.0, 10.0, 20.0], 10.0, 8)
    assert cmp["ok"] and cmp["nan_pattern_equal"]


def test_reference_detects_a_wrong_pipeline(qwen_pair, monkeypatch):
    """Negative control: dropping gamma from the production scores must fail."""
    from energy_llm import bank as bank_mod

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
    prod = compute_item(bundle, bank, tr, "x", 10.0, [0.5, 1.0, 2.0], 8, list(range(L)))
    r = ref.reference_item(bundle.model, {l: tr.residuals[l].numpy() for l in range(L)},
                           tr.n_prompt, tr.gen_ids, bundle.stop_token_ids, [5.0, 10.0, 20.0], 8,
                           list(range(L)), 10.0)
    assert not compare_with_reference(bundle, bank, tr, prod, r, list(range(L)),
                                      [5.0, 10.0, 20.0], 10.0, 8)["ok"]
