"""Production pipeline vs the naive float64 reference (dev/reference.py).

Tolerances (energy_llm.verification.verify.reference_tolerances / score_tolerance):
- scores: |ds| <= D u_fp32 (first-order bound for a float32 dot product of D
  terms; Cauchy-Schwarz bounds the normalised term sum by 1);
- every trace: first-order bound in the max norm from the OBSERVED score
  error at each beta (E: ds; margin: 2 ds; H_bar: 2 beta ds; mean/var score:
  2 beta ds range(s) + direct term; topk_mass: max(k, 2) beta ds; JS: 4 beta ds;
  dkl: 8 beta ds), plus 1e-5 * max(1, |value|) for float32 reductions;
- topk_ids as sets: row Jaccard >= 0.98.
"""

import numpy as np
import pytest

from energy_llm.core.bank import build_bank
from energy_llm.core.capture import ResidualHooks, compute_item, forward_item
from energy_llm.core.prompting import ROLE_QUESTION, build_prompt
from energy_llm.verification.verify import compare_with_reference, load_reference
from energy_llm.core.bank import VARIANTS
from energy_llm.verification.verify import view
from tests.conftest import DATE, QUESTION_WORDS, SYSTEM, both, make_bundle, make_questions

ref = load_reference()


K_MASS, K_IDS = 4, 8


def _ref(bundle, tr, betas):
    L = bundle.dims.n_layers
    q_last = max(i for i, r in enumerate(tr.prompt.roles) if r == ROLE_QUESTION)
    return ref.reference_item(bundle.model, {l: tr.residuals[l].numpy() for l in range(L)},
                              tr.n_prompt, tr.gen_ids, bundle.stop_token_ids, list(range(L)),
                              both(betas), q_last, K_MASS, K_IDS)


def _both(bundle, bank, question, betas):
    hooks = ResidualHooks(bundle.model)
    date = DATE if "date_string" in bundle.tokenizer.chat_template else None
    tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, question, date))
    hooks.remove()
    L = bundle.dims.n_layers
    prod = compute_item(bundle, bank, tr, "x", both(betas), list(range(L)), K_MASS, both(K_IDS))
    return tr, prod.traces, _ref(bundle, tr, betas)


@pytest.mark.parametrize("beta_star", [0.7, 12.0, 300.0])
def test_production_matches_reference(pair, beta_star):
    bundle = make_bundle(*pair)
    bank = build_bank(bundle.model, bundle.dims)
    betas = [1.0, 100.0, 1e4, beta_star]
    for q in make_questions(3, QUESTION_WORDS, 5):
        tr, prod, r = _both(bundle, bank, q, betas)
        assert r["response_positions"] == tr.response_positions
        assert r["reference_positions"] == tr.reference_positions
        for v in VARIANTS:
            pv = view(prod, v)
            assert set(r["traces"][v]) | {"query_gain", "xhat_top1_frac"} == set(pv)
            cmp = compare_with_reference(bundle, bank, tr, pv, r, list(range(bundle.dims.n_layers)),
                                         betas, K_MASS, v)
            assert cmp["score_error"] <= cmp["score_tolerance"], v
            assert cmp["nan_pattern_equal"]
            assert cmp["max_error_over_tolerance"] <= 1.0, (v, cmp)
            assert {"query_gain", "xhat_top1_frac", "topk_ids"} <= set(cmp["traces"])
            assert cmp["ok"], v


def test_reference_handles_empty_response(qwen_pair):
    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    q = "who wrote hamlet"
    first = _both(bundle, bank, q, [10.0])[0].gen_ids[0]
    stops = sorted(set(bundle.stop_token_ids) | {first})
    bundle.stop_token_ids, bundle.generation_config.eos_token_id = stops, stops
    tr, prod, r = _both(bundle, bank, q, [1.0, 10.0])
    assert tr.empty_response and r["response_positions"] == []
    for v in VARIANTS:
        assert np.isnan(r["traces"][v]["dkl/response_first"]).all()
        cmp = compare_with_reference(bundle, bank, tr, view(prod, v), r, list(range(bundle.dims.n_layers)),
                                     [1.0, 10.0], K_MASS, v)
        assert cmp["ok"] and cmp["nan_pattern_equal"], v


def test_reference_detects_a_wrong_pipeline(qwen_pair, monkeypatch):
    """Negative control: dropping gamma from the production pre-activation
    must fail for both variants."""
    from energy_llm.core import bank as bank_mod

    bundle = make_bundle(*qwen_pair)
    bank = build_bank(bundle.model, bundle.dims)
    hooks = ResidualHooks(bundle.model)
    tr = forward_item(bundle, hooks, build_prompt(bundle.tokenizer, SYSTEM, "who wrote hamlet", None))
    hooks.remove()

    def no_gamma(self, l, x_hat):
        return x_hat @ self.gate_weights[l].float().T

    monkeypatch.setattr(bank_mod.Bank, "preactivation", no_gamma)
    L = bundle.dims.n_layers
    prod = compute_item(bundle, bank, tr, "x", both([10.0]), list(range(L)), K_MASS, both(K_IDS))
    r = _ref(bundle, tr, [10.0])
    for v in VARIANTS:
        assert not compare_with_reference(bundle, bank, tr, view(prod.traces, v), r, list(range(L)),
                                          [10.0], K_MASS, v)["ok"], v
