"""Beta calibration for Hopfield retrieval sharpness.

Runs one prefill pass per calibration sample to capture query vectors (beta-
independent), then evaluates all candidate betas analytically from the
captured scores without any additional forward passes.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from hopfield_llm.utils.logging import get_logger

log = get_logger("metrics.calibration")

_DEFAULT_CANDIDATES = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]


def calibrate_beta(
    llm,
    banks: dict[int, torch.Tensor],
    questions: list[str],
    candidates: list[float] = _DEFAULT_CANDIDATES,
    target_fraction: float = 0.55,
    hook_target: str = "mlp_input",
    normalize_query: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Find β such that mean prefill norm_entropy ≈ target_fraction × log(K).

    Args:
        llm:              HFLLM instance.
        banks:            Per-layer memory banks {layer_idx → Tensor[K, d]}.
        questions:        Calibration questions (20 is usually sufficient).
        candidates:       Beta values to evaluate (log-spaced recommended).
        target_fraction:  Target = target_fraction × log(K). 0.55 keeps the
                          distribution sensitive but not uniform.
        hook_target:      Must match the hook_target used in the main run.
        normalize_query:  Must match the normalize_query used in the main run.

    Returns:
        (beta_star, info_dict)
        beta_star is the candidate (or interpolated value) that hits the target.
        info_dict contains per-beta entropy values and calibration metadata.
    """
    from hopfield_llm.hooks.capture import capture_prefill

    L = len(llm.layers)
    K = next(iter(banks.values())).shape[0]
    log_K = math.log(K)

    log.info(
        "calibrate_beta: %d samples, K=%d, log_K=%.3f, target=%.2f × log_K",
        len(questions), K, log_K, target_fraction,
    )

    all_scores: list[dict[int, torch.Tensor]] = []

    for q_text in questions:
        _, x_raw = capture_prefill(
            llm, q_text, banks, beta=1.0,
            return_x=True, compute_p_ref=False,
            hook_target=hook_target, normalize_query=normalize_query,
        )
        sample_scores: dict[int, torch.Tensor] = {}
        for l_idx in range(L):
            if l_idx not in banks:
                continue
            bank = banks[l_idx]                                    # [K, d]
            x_l  = torch.from_numpy(x_raw[l_idx]).to(bank.device) # [T, d]
            sample_scores[l_idx] = x_l @ bank.T                   # [T, K]
        all_scores.append(sample_scores)
        del x_raw

    candidates_sorted = sorted(candidates)
    mean_norm_entropy: dict[float, float] = {}

    for beta in candidates_sorted:
        total = 0.0
        count = 0
        for sample_scores in all_scores:
            for scores in sample_scores.values():         # [T, K]
                dist = torch.softmax(beta * scores, dim=-1)
                ent  = -(dist * dist.clamp(1e-9).log()).sum(dim=-1)  # [T]
                total += float(ent.sum()) / log_K
                count += scores.shape[0]
        mean_norm_entropy[beta] = total / count if count > 0 else float("nan")
        log.debug("  beta=%.1f  mean_norm_entropy=%.4f", beta, mean_norm_entropy[beta])

    beta_star = _interpolate_beta(candidates_sorted, mean_norm_entropy, target_fraction)

    log.info(
        "calibrate_beta: β*=%.2f  (target_norm_entropy=%.3f, actual range=[%.3f, %.3f])",
        beta_star, target_fraction,
        mean_norm_entropy[candidates_sorted[-1]],
        mean_norm_entropy[candidates_sorted[0]],
    )

    for s in all_scores:
        s.clear()

    return beta_star, {
        "beta_star":             beta_star,
        "target_fraction":       target_fraction,
        "K":                     K,
        "log_K":                 round(log_K, 4),
        "n_calibration_samples": len(questions),
        "mean_norm_entropy_by_beta": {
            str(b): round(v, 6) for b, v in mean_norm_entropy.items()
        },
    }


def _interpolate_beta(
    candidates: list[float],
    entropies: dict[float, float],
    target: float,
) -> float:
    """Linearly interpolate to find the beta that hits target norm_entropy.

    Entropy decreases monotonically as beta increases.  If target is outside
    the candidate range, the nearest boundary candidate is returned.
    """
    vals = [entropies[b] for b in candidates]

    if target >= vals[0]:    # target above highest entropy → use lowest beta
        return float(candidates[0])
    if target <= vals[-1]:   # target below lowest entropy → use highest beta
        return float(candidates[-1])

    for i in range(len(candidates) - 1):
        e_low_beta  = vals[i]        # higher entropy  (lower beta)
        e_high_beta = vals[i + 1]    # lower entropy   (higher beta)
        b_low       = candidates[i]
        b_high      = candidates[i + 1]
        if e_high_beta <= target <= e_low_beta:
            span = e_low_beta - e_high_beta
            if span < 1e-10:
                return float(b_low)
            frac = (e_low_beta - target) / span
            log_b = math.log(b_low) + frac * (math.log(b_high) - math.log(b_low))
            return float(math.exp(log_b))

    return float(candidates[-1])
