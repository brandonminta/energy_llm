"""Information-theoretic primitives of the method.

Every function here is a pure NumPy/SciPy function of retrieval scores or
probability vectors. The naming follows the thesis symbols:

- ``retrieval_scores``  s   (eq:retrieval_scores)
- ``lse_term`` / ``quadratic_term``  the two stored components of the Modern
  Hopfield energy E (eq:modern_hopfield_energy, with the query-independent
  constant c_0 omitted as stated there)
- ``entropy`` H (eq:retrieval_entropy), ``norm_entropy`` H_bar
  (eq:normalised_entropy)
- ``js_nats`` (eq:js_convention), ``hellinger_sq`` (eq:hellinger_squared)

All divergences operate on the retrieval distribution p = softmax(beta * s)
over the rescaled scores s, never on the raw gate pre-activations g
(subsec:retrieval_scoring).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp, softmax

LOG2 = float(np.log(2.0))


def retrieval_scores(m_hat: np.ndarray, r: np.ndarray) -> np.ndarray:
    """s = M_hat @ r  (eq:retrieval_scores).

    ``m_hat``: row-normalised memory bank [K, D]; ``r``: probe point(s),
    [D] or [T, D]. Returns [K] or [T, K].
    """
    return np.asarray(r, dtype=np.float64) @ np.asarray(m_hat, dtype=np.float64).T


def retrieval_distribution(s: np.ndarray, beta: float) -> np.ndarray:
    """p_i = softmax(beta * s)_i over the last axis (eq:retrieval_distribution)."""
    return softmax(beta * np.asarray(s, dtype=np.float64), axis=-1)


def lse_term(s: np.ndarray, beta: float) -> np.ndarray:
    """-(1/beta) * log sum_i exp(beta * s_i), over the last axis.

    Evaluated with scipy.special.logsumexp (max-shifted) as required by
    subsec:stage2 to avoid overflow at large beta.
    """
    return -logsumexp(beta * np.asarray(s, dtype=np.float64), axis=-1) / beta


def quadratic_term(r: np.ndarray) -> np.ndarray:
    """0.5 * ||r||_2^2 over the last axis (the query-norm component of E)."""
    r = np.asarray(r, dtype=np.float64)
    return 0.5 * np.sum(r * r, axis=-1)


def energy(s: np.ndarray, r: np.ndarray, beta: float) -> np.ndarray:
    """E = lse_term + quadratic_term (eq:modern_hopfield_energy, c_0 omitted).

    Reference implementation; the pipeline stores the two terms separately
    (subsec:pa_scale_confounds) and sums them only at feature time.
    """
    return lse_term(s, beta) + quadratic_term(r)


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats over the last axis (eq:retrieval_entropy)."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0.0, p * np.log(p), 0.0)
    return -np.sum(plogp, axis=-1)


def norm_entropy(p: np.ndarray) -> np.ndarray:
    """H_bar = H / log K over the last axis (eq:normalised_entropy)."""
    k = np.asarray(p).shape[-1]
    return entropy(p) / np.log(k)


def js_nats(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence in nats, exactly per eq:js_convention:

        JS_nats(P, Q) = jensenshannon(P, Q, base=2)**2 * log(2)

    scipy returns the square root of the divergence in the given base, so the
    square restores the divergence and the log(2) factor converts bits to
    nats, keeping the bound JS <= log 2 exact.
    """
    root = jensenshannon(np.asarray(p, dtype=np.float64),
                         np.asarray(q, dtype=np.float64),
                         base=2, axis=axis)
    # Identical distributions can yield a NaN root from a 0/0 inside scipy;
    # the divergence there is exactly 0.
    root = np.nan_to_num(root, nan=0.0)
    return root**2 * LOG2


def hellinger_sq(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Squared Hellinger distance, directly in NumPy (eq:hellinger_squared):

        H^2_Hel(P, Q) = 0.5 * sum_i (sqrt(p_i) - sqrt(q_i))**2
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=axis)
