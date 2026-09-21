"""Closed forms for the exact information counter and the exposure-corrected axis."""
from __future__ import annotations

import numpy as np

from synth.world import OBJ_BITS


def zipf_probs(n_facts: int, a: float) -> np.ndarray:
    """p_k proportional to k^-(1+a) over ranks k = 1..K, float64, normalised."""
    k = np.arange(1, n_facts + 1, dtype=np.float64)
    p = k ** (-(1.0 + a))
    return p / p.sum()


def expected_facts_delivered(p: np.ndarray, n_docs: int) -> float:
    """sum_k [1 - (1 - p_k)^N]."""
    return float(np.sum(-np.expm1(n_docs * np.log1p(-p))))


def ideal_loss_bits_zipf(p: np.ndarray, n_docs: int) -> float:
    """Ideal-learner in-distribution object loss in bits: 12 * sum_k p_k (1 - p_k)^N."""
    return OBJ_BITS * float(np.sum(p * np.exp(n_docs * np.log1p(-p))))


def ideal_loss_bits_eq(n_facts: int, repeats: int, n_docs: int) -> float:
    """Ideal learner on the EQ corpus (uniform test distribution).

    A fact is unseen after N docs of the globally shuffled corpus iff all `repeats`
    copies fall in the remaining total - N slots (hypergeometric, exact).
    """
    total = n_facts * repeats
    rest = total - n_docs
    p_unseen = 1.0
    for j in range(repeats):
        p_unseen *= max(rest - j, 0) / (total - j)
    return OBJ_BITS * p_unseen


def i_eff(n_k: np.ndarray, n0: float) -> float:
    """Exposure-corrected information axis: 12 * sum_k (1 - exp(-n_k / n0))."""
    nk = n_k[n_k > 0].astype(np.float64)
    return OBJ_BITS * float(np.sum(-np.expm1(-nk / n0)))


def shifted_zipf_probs(n_facts: int, a: float, q: float) -> np.ndarray:
    """p_k proportional to (k + q)^-(1+a) over ranks k = 1..K (v3 base distribution)."""
    k = np.arange(1, n_facts + 1, dtype=np.float64)
    p = (k + q) ** (-(1.0 + a))
    return p / p.sum()


def expected_facts_with_at_least(p: np.ndarray, n_docs: int, c: int) -> float:
    """sum_k P(Binomial(N, p_k) >= c), Poisson approximation (N p_k small for all but the head)."""
    from scipy.stats import poisson
    lam = n_docs * p
    return float(np.sum(poisson.sf(c - 1, lam)))
