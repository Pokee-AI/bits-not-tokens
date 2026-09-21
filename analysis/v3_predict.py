"""v3 Phase P pre-launch prediction: facts expected to reach c* exposures within budget B.

RAW: closed form sum_k P(Binomial(N, p_k) >= c*) (Poisson approximation), N = B / E[tokens per doc].
CAPPED / CURATED: count-only simulation of the capped stream (warm-up W uncapped, then blocks
of 50,000 draws from the renormalised distribution over uncapped facts, cap enforced), which
is the stream's own sampling rule without documents or a model.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ROOT  # noqa: E402
from synth.information import expected_facts_with_at_least, shifted_zipf_probs  # noqa: E402

B = 300_000_000
K = 1_000_000
MEAN_TEMPLATE = 6.0  # template length uniform in {4..8}


def tokens_per_doc(filler_n: int) -> float:
    return 1 + MEAN_TEMPLATE + 4 + filler_n + 1


def simulate_capped(p: np.ndarray, cap: int, warmup_docs: int, n_docs: int, block: int = 50_000,
                    seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nk = np.zeros(K, dtype=np.int64)
    cdf = np.cumsum(p)
    drawn = 0
    while drawn < n_docs:
        if drawn >= warmup_docs:
            open_ = nk < cap
            if not open_.any():
                break
            q = np.where(open_, p, 0.0)
            cdf = np.cumsum(q / q.sum())
        want = min(block, n_docs - drawn) if drawn >= warmup_docs else min(warmup_docs - drawn, n_docs - drawn)
        need = want
        while need > 0:
            r = np.minimum(np.searchsorted(cdf, rng.random(need), side="right"), K - 1)
            cnt = np.bincount(r, minlength=K)
            if drawn >= warmup_docs:
                acc = np.minimum(cnt, np.maximum(cap - nk, 0))
            else:
                acc = cnt
            nk += acc
            need -= int(acc.sum())
        drawn += want
    return nk, drawn


def main():
    cs = pd.read_csv(os.path.join(ROOT, "results", "v3_cstar.csv")).set_index("model_size")
    p = shifted_zipf_probs(K, 0.5, 1000)
    rows = []
    for m in ["S", "M", "L"]:
        cstar, W = int(cs.loc[m, "c_star"]), int(cs.loc[m, "W"])
        for corpus, filler, capped in [("RAW", 16, False), ("CAPPED", 16, True), ("CURATED", 0, True)]:
            n_docs = int(B / tokens_per_doc(filler))
            if not capped:
                pred = expected_facts_with_at_least(p, n_docs, cstar)
                delivered = float(np.sum(-np.expm1(n_docs * np.log1p(-p))))
                used = n_docs
            else:
                nk, used = simulate_capped(p, cstar, W, n_docs)
                pred = int((nk >= cstar).sum())
                delivered = int((nk > 0).sum())
            rows.append({"model_size": m, "corpus": corpus, "c_star": cstar, "W": W, "budget_tokens": B,
                         "docs_in_budget": n_docs, "docs_used": used,
                         "predicted_facts_reaching_cstar": pred, "predicted_facts_delivered": delivered})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="model_size", columns="corpus", values="predicted_facts_reaching_cstar")
    df = df.merge((piv["CURATED"] / piv["RAW"]).rename("predicted_ratio_CURATED_over_RAW").reset_index(), on="model_size")
    df.to_csv(os.path.join(ROOT, "results", "v3_predicted_storable.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
