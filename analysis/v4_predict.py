"""v4 W1: level table (tau, Z, facts at cap, expected exposures) and pre-launch predictions.

Prediction: for each fact, lambda_k = W p_k + N_stat q_k; predicted facts stored =
sum_k E[A_size(n)], n ~ Poisson(lambda_k), with A_size the v3 RAW absorption curve of that
model size (seed mean, log-binned, interpolated in log n, clipped to [0, 1], flat beyond the
largest bin). Stated assumption: absorption depends on a fact's own exposure count alone.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ROOT  # noqa: E402
from synth.information import capped_exposures, flatten, shifted_zipf_probs, solve_tau  # noqa: E402
from synth.world import World  # noqa: E402

B = 300_000_000
K = 1_000_000
MEAN_TOKENS = {16: 28.0, 0: 12.0}  # BOS + template(6) + 4 + filler + EOS
LEVELS = [1000, 300, 100, 30]
CHANCE = 1 / 4096


def w_by_size() -> dict:
    """Warm-up W per size (documents), as frozen in C_STAR.md and recorded in results/v4_levels.csv."""
    lv = pd.read_csv(os.path.join(ROOT, "results", "v4_levels.csv"))
    return {m: int(lv[lv.model_size == m].W.iloc[0]) for m in ["S", "M", "L"]}


def absorption_curve(size: str):
    """A(n) from the v3 RAW runs (both seeds pooled), log-binned."""
    nks, hits = [], []
    for d in sorted(glob.glob(os.path.join(ROOT, "artifacts", f"RAW_seed*_{size}"))):
        pts = sorted(int(os.path.basename(f)[3:-4]) for f in glob.glob(os.path.join(d, "nk_*.npz")))
        nks.append(np.load(os.path.join(d, f"nk_{pts[-1]}.npz"))["n_k"])
        hits.append(np.unpackbits(np.load(os.path.join(d, f"hits_{pts[-1]}.npz"))["hit"])[:K].astype(bool))
    nk, hit = np.concatenate(nks), np.concatenate(hits)
    edges = np.unique(np.round(np.geomspace(1, nk.max() + 1, 40)).astype(int))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (nk >= lo) & (nk < hi)
        if m.sum() >= 30:
            xs.append(np.sqrt(lo * max(hi - 1, lo)))
            ys.append(float(np.clip(hit[m].mean() - CHANCE, 0, 1)))
    xs, ys = np.array(xs), np.array(ys)
    curve = pd.DataFrame({"model_size": size, "n_center": xs, "p_stored": ys})
    return (lambda n: np.interp(np.log(np.maximum(np.asarray(n, float), 1.0)), np.log(xs), ys)), curve


def predicted_stored(A, lam: np.ndarray, nmax: int = 800) -> float:
    ns = np.arange(0, nmax)
    an = A(ns)
    an[0] = 0.0
    idx = np.digitize(lam, np.geomspace(1e-4, 1e6, 800))
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        l = float(lam[m].mean())
        pm = poisson.pmf(ns, l)
        total += m.sum() * (pm @ an + (1.0 - pm.sum()) * an[-1])
    return float(total)


def main():
    p = shifted_zipf_probs(K, 0.5, 1000)
    W = w_by_size()
    levels, preds, curves = [], [], []
    for size in ["S", "M", "L"]:
        A, curve = absorption_curve(size)
        curves.append(curve)
        n_flat = int(B / MEAN_TOKENS[16])
        n_cur = int(B / MEAN_TOKENS[0])
        # RAW reference (v3 RAW: N documents i.i.d. from p)
        lam_raw = n_flat * p
        pred_raw = predicted_stored(A, lam_raw)
        wacc_raw = float((p * A(lam_raw)).sum())
        preds.append({"model_size": size, "corpus_type": "RAW", "level": "RAW", "tau": np.nan, "W": 0,
                      "N": n_flat, "N_stat": n_flat, "predicted_facts_stored": pred_raw,
                      "predicted_ratio_to_RAW": 1.0, "predicted_weighted_acc_p": wacc_raw})
        # levels: tau defined on FLAT's N_stat; CURATED shares tau; CURATED-x30 has its own tau
        specs = [(c, "FLAT", n_flat) for c in LEVELS] + [(c, "CURATED", n_cur) for c in LEVELS] + [("x30", "CURATED", n_cur)]
        for c, ctype, n_docs in specs:
            n_stat = n_docs - W[size]
            if c == "x30":
                tau = solve_tau(p, n_stat, 30)
            else:
                tau = solve_tau(p, n_flat - W[size], c)
            q = flatten(p, tau)
            lam = W[size] * p + n_stat * q
            pred = predicted_stored(A, lam)
            levels.append({"model_size": size, "corpus_type": ctype, "level": str(c), "tau": tau,
                           "Z": float(np.minimum(p, tau).sum()), "W": W[size], "N": n_docs, "N_stat": n_stat,
                           "n_facts_at_cap": int((p > tau).sum()),
                           "expected_stationary_exposures_at_cap": capped_exposures(p, tau, n_stat),
                           "expected_total_exposures_at_cap": float(lam[p > tau].mean())})
            preds.append({"model_size": size, "corpus_type": ctype, "level": str(c), "tau": tau, "W": W[size],
                          "N": n_docs, "N_stat": n_stat, "predicted_facts_stored": pred,
                          "predicted_ratio_to_RAW": pred / pred_raw,
                          "predicted_weighted_acc_p": float((p * A(lam)).sum())})
    pd.DataFrame(levels).to_csv(os.path.join(ROOT, "results", "v4_levels.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(ROOT, "results", "v4_predicted.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(ROOT, "results", "v4_absorption_curve_raw.csv"), index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(levels).round(3).to_string(index=False))
    print(pd.DataFrame(preds).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
