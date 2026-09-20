"""H1: fit beta = -d log(loss) / d log(docs) per Zipf corpus and seed (section 8)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import CORPORA, PREDICTED_BETA, ROOT, load_runs  # noqa: E402

LO, HI = 0.05, 6.0  # loss window in bits


def fit_slope(docs, loss, mask=None):
    """-slope of log(loss) vs log(docs). Default mask: points with LO <= loss <= HI."""
    m = ((loss >= LO) & (loss <= HI)) if mask is None else mask
    if m.sum() < 2:
        return np.nan, int(m.sum()), m
    slope = np.polyfit(np.log(docs[m]), np.log(loss[m]), 1)[0]
    return -slope, int(m.sum()), m


def fit_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CORPORA:
        if not c.startswith("Z"):
            continue
        for seed, g in df[df.corpus == c].groupby("seed"):
            g = g.sort_values("docs")
            beta, n, mask = fit_slope(g.docs.values, g.obj_loss_bits_indist.values)
            # ideal learner fitted over the same document range (same points)
            beta_ideal, _, _ = fit_slope(g.docs.values, g.ideal_loss_bits.values, mask)
            rows.append({"corpus": c, "zipf_a": g.zipf_a.iloc[0], "filler_n": g.filler_n.iloc[0],
                         "seed": seed, "beta": beta, "beta_ideal_same_range": beta_ideal,
                         "beta_predicted": PREDICTED_BETA[float(g.zipf_a.iloc[0])],
                         "n_points": n, "docs_min": g.docs.values[mask].min() if n else np.nan,
                         "docs_max": g.docs.values[mask].max() if n else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "beta_fits.csv"), index=False)
    return out


def summarize(fits: pd.DataFrame) -> dict:
    """Seed-averaged betas and the H1 verdict."""
    s = {}
    for c, g in fits.groupby("corpus"):
        s[c] = {"beta_mean": g.beta.mean(), "beta_per_seed": dict(zip(g.seed, g.beta)),
                "beta_ideal_mean": g.beta_ideal_same_range.mean(),
                "beta_predicted": g.beta_predicted.iloc[0], "zipf_a": g.zipf_a.iloc[0],
                "filler_n": g.filler_n.iloc[0]}
    verdict = True
    for a in (0.5, 1.0):
        for n in (0, 64):
            c = f"Z{'05' if a == 0.5 else '10'}_n{n}"
            if c in s and abs(s[c]["beta_mean"] - PREDICTED_BETA[a]) > 0.05:
                verdict = False
        c0, c64 = f"Z{'05' if a == 0.5 else '10'}_n0", f"Z{'05' if a == 0.5 else '10'}_n64"
        if c0 in s and c64 in s:
            s[f"delta_a{a}"] = s[c64]["beta_mean"] - s[c0]["beta_mean"]
            if abs(s[f"delta_a{a}"]) >= 0.03:
                verdict = False
    s["H1_PASS"] = verdict and all(c in s for c in ["Z05_n0", "Z05_n64", "Z10_n0", "Z10_n64"])
    return s


if __name__ == "__main__":
    fits = fit_all(load_runs())
    print(fits.to_string(index=False))
    for k, v in summarize(fits).items():
        print(k, v)
