"""v3 Phase 0: token-waste accounting from the existing Zipf runs (no training, no GPU).

Inputs: results/absorption.csv (per-fact top-1 hits binned by n_k, from the bit-identical
reruns), artifacts/<run>_rerun/{nk,hits}_<final>.npz, results/runs.csv.
Outputs: results/v3_absorption_c.csv (c50/c90), results/v3_waste.csv.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.absorption import BINS, run_dirs  # noqa: E402
from analysis.common import ROOT  # noqa: E402

ZIPF = ["Z05_n0", "Z05_n64", "Z10_n0", "Z10_n64"]
CHANCE = 1 / 4096


def bin_centers():
    """Geometric centre of each n_k bin (the open last bin uses 2x its lower edge)."""
    return np.array([np.sqrt(lo * (hi if hi < 10**9 else 2 * lo)) for lo, hi in BINS])


def c_at(level: float, x: np.ndarray, y: np.ndarray):
    """Exposure count at which P(stored) first reaches `level`; linear interpolation in log x."""
    for i in range(len(y)):
        if y[i] >= level:
            if i == 0:
                return float(x[0])
            lx = np.log(x[i - 1]) + (level - y[i - 1]) * (np.log(x[i]) - np.log(x[i - 1])) / (y[i] - y[i - 1])
            return float(np.exp(lx))
    return None


def absorption_c(ab: pd.DataFrame) -> pd.DataFrame:
    x = bin_centers()
    rows = []
    for c in ZIPF:
        g = ab[ab.corpus == c]
        per_seed = {}
        for seed, gs in g.groupby("seed"):
            y = gs.set_index("bin").loc[[b for b, _ in [(f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">={lo}"), 0) for lo, hi in BINS]], "frac_stored"].values
            per_seed[seed] = (c_at(0.5, x, y), c_at(0.9, x, y))
        ym = g.groupby("bin", sort=False).frac_stored.mean().values
        rows.append({"corpus": c, "c50": c_at(0.5, x, ym), "c90": c_at(0.9, x, ym),
                     **{f"c50_seed{s}": v[0] for s, v in per_seed.items()},
                     **{f"c90_seed{s}": v[1] for s, v in per_seed.items()}})
    return pd.DataFrame(rows)


def waste(runs: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ZIPF:
        c90 = cs.set_index("corpus").loc[c, "c90"]
        for seed, d in run_dirs(c).items():
            final = max(int(os.path.basename(f)[3:-4]) for f in glob.glob(os.path.join(d, "nk_*.npz")))
            nk = np.load(os.path.join(d, f"nk_{final}.npz"))["n_k"].astype(np.int64)
            hit = np.unpackbits(np.load(os.path.join(d, f"hits_{final}.npz"))["hit"])[: len(nk)].astype(bool)
            total = nk.sum()
            stored = hit & (nk > 0)
            unabsorbed = nk[~stored].sum() / total
            excess = np.maximum(0, nk[stored] - c90).sum() / total if c90 else np.nan
            r = runs[(runs.corpus == c) & (runs.seed == seed) & (runs.docs == final)].iloc[0]
            srt = np.sort(nk)[::-1]
            rows.append({"corpus": c, "seed": seed, "docs": int(total), "train_tokens": int(r.train_tokens),
                         "c90_used": c90, "facts_delivered": int((nk > 0).sum()), "facts_stored_top1": int(stored.sum()),
                         "unabsorbed": unabsorbed, "excess": excess, "useful": 1 - unabsorbed - excess,
                         # per document: BOS + subject(2) + relation + object + EOS = 6 non-noise tokens
                         "frac_tokens_template_filler": 1 - 6 * total / r.train_tokens,
                         "share_docs_top1_fact": srt[0] / total, "share_docs_top100_facts": srt[:100].sum() / total})
    return pd.DataFrame(rows)


def main():
    ab = pd.read_csv(os.path.join(ROOT, "results", "absorption.csv"))
    ab = ab[ab.corpus.isin(ZIPF)]
    ab = ab[ab.docs == ab.groupby(["corpus", "seed"]).docs.transform("max")]
    cs = absorption_c(ab)
    cs.to_csv(os.path.join(ROOT, "results", "v3_absorption_c.csv"), index=False)
    runs = pd.read_csv(os.path.join(ROOT, "results", "runs.csv"))
    w = waste(runs, cs)
    w.to_csv(os.path.join(ROOT, "results", "v3_waste.csv"), index=False)
    pd.set_option("display.width", 250)
    print(cs.round(1).to_string(index=False))
    print(w.round(4).to_string(index=False))
    print(w.groupby("corpus")[["unabsorbed", "excess", "useful", "frac_tokens_template_filler",
                               "share_docs_top1_fact", "share_docs_top100_facts"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
